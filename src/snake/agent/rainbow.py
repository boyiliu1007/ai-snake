from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from snake.agent.network import RainbowNet
from snake.agent.replay_buffer import NStepBuffer, PrioritizedReplayBuffer


class RainbowAgent:
    """Rainbow DQN: Double DQN + Dueling + PER + N-step + NoisyNet."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        grid_size: int,
        # PER
        buffer_capacity: int = 100_000,
        per_alpha: float = 0.6,
        # N-step
        n_step: int = 3,
        gamma: float = 0.99,
        # Optimisation
        lr: float = 1e-4,
        # Device
        device: Optional[str] = None,
        v_min: float = -2.0,
        v_max: float = 15.0,
        n_atoms: int = 51,
        n_flags: int = 0,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_step = n_step

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        # 建立固定的 51 個分數刻度 (Support)
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)

        self.online_net = RainbowNet(in_channels, n_actions, grid_size, n_atoms, n_flags).to(self.device)
        self.target_net = RainbowNet(in_channels, n_actions, grid_size, n_atoms, n_flags).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)
        # gamma^n is the discount applied to the bootstrap value
        self._gamma_n = gamma ** n_step
        self.n_step_buf = NStepBuffer(n_step, gamma)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if evaluate:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.reset_noise()

        with torch.no_grad():
            # 🔧 取得 log 機率，轉回一般機率，然後乘上 support 算出平均 Q 值
            log_p = self.online_net(state)          # (1, n_actions, n_atoms)
            p = log_p.exp()                         
            q_values = (p * self.support).sum(dim=2) # (1, n_actions)
            
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        ready = self.n_step_buf.add((obs, action, reward, next_obs, done))
        for s, a, R, ns, d in ready:
            self.replay_buffer.add(s, a, R, ns, d)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int, beta: float) -> float:
        (states, actions, rewards, next_states, dones), indices, weights = (
            self.replay_buffer.sample(batch_size, beta)
        )
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        states      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions,     dtype=torch.int64,   device=self.device)
        rewards     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,       dtype=torch.float32, device=self.device)
        weights     = torch.tensor(weights,     dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # 1. Online Net 負責選動作 (期望值最大的動作)
            next_p = self.online_net(next_states).exp()
            next_q = (next_p * self.support).sum(dim=2)
            next_actions = next_q.argmax(dim=1)
            
            # 2. Target Net 給出該動作的機率分佈
            next_target_p = self.target_net(next_states).exp()
            next_target_p = next_target_p[range(batch_size), next_actions]
            
            # 3. 將支撐點 (Support) 依據 Reward 和 Gamma 進行平移
            Tz = rewards.unsqueeze(1) + self._gamma_n * (1.0 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            
            # 4. C51 投影演算法 (把平移後的機率，依比例分配給相鄰的兩個固定柱子)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            m = torch.zeros(batch_size, self.n_atoms, device=self.device)
            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size, device=self.device).long().unsqueeze(1)
            
            # m 陣列收集投影後的真實機率 (Target Distribution)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_target_p * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_target_p * (b - l.float())).view(-1))

        # 5. 取得當前狀態下，實際執行動作的預測 Log 機率
        log_p = self.online_net(states)
        log_p_a = log_p[range(batch_size), actions]
        
        # 6. 計算 Cross Entropy Loss (KL 散度)
        td_errors = -(m * log_p_a).sum(dim=1)
        loss = (weights * td_errors).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        return float(loss.item())

    def update_target(self, tau: float = 0.005) -> None:
        """Soft update of the target network."""
        for online_p, target_p in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
