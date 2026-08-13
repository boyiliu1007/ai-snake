from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from snake.agent.network import RainbowNet
from snake.agent.replay_buffer import NStepBuffer, PrioritizedReplayBuffer


class RainbowAgent:
    """Rainbow DQN Agent integrating Double DQN, Dueling, PER, N-step, NoisyNet, and C51."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        grid_size: int,
        buffer_capacity: int,
        per_alpha: float,
        n_step: int,
        gamma: float,
        lr: float,
        v_min: float,
        v_max: float,
        n_atoms: int,
        device: Optional[str] = None,
        n_flags: int = 0,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_step = n_step

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # C51 Distributional RL setup
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)

        # Networks
        self.online_net = RainbowNet(in_channels, n_actions, grid_size, n_atoms, n_flags).to(self.device)
        self.target_net = RainbowNet(in_channels, n_actions, grid_size, n_atoms, n_flags).to(self.device)
        
        self.online_net = torch.compile(self.online_net)
        self.target_net = torch.compile(self.target_net)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)
        self._gamma_n = gamma ** n_step
        self.n_step_buf = NStepBuffer(n_step, gamma)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        """Selects an action given the current observation."""
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if evaluate:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.reset_noise()

        with torch.no_grad():
            log_p = self.online_net(state)           # (1, n_actions, n_atoms)
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
        """Stores a transition in the N-step buffer and subsequently the PER buffer."""
        ready = self.n_step_buf.add((obs, action, reward, next_obs, done))
        for s, a, R, ns, d in ready:
            self.replay_buffer.add(s, a, R, ns, d)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def train_step(self, batch_size: int, beta: float) -> float:
        """Performs a single gradient update step using C51 distributional loss."""
        (states, actions, rewards, next_states, dones), indices, weights = (
            self.replay_buffer.sample(batch_size, beta)
        )

        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # Convert to tensors
        states      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions,     dtype=torch.int64,   device=self.device)
        rewards     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,       dtype=torch.float32, device=self.device)
        weights     = torch.tensor(weights,     dtype=torch.float32, device=self.device)

        # Use Automatic Mixed Precision (AMP) for faster training on supported GPUs
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                # ---- 1. Double DQN: Select next action using Online Net ----
                next_p = self.online_net(next_states).float().exp()
                next_q = (next_p * self.support).sum(dim=2)
                next_actions = next_q.argmax(dim=1)
                
                # ---- 2. Evaluate selected action using Target Net ----
                next_target_p = self.target_net(next_states).float().exp()
                next_target_p = next_target_p[range(batch_size), next_actions]
                
                # ---- 3. Compute Projected Distribution (Bellman Update) ----
                Tz = rewards.unsqueeze(1) + self._gamma_n * (1.0 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
                Tz = Tz.clamp(min=self.v_min, max=self.v_max)
                
                b = (Tz - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Distribute probabilities into the support atoms
                m = torch.zeros(batch_size, self.n_atoms, device=self.device, dtype=torch.float32)
                offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size, device=self.device).long().unsqueeze(1)
                
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_target_p * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_target_p * (b - l.float())).view(-1))

            # ---- 4. Compute Cross-Entropy Loss ----
            log_p = self.online_net(states).float()
            log_p_a = log_p[range(batch_size), actions]
            
            td_errors = -(m * log_p_a).sum(dim=1)
            loss = (weights * td_errors).mean()

        # ---- 5. Gradient Descent ----
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- 6. Update PER Priorities ----
        self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

        return float(loss.item())

    def update_target(self, tau: float = 0.005) -> None:
        """Soft update of the target network (Polyak averaging)."""
        for online_p, target_p in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Saves the agent's state dicts to a file."""
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
        """Loads the agent's state dicts from a file."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])