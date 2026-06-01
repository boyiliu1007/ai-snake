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

        self.online_net = RainbowNet(in_channels, n_actions, grid_size).to(self.device)
        self.target_net = RainbowNet(in_channels, n_actions, grid_size).to(self.device)
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
        
        # Eval mode: disable noise and use mean weights; train mode: resample noise
        if evaluate:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.reset_noise()

        with torch.no_grad():
            q = self.online_net(state)
        return int(q.argmax(dim=1).item())

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
        """One gradient step. Returns scalar loss."""
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

        # Double DQN target: online net selects action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets      = rewards + self._gamma_n * (1.0 - dones) * next_q

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors = targets - current_q
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        return float(loss.item())

    def update_target(self, tau: float = 0.005) -> None:
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
