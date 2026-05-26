import torch
import torch.nn as nn


class _CNN(nn.Module):
    """
    3-layer CNN tuned for 12×12 grids.
    3×3 kernels with padding preserve spatial dims throughout,
    avoiding the over-pooling that Nature-DQN's strides cause on small grids.
    """

    def __init__(self, in_channels: int, grid_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size * grid_size, 512),
            nn.ReLU(),
        )
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RainbowNet(nn.Module):
    """
    CNN feature extractor + Dueling head.

    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

    Dueling separates state-value estimation from action-advantage,
    which stabilises learning when many actions have similar values
    (common in Snake when the snake isn't near food or walls).
    """

    def __init__(self, in_channels: int, n_actions: int, grid_size: int):
        super().__init__()
        self.cnn = _CNN(in_channels, grid_size)
        feat = self.cnn.out_dim

        self.value_stream = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        value = self.value_stream(features)          # (B, 1)
        advantage = self.advantage_stream(features)  # (B, n_actions)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
