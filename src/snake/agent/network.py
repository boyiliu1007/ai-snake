import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNN(nn.Module):
    """Lightweight CNN feature extractor with stride-2 downsampling."""
    def __init__(self, in_channels: int, grid_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding_mode='replicate', padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding_mode='replicate', padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding_mode='replicate', padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding_mode='replicate', stride=1),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
        )
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyLinear(nn.Module):
    """
    Factorized NoisyLinear layer for Rainbow DQN.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters: mean (mu) and standard deviation (sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        # Generate on the same device as the buffers to avoid CPU→GPU copies each step
        x = torch.randn(size, device=self.weight_epsilon.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        # Resample noise (call before each env step or loss computation)
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training mode uses noisy weights; eval mode uses mean (mu) only
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNet(nn.Module):
    """CNN + Dueling NoisyLinear heads + C51 distributional output.

    If ``n_flags > 0`` the input carries that many extra constant "danger-flag"
    planes appended after the spatial channels. They bypass the CNN and are
    concatenated straight onto the flattened features, so they reach the dueling
    heads directly (aligned with the action outputs) instead of being smeared
    through the convolutions.
    """

    def __init__(self, in_channels: int, n_actions: int, grid_size: int,
                 n_atoms: int, n_flags: int = 0):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.spatial_channels = in_channels
        self.n_flags = n_flags

        self.cnn = _CNN(in_channels, grid_size)
        feat = self.cnn.out_dim + n_flags  # flags concatenated post-CNN

        self.value_stream = nn.Sequential(
            NoisyLinear(feat, 256),
            nn.ReLU(),
            NoisyLinear(256, n_atoms),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feat, 256),
            nn.ReLU(),
            NoisyLinear(256, n_actions * n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        if self.n_flags > 0:
            spatial = x[:, : self.spatial_channels]
            # Flag planes are constant, so any cell holds the value — read a corner.
            flags = x[:, self.spatial_channels :, 0, 0]      # (B, n_flags)
            features = self.cnn(spatial)
            features = torch.cat([features, flags], dim=1)
        else:
            features = self.cnn(x)

        value = self.value_stream(features).view(batch_size, 1, self.n_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.n_actions, self.n_atoms)

        # Dueling combination, then log-probabilities over the C51 atoms
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.log_softmax(q_atoms, dim=2)

    def reset_noise(self) -> None:
        """Resets noise for all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()