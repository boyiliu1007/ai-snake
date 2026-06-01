import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNN(nn.Module):
    """
    維持我們上次修改的精簡版 CNN (stride=2 降維)
    """
    def __init__(self, in_channels: int, grid_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (grid_size // 2) * (grid_size // 2), 256),
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

        # 可學習的參數：均值 (mu) 與 標準差 (sigma)
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
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        # 重製雜訊 (每次環境 step 或計算 loss 前都要呼叫)
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 訓練模式(互動與學習)使用雜訊，測試模式(evaluator)只使用均值(mu)
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNet(nn.Module):
    """
    CNN feature extractor + Dueling head using NoisyLinear.
    """
    def __init__(self, in_channels: int, n_actions: int, grid_size: int):
        super().__init__()
        self.cnn = _CNN(in_channels, grid_size)
        feat = self.cnn.out_dim

        # 🔧 將最後兩層全部換成 NoisyLinear
        self.value_stream = nn.Sequential(
            NoisyLinear(feat, 256),
            nn.ReLU(),
            NoisyLinear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feat, 256),
            nn.ReLU(),
            NoisyLinear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        """Resets noise for all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()