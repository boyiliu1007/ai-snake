from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RainbowConfig:
    # ---- Environment ----
    grid_size: int = 12
    max_steps_no_food: int = 144        # grid_size² — truncates circling episodes

    # ---- Training schedule ----
    total_steps: int = 3_000_000
    warmup_steps: int = 10_000        # steps before first gradient update
    batch_size: int = 128
    train_freq: int = 1              # gradient steps per env step

    # ---- Network ----
    lr: float = 1e-4
    lr_end: float = 1e-5
    # ---- RL ----
    gamma: float = 0.99
    n_step: int = 5

    # ---- PER ----
    buffer_capacity: int = 500_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 3_000_000    # anneal beta over this many env steps

    # ---- Exploration (ε-greedy) ----
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_500_000

    # ---- Target network ----
    target_update_freq: int = 5_000  # hard update every N env steps

    # ---- Logging / checkpointing ----
    log_freq: int = 1_000            # log to TensorBoard every N steps
    eval_freq: int = 25_000          # run evaluator every N steps
    eval_episodes: int = 10
    checkpoint_freq: int = 50_000
    run_dir: str = "runs/full_rainbow"

    def epsilon(self, step: int) -> float:
        frac = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def beta(self, step: int) -> float:
        frac = min(step / self.per_beta_steps, 1.0)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def learning_rate(self, step: int) -> float:
        decay_start = 1_000_000
        if step <= decay_start:
            return self.lr
        decay_duration = max(1, self.total_steps - decay_start)
        frac = min((step - decay_start) / decay_duration, 1.0)
        return self.lr + frac * (self.lr_end - self.lr)
    
    @property
    def run_path(self) -> Path:
        return Path(self.run_dir)
