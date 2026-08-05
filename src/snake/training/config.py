from dataclasses import dataclass
from pathlib import Path

@dataclass
class RainbowConfig:
    """Hyperparameters and configuration for the Rainbow DQN agent."""
    
    # ---- Environment ----
    grid_size: int = 10
    max_steps_no_food: int = grid_size ** 2 * 2  # Truncates circling episodes
    egocentric: bool = False                     # Rotate obs to heading-up + 3 relative actions

    # ---- Training Schedule ----
    total_steps: int = 20_000_000
    warmup_steps: int = 10_000                   # Steps before first gradient update
    batch_size: int = 512
    train_freq: int = 1                          # Gradient steps per env step

    # ---- Network ----
    lr: float = 1e-4
    lr_end: float = 1e-5
    lr_decay_start: int = 10_000_000             # Start decaying learning rate after this many env steps

    # ---- RL Parameters ----
    gamma: float = 0.99
    n_step: int = 5
    v_min: float = -10.0                         # Minimum expected return for C51
    v_max: float = 150.0                         # Maximum expected return for C51
    n_atoms: int = 51                            # Number of discrete atoms for C51 distribution

    # ---- Prioritized Experience Replay (PER) ----
    buffer_capacity: int = 2_000_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 15_000_000             # Anneal beta linearly over this many env steps

    # ---- Target Network ----
    target_update_freq: int = 5_000              # Hard target network update every N env steps

    # ---- Logging & Checkpointing ----
    log_freq: int = 1_000                        # Log metrics to TensorBoard every N steps
    eval_freq: int = 25_000                      # Run greedy evaluation every N steps
    eval_episodes: int = 10
    checkpoint_freq: int = 50_000
    run_dir: str = "runs/20M_ultimate_10x10"      # Save directory for checkpoints and logs

    def beta(self, step: int) -> float:
        """Calculate the linearly annealed beta value for PER IS-weights."""
        frac = min(step / self.per_beta_steps, 1.0)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def learning_rate(self, step: int) -> float:
        """Calculate the learning rate with a delayed linear decay."""
        if step <= self.lr_decay_start:
            return self.lr
            
        decay_duration = max(1, self.total_steps - self.lr_decay_start)
        frac = min((step - self.lr_decay_start) / decay_duration, 1.0)
        return self.lr + frac * (self.lr_end - self.lr)
        
    @property
    def run_path(self) -> Path:
        """Get the Path object representing the run directory."""
        return Path(self.run_dir)