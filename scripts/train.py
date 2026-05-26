"""
Entry point for training.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --steps 500000 --run-dir runs/exp_01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake.agent.rainbow import RainbowAgent
from snake.env.channels import N_CHANNELS
from snake.env.snake_env import SnakeEnv
from snake.training.config import RainbowConfig
from snake.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",     type=int,   default=1_000_000)
    p.add_argument("--run-dir",   type=str,   default="runs/rainbow_baseline")
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--n-step",    type=int,   default=3)
    p.add_argument("--buffer",    type=int,   default=100_000)
    p.add_argument("--resume",    type=str,   default=None,
                   help="Path to checkpoint .pt to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = RainbowConfig(
        total_steps=args.steps,
        run_dir=args.run_dir,
        lr=args.lr,
        batch_size=args.batch,
        n_step=args.n_step,
        buffer_capacity=args.buffer,
    )

    env = SnakeEnv(grid_size=cfg.grid_size, render_mode=None)

    agent = RainbowAgent(
        in_channels=N_CHANNELS,
        n_actions=int(env.action_space.n),
        grid_size=cfg.grid_size,
        buffer_capacity=cfg.buffer_capacity,
        per_alpha=cfg.per_alpha,
        n_step=cfg.n_step,
        gamma=cfg.gamma,
        lr=cfg.lr,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    trainer = Trainer(agent, env, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
