"""
Entry point for training (Vectorized Environments).
"""
import argparse
import sys
from pathlib import Path
import gymnasium as gym
import torch
torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake.agent.rainbow import RainbowAgent
from snake.env.channels import N_CHANNELS
from snake.env.snake_env import SnakeEnv
from snake.training.config import RainbowConfig
from snake.training.trainer import Trainer

def parse_args() -> argparse.Namespace:
    default_cfg = RainbowConfig()
    
    p = argparse.ArgumentParser()
    p.add_argument("--steps",     type=int,   default=default_cfg.total_steps)
    p.add_argument("--run-dir",   type=str,   default=default_cfg.run_dir)
    p.add_argument("--lr",        type=float, default=default_cfg.lr)
    p.add_argument("--batch",     type=int,   default=default_cfg.batch_size)
    p.add_argument("--n-step",    type=int,   default=default_cfg.n_step)
    p.add_argument("--buffer",    type=int,   default=default_cfg.buffer_capacity)
    p.add_argument("--resume",    type=str,   default=None,
                   help="Path to checkpoint .pt to resume from")
    p.add_argument("--num-envs",  type=int,   default=16,
                   help="Number of parallel environments to run")
    return p.parse_args()

def make_env(grid_size, max_steps_no_food):
    def _init():
        return SnakeEnv(grid_size=grid_size, max_steps_no_food=max_steps_no_food, render_mode=None)
    return _init

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

    print(f"Starting {args.num_envs} parallel environments...")
    envs = gym.vector.AsyncVectorEnv([
        make_env(cfg.grid_size, cfg.max_steps_no_food) for _ in range(args.num_envs)
    ])

    agent = RainbowAgent(
        in_channels=N_CHANNELS,
        n_actions=4,
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

    trainer = Trainer(agent, envs, cfg)
    trainer.train()

if __name__ == "__main__":
    main()