"""
Watch a trained agent play.

Usage:
    uv run python scripts/evaluate.py --checkpoint runs/rainbow_baseline/final.pt
    uv run python scripts/evaluate.py --checkpoint runs/rainbow_baseline/final.pt --episodes 5 --fps 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake.agent.rainbow import RainbowAgent
from snake.training.config import RainbowConfig
from snake.env.channels import obs_shape_config
from snake.env.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes",   type=int, default=10)
    p.add_argument("--fps",        type=int, default=60)
    p.add_argument("--grid",       type=int, default=12)
    p.add_argument("--egocentric", action="store_true",
                   help="Must match how the checkpoint was trained")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RainbowConfig(egocentric=args.egocentric)

    env = SnakeEnv(grid_size=args.grid, render_mode="human", egocentric=args.egocentric, render_fps=args.fps)
    in_channels, n_flags = obs_shape_config(args.egocentric)
    agent = RainbowAgent(
        in_channels=in_channels,
        n_actions=int(env.action_space.n),
        grid_size=args.grid,
        buffer_capacity=cfg.buffer_capacity,
        per_alpha=cfg.per_alpha,
        n_step=cfg.n_step,
        gamma=cfg.gamma,
        lr=cfg.lr,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
        n_atoms=cfg.n_atoms,
        n_flags=n_flags,
    )
    agent.load(args.checkpoint)
    print(f"Loaded {args.checkpoint}  |  device: {agent.device}")

    scores = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        score = info["score"]
        scores.append(score)
        print(f"  Episode {ep:>3}: score={score}  length={info['snake_length']}  reward={ep_reward:.2f}")

    print(f"\nMean score: {sum(scores)/len(scores):.2f}  Max: {max(scores)}")
    env.close()


if __name__ == "__main__":
    main()
