"""
Watch a trained agent play.

Usage:
    uv run python scripts/evaluate.py --checkpoint runs/rainbow_baseline/final.pt
    uv run python scripts/evaluate.py --checkpoint runs/rainbow_baseline/final.pt --episodes 5 --fps 6
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake.agent.rainbow import RainbowAgent
from snake.env.channels import obs_shape_config
from snake.env.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes",   type=int, default=10)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--grid",       type=int, default=12)
    p.add_argument("--egocentric", action="store_true",
                   help="Must match how the checkpoint was trained")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = SnakeEnv(grid_size=args.grid, render_mode="human", egocentric=args.egocentric)
    in_channels, n_flags = obs_shape_config(args.egocentric)
    agent = RainbowAgent(
        in_channels=in_channels,
        n_actions=int(env.action_space.n),
        grid_size=args.grid,
        v_min=-2,
        v_max=15,
        n_atoms=51,
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
            time.sleep(1.0 / args.fps)

        score = info["score"]
        scores.append(score)
        print(f"  Episode {ep:>3}: score={score}  length={info['snake_length']}  reward={ep_reward:.2f}")

    print(f"\nMean score: {sum(scores)/len(scores):.2f}  Max: {max(scores)}")
    env.close()


if __name__ == "__main__":
    main()
