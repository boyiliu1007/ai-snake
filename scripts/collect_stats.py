"""
Run a trained checkpoint through N deterministic evaluation episodes and
report the numbers: board-clear (full-grid) success rate, score distribution, and episode length.

Usage:
    python scripts/collect_stats.py --checkpoint runs/20M_ultimate_10x10/final.pt
    python scripts/collect_stats.py --checkpoint <ckpt>.pt --episodes 200 --grid 10 --seed 0
    python scripts/collect_stats.py --checkpoint <ckpt>.pt --train-steps 20000000 --out results/10x10.json

Notes:
- "evaluate=True" puts NoisyNet layers in eval mode (mean weights, no
  sampled noise) and picks the greedy argmax action every step — same
  deterministic policy scripts/evaluate.py renders, just without Pygame.
- v_min/v_max/n_atoms/n_step/gamma must match the values the checkpoint was
  trained with (only the C51 support and n_step affect inference; they're
  read from RainbowConfig by default so this stays in sync with training).
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake.agent.rainbow import RainbowAgent
from snake.env.channels import obs_shape_config
from snake.env.snake_env import SnakeEnv
from snake.training.config import RainbowConfig


def parse_args() -> argparse.Namespace:
    cfg = RainbowConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--grid", type=int, default=cfg.grid_size)
    p.add_argument("--egocentric", action="store_true",
                    help="Must match how the checkpoint was trained")
    p.add_argument("--seed", type=int, default=0,
                    help="Episode i is reset with seed + i, for reproducible runs")
    p.add_argument("--v-min", type=float, default=cfg.v_min)
    p.add_argument("--v-max", type=float, default=cfg.v_max)
    p.add_argument("--n-atoms", type=int, default=cfg.n_atoms)
    p.add_argument("--train-steps", type=int, default=None,
                    help="Env steps the checkpoint was trained for, for the report. "
                         "Auto-parsed from the filename (e.g. checkpoint_00050016.pt) if omitted.")
    p.add_argument("--out", type=str, default=None,
                    help="Write raw per-episode results + summary to this JSON path")
    return p.parse_args()


def infer_train_steps(checkpoint: str) -> Optional[int]:
    match = re.search(r"(\d{5,})", Path(checkpoint).stem)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()

    env = SnakeEnv(grid_size=args.grid, render_mode=None, egocentric=args.egocentric)
    in_channels, n_flags = obs_shape_config(args.egocentric)
    agent = RainbowAgent(
        in_channels=in_channels,
        n_actions=int(env.action_space.n),
        grid_size=args.grid,
        v_min=args.v_min,
        v_max=args.v_max,
        n_atoms=args.n_atoms,
        n_flags=n_flags,
    )
    agent.load(args.checkpoint)
    print(f"Loaded {args.checkpoint}  |  device: {agent.device}")

    full_board = args.grid * args.grid
    episodes = []

    t0 = time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        solved = info["snake_length"] >= full_board
        episodes.append({
            "episode": ep,
            "score": int(info["score"]),
            "snake_length": int(info["snake_length"]),
            "steps": ep_steps,
            "reward": float(ep_reward),
            "solved": bool(solved),
        })
        flag = "SOLVED" if solved else ""
        print(f"  ep {ep:>4}: score={info['score']:>3}  length={info['snake_length']:>3}  "
              f"steps={ep_steps:>4}  reward={ep_reward:>7.2f}  {flag}")

    elapsed = time.time() - t0
    env.close()

    scores = [e["score"] for e in episodes]
    lengths = [e["snake_length"] for e in episodes]
    steps = [e["steps"] for e in episodes]
    solved_flags = [e["solved"] for e in episodes]
    n_solved = sum(solved_flags)

    summary = {
        "checkpoint": args.checkpoint,
        "grid_size": args.grid,
        "n_episodes": args.episodes,
        "train_steps": args.train_steps or infer_train_steps(args.checkpoint),
        "success_rate_pct": 100.0 * n_solved / args.episodes,
        "n_solved": n_solved,
        "mean_score": statistics.mean(scores),
        "std_score": statistics.pstdev(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "mean_length": statistics.mean(lengths),
        "mean_steps": statistics.mean(steps),
        "eval_wall_time_sec": elapsed,
    }

    print("\n" + "=" * 60)
    print(f"Episodes:        {args.episodes}")
    print(f"Board size:      {args.grid}x{args.grid} ({full_board} cells)")
    print(f"Success rate:    {summary['success_rate_pct']:.1f}%  ({n_solved}/{args.episodes} full clears)")
    print(f"Mean score:      {summary['mean_score']:.2f} +/- {summary['std_score']:.2f}  "
          f"(min {summary['min_score']}, max {summary['max_score']})")
    print(f"Mean length:     {summary['mean_length']:.1f}")
    print(f"Mean steps/ep:   {summary['mean_steps']:.1f}")
    if summary["train_steps"]:
        print(f"Train steps:     {summary['train_steps']:,}")
    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2))
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
