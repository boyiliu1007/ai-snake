"""
Quick sanity check: runs 50 random steps, asserts obs shape and value range.
Pass render_mode="human" to visually verify the Pygame window.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from snake.env.snake_env import SnakeEnv
from snake.env.channels import N_CHANNELS

RENDER = "--render" in sys.argv
GRID = 8
STEPS = 50


def main():
    env = SnakeEnv(grid_size=GRID, render_mode="human" if RENDER else None)
    obs, info = env.reset(seed=42)

    assert obs.shape == (N_CHANNELS, GRID, GRID), f"Bad obs shape: {obs.shape}"
    assert obs.dtype == np.float32, f"Bad dtype: {obs.dtype}"

    for step in range(STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (N_CHANNELS, GRID, GRID), f"Step {step}: bad shape"
        assert obs.min() >= 0.0, f"Step {step}: negative value"
        assert obs.max() <= 1.0, f"Step {step}: value > 1"
        assert isinstance(reward, float), f"Step {step}: reward not float"

        if RENDER:
            time.sleep(0.1)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"Sanity check passed. {N_CHANNELS}-channel obs shape & value range OK.")


if __name__ == "__main__":
    main()
