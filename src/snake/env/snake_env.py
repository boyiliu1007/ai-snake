from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from snake.env.game import SnakeGame, GRID_SIZE
from snake.env.channels import build_channels, N_CHANNELS, _head_channel, _body_gradient


class SnakeEnv(gym.Env):
    """
    Gymnasium environment for Snake.
    Observation: float32 array (N_CHANNELS, GRID_SIZE, GRID_SIZE).
    Action:      Discrete(4)  — 0=Up, 1=Down, 2=Left, 3=Right.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        max_steps_no_food: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_CHANNELS, grid_size, grid_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._game = SnakeGame(grid_size=grid_size, max_steps_no_food=max_steps_no_food)
        self._prev_head_ch: Optional[np.ndarray] = None
        self._prev_body_grad_ch: Optional[np.ndarray] = None

        self._renderer = None
        self._last_reward: float = 0.0
        self._episode_reward: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._game.reset(seed=seed)

        self._prev_head_ch = None
        self._prev_body_grad_ch = None
        self._last_reward = 0.0
        self._episode_reward = 0.0

        obs = self._get_obs()
        return obs, self._game._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Snapshot current head/body channels for the t-1 slots before advancing.
        # Only these two channels are needed here, so skip the expensive flood-fill BFS.
        self._prev_head_ch = _head_channel(self._game.snake, self.grid_size)
        self._prev_body_grad_ch = _body_gradient(self._game.snake, self.grid_size)

        reward, terminated, truncated, info = self._game.step(int(action))

        obs = self._get_obs()
        # Bonus for keeping reachable area open — encourages coiling over space-wasting paths.
        # Only on non-terminal steps so it doesn't interfere with death/truncation penalties.
        if not terminated and not truncated:
            free_cells = self.grid_size * self.grid_size - len(self._game.snake)
            if free_cells > 0:
                reachable_fraction = obs[4].sum() / free_cells
                reward += 0.005 * reachable_fraction

        self._last_reward = reward
        self._episode_reward += reward
        info["episode_reward"] = self._episode_reward

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._renderer is None:
            from snake.rendering.renderer import SnakeRenderer
            self._renderer = SnakeRenderer(self.grid_size)

        self._renderer.render(
            snake=self._game.snake,
            food=self._game.food,
            score=self._game.score,
            step=self._game.steps,
            reward=self._last_reward,
            episode_reward=self._episode_reward,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return build_channels(
            snake=self._game.snake,
            food=self._game.food,
            grid_size=self.grid_size,
            prev_head_ch=self._prev_head_ch,
            prev_body_grad_ch=self._prev_body_grad_ch,
        )
