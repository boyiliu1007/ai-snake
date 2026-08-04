from typing import TYPE_CHECKING

import numpy as np

from snake.env.snake_env import SnakeEnv

if TYPE_CHECKING:
    from snake.agent.rainbow import RainbowAgent


class Evaluator:
    def __init__(
        self,
        agent: "RainbowAgent",
        grid_size: int,
        n_episodes: int = 10,
        max_steps_no_food: int = None,
        egocentric: bool = False,
    ):
        self.agent = agent
        self.n_episodes = n_episodes
        # Separate env with no rendering so eval doesn't affect training display.
        # Match the training truncation limit and obs format so eval is comparable.
        self._env = SnakeEnv(
            grid_size=grid_size,
            max_steps_no_food=max_steps_no_food,
            render_mode=None,
            egocentric=egocentric,
        )

    def evaluate(self) -> dict:
        lengths, scores, rewards = [], [], []

        for _ in range(self.n_episodes):
            obs, _ = self._env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                ep_reward += reward

            lengths.append(info["snake_length"])
            scores.append(info["score"])
            rewards.append(ep_reward)

        return {
            "mean_length": float(np.mean(lengths)),
            "mean_score":  float(np.mean(scores)),
            "mean_reward": float(np.mean(rewards)),
            "max_score":   int(np.max(scores)),
        }
