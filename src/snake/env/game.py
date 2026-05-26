from collections import deque
from typing import Optional
import random

GRID_SIZE = 12

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeGame:
    """
    Pure Snake game logic — no gymnasium, no torch.
    Snake deque: index 0 = head, index -1 = tail.
    """

    def __init__(self, grid_size: int = GRID_SIZE, seed: Optional[int] = None):
        self.grid_size = grid_size
        self._rng = random.Random(seed)

        self.snake: deque[tuple[int, int]] = deque()
        self.food: tuple[int, int] = (0, 0)
        self.direction: int = RIGHT
        self.steps: int = 0
        self.steps_since_meal: int = 0
        self.score: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.direction = RIGHT
        self.steps = 0
        self.steps_since_meal = 0
        self.score = 0
        self._done = False
        self._place_food()

    def step(self, action: int) -> tuple[float, bool, dict]:
        """
        Returns (reward, terminated, info).
        Silently ignores reversal attempts (keeps current direction).
        """
        if action == _OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action

        dr, dc = _DELTA[action]
        hr, hc = self.snake[0]
        new_head = (hr + dr, hc + dc)

        self.steps += 1
        self.steps_since_meal += 1

        # Wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self._done = True
            return -10.0, True, self._info()

        # Self-collision — tail moves away this step so exclude it
        if new_head in set(list(self.snake)[:-1]):
            self._done = True
            return -10.0, True, self._info()

        self.snake.appendleft(new_head)

        ate_food = new_head == self.food
        if ate_food:
            self.score += 1
            self.steps_since_meal = 0
            self._place_food()
        else:
            self.snake.pop()

        reward = self._step_reward(ate_food)
        return reward, False, self._info()

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_food(self) -> None:
        snake_set = set(self.snake)
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in snake_set
        ]
        if empty:
            self.food = self._rng.choice(empty)

    def _step_reward(self, ate_food: bool) -> float:
        length = len(self.snake)
        head = self.snake[0]
        dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        if ate_food:
            return min(0.5 * length, 10.0)

        return (
            -0.05 * (dist / length)
            - 0.1 * (self.steps_since_meal / length)
        )

    def _info(self) -> dict:
        return {
            "snake_length": len(self.snake),
            "score": self.score,
            "steps": self.steps,
        }
