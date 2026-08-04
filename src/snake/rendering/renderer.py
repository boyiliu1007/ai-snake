from collections import deque

import pygame


class SnakeRenderer:
    CELL = 50            # pixels per cell
    STATS_H = 60         # pixel height of the stats bar

    _COLORS = {
        "bg":         (10,  10,  10),
        "grid":       (28,  28,  28),
        "food":       (220, 60,  60),
        "head":       (60,  180, 240),
        "body_start": (60,  180, 240),   # gradient start, just behind the head (blue)
        "body_end":   (80,  220, 120),   # gradient end, at the tail (green)
        "text":       (220, 220, 220),
        "label":      (140, 140, 140),
    }

    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.w = grid_size * self.CELL
        self.h = grid_size * self.CELL + self.STATS_H

        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("AI Snake — Rainbow DQN")
        self.font_sm = pygame.font.SysFont("monospace", 14)
        self.font_md = pygame.font.SysFont("monospace", 17, bold=True)
        self.clock = pygame.time.Clock()

    def render(
        self,
        snake: deque,
        food: tuple,
        score: int,
        step: int,
        reward: float,
        episode_reward: float,
        fps: int = 10,
    ) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self.screen.fill(self._COLORS["bg"])
        self._draw_grid()
        self._draw_food(food)
        self._draw_snake(snake)
        self._draw_stats(score, step, reward, episode_reward, len(snake))
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cell_rect(self, r: int, c: int, margin: int = 2) -> pygame.Rect:
        return pygame.Rect(
            c * self.CELL + margin,
            r * self.CELL + margin,
            self.CELL - 2 * margin,
            self.CELL - 2 * margin,
        )

    def _draw_grid(self) -> None:
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.CELL, r * self.CELL, self.CELL, self.CELL)
                pygame.draw.rect(self.screen, self._COLORS["grid"], rect, 1)

    def _draw_food(self, food: tuple) -> None:
        fr, fc = food
        rect = pygame.Rect(
            fc * self.CELL + 8, fr * self.CELL + 8,
            self.CELL - 16, self.CELL - 16,
        )
        pygame.draw.ellipse(self.screen, self._COLORS["food"], rect)

    def _draw_snake(self, snake: deque) -> None:
        n_body = len(snake) - 1
        for i, (r, c) in enumerate(snake):
            if i == 0:
                color = self._COLORS["head"]
            else:
                t = (i - 1) / n_body if n_body > 1 else 0.0
                color = self._lerp_color(self._COLORS["body_start"], self._COLORS["body_end"], t)
            rect = self._cell_rect(r, c, margin=3)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

    @staticmethod
    def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def _draw_stats(
        self,
        score: int,
        step: int,
        reward: float,
        episode_reward: float,
        length: int,
    ) -> None:
        y0 = self.grid_size * self.CELL + 8
        items = [
            ("SCORE", str(score)),
            ("LENGTH", str(length)),
            ("STEP", str(step)),
            ("REWARD", f"{reward:.3f}"),
            ("EP RWD", f"{episode_reward:.2f}"),
        ]
        col_w = self.w // len(items)
        for i, (label, value) in enumerate(items):
            x = i * col_w + 6
            self.screen.blit(self.font_sm.render(label, True, self._COLORS["label"]), (x, y0))
            self.screen.blit(self.font_md.render(value, True, self._COLORS["text"]), (x, y0 + 18))
