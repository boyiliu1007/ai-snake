from collections import deque
from typing import Optional

import numpy as np

N_CHANNELS = 7


def obs_shape_config(egocentric: bool) -> tuple[int, int]:
    """
    Return (n_spatial, n_flags) for the given mode.

    Absolute mode: all 7 spatial channels, no flags.
    Egocentric mode: drop the two t-1 channels (5, 6) — redundant once the board
    is rotated heading-up — and add 3 heading-relative danger-flag planes.
    """
    if egocentric:
        return 5, 3
    return N_CHANNELS, 0


def _head_channel(snake: deque, grid_size: int) -> np.ndarray:
    ch = np.zeros((grid_size, grid_size), dtype=np.float32)
    r, c = snake[0]
    ch[r, c] = 1.0
    return ch


def _body_gradient(snake: deque, grid_size: int) -> np.ndarray:
    """
    Encodes snake age per cell: neck = (n-1)/n ≈ 1.0, tail = 1/n ≈ 0.
    This lets the agent infer which body segments disappear soonest.
    """
    ch = np.zeros((grid_size, grid_size), dtype=np.float32)
    n = len(snake)
    for i, (r, c) in enumerate(snake):
        ch[r, c] = 0.15 + 0.85 * (n - i) / n  # head=1.0, tail≥0.15 regardless of length
    return ch


def _food_channel(food: tuple, grid_size: int) -> np.ndarray:
    ch = np.zeros((grid_size, grid_size), dtype=np.float32)
    ch[food[0], food[1]] = 1.0
    return ch


def _danger_map(snake: deque, grid_size: int) -> np.ndarray:
    """
    Marks cells that cause immediate death on entry.
    Body cells (tail excluded — it moves away) are marked 1.
    Wall-adjacent cells on grid boundary are marked 0.5 to distinguish.
    """
    ch = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Border proximity signal: cells on the grid edge can't move further
    ch[0, :] = 0.5
    ch[-1, :] = 0.5
    ch[:, 0] = 0.5
    ch[:, -1] = 0.5

    # Body collision (tail excluded — it moves away this step)
    body_without_tail = list(snake)[:-1]
    for r, c in body_without_tail:
        ch[r, c] = 1.0

    return ch


def _flood_fill(snake: deque, grid_size: int) -> np.ndarray:
    """
    BFS reachability from head through cells not blocked by body (tail excluded).
    1 = reachable, 0 = cut off.
    Gives the agent a spatial signal for self-entrapment.
    """
    blocked = set(list(snake)[:-1])  # tail moves away

    visited = np.zeros((grid_size, grid_size), dtype=np.float32)
    head = snake[0]
    queue: deque[tuple[int, int]] = deque([head])
    visited[head[0], head[1]] = 1.0

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid_size
                and 0 <= nc < grid_size
                and visited[nr, nc] == 0.0
                and (nr, nc) not in blocked
            ):
                visited[nr, nc] = 1.0
                queue.append((nr, nc))

    return visited


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def to_egocentric(obs: np.ndarray, direction: int) -> np.ndarray:
    """
    Rotate a (C, H, W) observation so the snake's heading always points 'up'.
    All channels rotate together, preserving spatial relationships. This exploits
    Snake's rotation symmetry so the network sees a single canonical orientation.
    """
    from snake.env.game import EGOCENTRIC_ROT
    return np.rot90(obs, k=EGOCENTRIC_ROT[direction], axes=(1, 2)).copy()


def build_channels(
    snake: deque,
    food: tuple,
    grid_size: int,
    prev_head_ch: Optional[np.ndarray] = None,
    prev_body_grad_ch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns float32 array of shape (7, grid_size, grid_size).

    Ch 0: head binary
    Ch 1: body gradient (neck≈1, tail≈0)
    Ch 2: food binary
    Ch 3: danger map (body=1, wall-edge=0.5)
    Ch 4: flood-fill reachability from head
    Ch 5: head at t-1
    Ch 6: body gradient at t-1
    """
    zeros = np.zeros((grid_size, grid_size), dtype=np.float32)

    return np.stack(
        [
            _head_channel(snake, grid_size),                          # Ch 0
            _body_gradient(snake, grid_size),                         # Ch 1
            _food_channel(food, grid_size),                           # Ch 2
            _danger_map(snake, grid_size),                            # Ch 3
            _flood_fill(snake, grid_size),                            # Ch 4
            prev_head_ch if prev_head_ch is not None else zeros,      # Ch 5
            prev_body_grad_ch if prev_body_grad_ch is not None else zeros,  # Ch 6
        ],
        axis=0,
    )
