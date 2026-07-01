from collections import deque
from typing import Optional

import numpy as np


# ------------------------------------------------------------------
# Sum Tree
# ------------------------------------------------------------------

class _SumTree:
    """
    Binary segment tree storing priorities in leaves and sums in internal nodes.
    O(log n) add / update / sample.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list = [None] * capacity
        self._write: int = 0
        self.n_entries: int = 0

    # ---- tree arithmetic ----

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right := left + 1, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    # ---- public ----

    def add(self, priority: float, data: object) -> None:
        leaf_idx = self._write + self.capacity - 1
        self._data[self._write] = data
        self.update(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float) -> tuple[int, float, object]:
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self._data[data_idx]


# ------------------------------------------------------------------
# N-Step Buffer
# ------------------------------------------------------------------

class NStepBuffer:
    """
    Accumulates raw transitions and emits n-step transitions.
    Call `add()` for every environment step; it returns a (possibly empty)
    list of ready n-step transitions to push into PER.
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self._buf: deque = deque()

    def add(self, transition: tuple) -> list[tuple]:
        """
        transition = (state, action, reward, next_state, done)
        Returns list of ready n-step transitions (0 or 1 normally; several on done).
        """
        self._buf.append(transition)
        _, _, _, _, done = transition

        ready = []
        if done:
            while self._buf:
                ready.append(self._make_nstep())
                self._buf.popleft()
        elif len(self._buf) >= self.n:
            ready.append(self._make_nstep())
            self._buf.popleft()

        return ready

    def _make_nstep(self) -> tuple:
        state, action, _, _, _ = self._buf[0]
        R = 0.0
        gamma_k = 1.0
        next_state = None
        done = False

        for _, _, r, ns, d in self._buf:
            R += gamma_k * r
            gamma_k *= self.gamma
            next_state = ns
            if d:
                done = True
                break

        return state, action, R, next_state, done


# ------------------------------------------------------------------
# Prioritized Experience Replay
# ------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    PER with IS-weight correction.
    alpha controls priority exponent; beta is annealed externally by Trainer.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self._tree = _SumTree(capacity)
        self._max_priority: float = 1.0
        self._rng = np.random.default_rng()

    # ---- write ----

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        priority = self._max_priority 
        self._tree.add(priority, (state, action, reward, next_state, done))

    # ---- read ----

    def sample(
        self, batch_size: int, beta: float
    ) -> tuple[tuple, np.ndarray, np.ndarray]:
        """
        Returns (transitions, leaf_indices, IS_weights).
        transitions is a tuple of arrays: states, actions, rewards, next_states, dones.
        """
        segment = self._tree.total / batch_size
        indices, priorities, raw = [], [], []

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = self._rng.uniform(lo, hi)
            idx, priority, data = self._tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            raw.append(data)

        # IS weights: w_i = (N * P(i))^{-beta} / max_j w_j
        probs = np.array(priorities, dtype=np.float64) / self._tree.total
        weights = (self._tree.n_entries * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        states      = np.stack([t[0] for t in raw])
        actions     = np.array([t[1] for t in raw], dtype=np.int64)
        rewards     = np.array([t[2] for t in raw], dtype=np.float32)
        next_states = np.stack([t[3] for t in raw])
        dones       = np.array([t[4] for t in raw], dtype=np.float32)

        return (states, actions, rewards, next_states, dones), np.array(indices), weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._tree.update(int(idx), float(p))
            self._max_priority = max(self._max_priority, float(p))

    def __len__(self) -> int:
        return self._tree.n_entries
