# AI Snake — Rainbow DQN

A Snake agent trained with Rainbow DQN (Double DQN + Dueling Network + Prioritized Experience Replay + N-step returns) on a 12×12 grid.

## Observation space — 7 channels

| Ch | Description |
|----|-------------|
| 0 | Snake head (binary) |
| 1 | Body gradient — neck ≈ 1.0, tail ≈ 0 (tells the agent which segments disappear soonest) |
| 2 | Food (binary) |
| 3 | Danger map — body cells = 1.0, wall-edge cells = 0.5 |
| 4 | Flood-fill reachability from head — 1 = reachable, 0 = cut off |
| 5 | Head at t−1 |
| 6 | Body gradient at t−1 |

---

## Installation

### macOS (Apple Silicon or Intel)

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd ai-snake
uv sync
uv pip install -e .
```

### Windows with NVIDIA GPU

Requires Python 3.9+ and pip. Install the CUDA build of PyTorch first, then the rest:

```powershell
git clone <repo-url>
cd ai-snake

python -m venv .venv
.venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium pygame tensorboard
pip install -e .
```

> If your CUDA version is not 12.1, replace `cu121` with the right tag from
> https://pytorch.org/get-started/locally/

---

## Usage

### 1. Verify the environment

```bash
# headless
python scripts/sanity_check.py

# with Pygame window
python scripts/sanity_check.py --render
```

### 2. Train

```bash
python scripts/train.py
```

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 1 000 000 | Total environment steps |
| `--run-dir` | `runs/rainbow_baseline` | Where checkpoints and TensorBoard logs are saved |
| `--lr` | 1e-4 | Learning rate |
| `--batch` | 32 | Batch size |
| `--n-step` | 3 | N-step return length |
| `--resume` | — | Path to a checkpoint `.pt` to continue from |

Resume after interruption:

```bash
python scripts/train.py --resume runs/rainbow_baseline/interrupted_00025000.pt
```

Training prints a line every 1 000 steps:

```
step   25,000 | ep  1,345 | loss 1.517 | ε 0.881 | best_score 0 | 120 sps
```

Checkpoints are saved every 50 000 steps and on Ctrl+C.

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir runs/
```

Key metrics:

| Tag | Meaning |
|-----|---------|
| `episode/score` | Food eaten per episode |
| `episode/length` | Snake length at death |
| `train/loss` | TD loss |
| `eval/mean_score` | Greedy policy score (10-episode average, logged every 25k steps) |

### 4. Watch a trained agent

```bash
python scripts/evaluate.py --checkpoint runs/rainbow_baseline/final.pt
```

Options: `--episodes 10`, `--fps 8`, `--grid 12`

---

## Device selection

The agent automatically picks the best available device:

| Priority | Device | Platform |
|----------|--------|----------|
| 1 | CUDA | Windows / Linux with NVIDIA GPU |
| 2 | MPS | macOS Apple Silicon |
| 3 | CPU | fallback |

---

## Expected convergence

| Steps | What to expect |
|-------|---------------|
| 0–100k | `eval/mean_score ≈ 0` — greedy policy on untrained network is worse than random |
| 100–300k | First consistent food eating |
| 300–600k | Steady score improvement |
| 600k+ | Strong performance |

---

## Project structure

```
src/snake/
├── env/
│   ├── game.py          # Pure game logic (no gym/torch — fast to unit test)
│   ├── channels.py      # 7-channel observation builder
│   └── snake_env.py     # Gymnasium wrapper
├── agent/
│   ├── network.py       # CNN (3×3 kernels for 12×12) + Dueling head
│   ├── replay_buffer.py # SumTree → Prioritized Replay + NStepBuffer
│   └── rainbow.py       # Rainbow DQN agent
├── training/
│   ├── config.py        # Typed dataclass — all hyperparameters in one place
│   └── trainer.py       # Training loop with TensorBoard logging
├── rendering/
│   └── renderer.py      # Pygame UI
└── evaluation/
    └── evaluator.py     # Deterministic rollouts

scripts/
├── train.py
├── evaluate.py
└── sanity_check.py
```
