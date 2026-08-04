# AI Snake — Rainbow DQN

A Snake agent trained from scratch with a **full Rainbow DQN** on a 10×10 grid,
using a CNN over a 7-channel board representation and 32 parallel environments
for fast experience collection.

## Demo

<img width="500" height="575" alt="Image" src="https://github.com/user-attachments/assets/2a12bc0e-d0a7-4afb-af93-27ce4d5c5687" />

---

## How the Rainbow DQN works

"Rainbow" combines six independent DQN improvements into one agent. All six are
active in this project (see [`rainbow.py`](src/snake/agent/rainbow.py) and
[`network.py`](src/snake/agent/network.py)):

| Component | What it does | Where |
|-----------|--------------|-------|
| **Double DQN** | The online net *selects* the next action, the target net *evaluates* it — removes Q-value overestimation. | `train_step` |
| **Dueling network** | Splits the head into a state-value stream `V(s)` and an advantage stream `A(s,a)`, recombined as `Q = V + A − mean(A)`. | `RainbowNet` |
| **Prioritized Experience Replay (PER)** | Samples transitions with high TD-error more often, via a SumTree; corrected with importance-sampling weights. | `replay_buffer.py` |
| **N-step returns** | Bootstraps from `n` steps ahead so reward signal propagates back faster on a sparse board. | `NStepBuffer` |
| **NoisyNet** | Learnable noise on the linear layers replaces ε-greedy — exploration is state-dependent and annealed automatically. | `NoisyLinear` |
| **C51 (Distributional RL)** | Instead of a single scalar `Q`, the net predicts a probability distribution over 51 fixed return "atoms" between `v_min` and `v_max`. The Bellman update is a projection of the shifted distribution; the loss is cross-entropy. | `RainbowNet` + `train_step` |

The target network is updated with a **soft (Polyak) update** (`τ = 0.005`) every
loop rather than a periodic hard copy, which keeps the target smooth.

---

## How we use the CNN

The board is a small image — 7 channels of 12×12 — so a compact CNN extracts
spatial features before the Dueling/C51 heads. See
[`_CNN`](src/snake/agent/network.py):

```
Input  (7, 12, 12)
  ├─ Conv 3×3, 32ch, pad 1      + ReLU          → (32, 12, 12)
  ├─ Conv 3×3, 64ch, pad 1, stride 2 + ReLU     → (64, 6, 6)   # downsample
  ├─ Conv 3×3, 64ch, pad 1      + ReLU          → (64, 6, 6)
  ├─ Flatten                                     → (2304,)
  └─ Linear 2304 → 256          + ReLU          → (256,)
```

The 256-d feature vector then feeds two **NoisyLinear** dueling streams, each
ending in a distributional (C51) output.

Design notes:
- **3×3 kernels with padding 1** preserve the grid so edge cells aren't lost.
- **One stride-2 layer** halves resolution (12→6), the only deliberate spatial
  downsampling.
- The input isn't raw pixels but **semantic binary/scalar masks**, so the CNN's
  local mixing detects patterns like "food adjacent" or "wall one step away"
  rather than fine visual detail.

### Observation space — 7 channels

| Ch | Description |
|----|-------------|
| 0 | Snake head (binary) |
| 1 | Body gradient — head ≈ 1.0, tail ≥ 0.15 (tells the agent which segments disappear soonest; floored so long-snake tails stay visible) |
| 2 | Food (binary) |
| 3 | Danger map — body cells = 1.0, wall-edge cells = 0.5 (tail excluded, it moves away) |
| 4 | Flood-fill reachability from head — 1 = reachable, 0 = cut off (self-entrapment signal) |
| 5 | Head at t−1 |
| 6 | Body gradient at t−1 |

---

## Parallel environment setup

Training collects experience from many Snake games at once using Gymnasium's
`AsyncVectorEnv`, which runs each environment in its own process. See
[`scripts/train.py`](scripts/train.py):

```python
envs = gym.vector.AsyncVectorEnv([
    make_env(cfg.grid_size, cfg.max_steps_no_food) for _ in range(args.num_envs)
])
```

Each training loop iteration:
1. Runs **one batched inference** over all `num_envs` states → one action per env.
2. Steps every env in parallel with a single `envs.step(actions)` call.
3. Feeds each env's transition into its **own n-step buffer**, then into the
   shared PER buffer.
4. Runs several gradient steps and a soft target update.

Set the number of parallel envs with `--num-envs` (default 16). A good rule of
thumb is roughly `2 × physical CPU cores`; going higher won't break anything but
may not increase throughput once the GPU or CPU saturates.

> **sps note:** the `sps` printed during training is the *total* environment
> steps/second summed across all envs (not per-env). `global_step` counts total
> env steps too, so `--steps` is a total-env-step budget.

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

### Windows / Linux with NVIDIA GPU

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
| `--steps` | 4 000 000 | Total environment steps (summed across all envs) |
| `--num-envs` | 16 | Number of parallel environments |
| `--run-dir` | `runs/C51_Rainbow` | Where checkpoints and TensorBoard logs are saved |
| `--lr` | 1e-4 | Learning rate |
| `--batch` | 128 | Batch size |
| `--n-step` | 5 | N-step return length |
| `--buffer` | 500 000 | Replay buffer capacity |
| `--resume` | — | Path to a checkpoint `.pt` to continue from |

Resume after interruption:

```bash
python scripts/train.py --resume runs/C51_Rainbow/interrupted_00025000.pt
```

Training prints a line every 1 000 steps:

```
step   25,000 | ep  1,345 | loss 1.517 | lr 1.00e-04 | best_score 0 | 340 sps
```

Checkpoints are saved every 50 000 steps and on Ctrl+C; a `final.pt` is written
on normal completion.

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir runs/
```

Key metrics:

| Tag | Meaning |
|-----|---------|
| `episode/score` | Food eaten per episode |
| `episode/length` | Snake length at death |
| `train/loss` | C51 cross-entropy loss |
| `eval/mean_score` | Greedy policy score (10-episode average, logged every 25k steps) |

### 4. Watch a trained agent

```bash
python scripts/evaluate.py --checkpoint runs/C51_Rainbow/final.pt
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

> On MPS, per-step GPU kernel-launch overhead dominates for this small network,
> so throughput is much lower than on CUDA. Use a Mac for verifying/evaluating
> and an NVIDIA GPU for real training runs.

---

## Project structure

```
src/snake/
├── env/
│   ├── game.py          # Pure game logic (no gym/torch — fast to unit test)
│   ├── channels.py      # 7-channel observation builder
│   └── snake_env.py     # Gymnasium wrapper (+ flood-fill area reward)
├── agent/
│   ├── network.py       # CNN + Dueling head + NoisyLinear + C51 output
│   ├── replay_buffer.py # SumTree → Prioritized Replay + NStepBuffer
│   └── rainbow.py       # Rainbow DQN agent (Double/Dueling/PER/N-step/Noisy/C51)
├── training/
│   ├── config.py        # Typed dataclass — all hyperparameters in one place
│   └── trainer.py       # Parallel training loop with TensorBoard logging
├── rendering/
│   └── renderer.py      # Pygame UI
└── evaluation/
    └── evaluator.py     # Deterministic rollouts

scripts/
├── train.py             # AsyncVectorEnv training entry point
├── evaluate.py          # Watch a trained checkpoint play
└── sanity_check.py
```
