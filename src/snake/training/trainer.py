import signal
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from snake.agent.rainbow import RainbowAgent
from snake.env.snake_env import SnakeEnv
from snake.evaluation.evaluator import Evaluator
from snake.training.config import RainbowConfig


class Trainer:
    def __init__(self, agent: RainbowAgent, env: SnakeEnv, config: RainbowConfig):
        self.agent = agent
        self.env = env
        self.cfg = config

        config.run_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(config.run_path / "tb"))
        self.evaluator = Evaluator(agent, config.grid_size, config.eval_episodes)

    def train(self) -> None:
        cfg = self.cfg
        obs, _ = self.env.reset()

        episode_reward = 0.0
        episode_steps = 0
        episode_count = 0
        best_score = 0
        loss_accum = 0.0
        loss_steps = 0
        t0 = time.time()

        # SIGTERM (e.g. kill <pid>) — not available on Windows
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

        print(f"Training on {self.agent.device} | run dir: {cfg.run_dir}")
        print(f"Warming up for {cfg.warmup_steps:,} steps…")

        try:
            for step in range(1, cfg.total_steps + 1):
                epsilon = cfg.epsilon(step)
                action = self.agent.select_action(obs, epsilon)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_reward += reward
                episode_steps += 1

                # ---- Learning ----
                if step >= cfg.warmup_steps and len(self.agent.replay_buffer) >= cfg.batch_size:
                    for _ in range(cfg.train_freq):
                        loss = self.agent.train_step(cfg.batch_size, cfg.beta(step))
                        loss_accum += loss
                        loss_steps += 1

                # ---- Target sync ----
                if step % cfg.target_update_freq == 0:
                    self.agent.update_target()

                # ---- Episode end ----
                if done:
                    self.writer.add_scalar("episode/reward", episode_reward, step)
                    self.writer.add_scalar("episode/length", info["snake_length"], step)
                    self.writer.add_scalar("episode/score", info["score"], step)
                    self.writer.add_scalar("train/epsilon", epsilon, step)

                    best_score = max(best_score, info["score"])
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_count += 1

                # ---- Periodic logging ----
                if step % cfg.log_freq == 0 and loss_steps > 0:
                    avg_loss = loss_accum / loss_steps
                    sps = cfg.log_freq / (time.time() - t0)
                    self.writer.add_scalar("train/loss", avg_loss, step)
                    self.writer.add_scalar("train/steps_per_sec", sps, step)
                    self.writer.add_scalar("train/buffer_size", len(self.agent.replay_buffer), step)
                    print(
                        f"step {step:>8,} | ep {episode_count:>5} | "
                        f"loss {avg_loss:.4f} | ε {epsilon:.3f} | "
                        f"best_score {best_score} | {sps:.0f} sps"
                    )
                    loss_accum = 0.0
                    loss_steps = 0
                    t0 = time.time()

                # ---- Evaluation ----
                if step % cfg.eval_freq == 0:
                    metrics = self.evaluator.evaluate()
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"eval/{k}", v, step)
                    print(
                        f"  [eval] mean_score={metrics['mean_score']:.1f} "
                        f"mean_length={metrics['mean_length']:.1f} "
                        f"max_score={metrics['max_score']}"
                    )

                # ---- Checkpoint ----
                if step % cfg.checkpoint_freq == 0:
                    ckpt_path = cfg.run_path / f"checkpoint_{step:08d}.pt"
                    self.agent.save(ckpt_path)
                    print(f"  [ckpt] saved → {ckpt_path}")

        except KeyboardInterrupt:
            print(f"\nInterrupted at step {step:,}. Saving…")

        finally:
            interrupt_path = cfg.run_path / f"interrupted_{step:08d}.pt"
            self.agent.save(interrupt_path)
            self.writer.close()
            print(f"Saved → {interrupt_path}")
