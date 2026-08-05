import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from snake.agent.rainbow import RainbowAgent
from snake.evaluation.evaluator import Evaluator
from snake.training.config import RainbowConfig
from snake.agent.replay_buffer import NStepBuffer


class Trainer:
    """Handles the training loop, experience collection, and logging for Rainbow DQN."""

    def __init__(self, agent: RainbowAgent, env, config: RainbowConfig):
        self.agent = agent
        self.env = env
        self.cfg = config
        
        # Setup run directory and TensorBoard writer
        config.run_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(config.run_path / "tb"))
        
        self.evaluator = Evaluator(
            agent, 
            config.grid_size, 
            config.eval_episodes,
            config.max_steps_no_food, 
            config.egocentric,
        )

    def train(self) -> None:
        """Executes the main training loop across all parallel environments."""
        cfg = self.cfg
        
        print("\n" + "=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"Total Steps:        {cfg.total_steps:,}")
        print(f"Batch Size:         {cfg.batch_size}")
        print(f"Learning Rate:      {cfg.lr}")
        print(f"Buffer Capacity:    {cfg.buffer_capacity:,}")
        print(f"Grid Size:          {cfg.grid_size}")
        print(f"Warmup Steps:       {cfg.warmup_steps:,}")
        print(f"Log Frequency:      {cfg.log_freq:,}")
        print(f"Checkpoint Freq:    {cfg.checkpoint_freq:,}")
        print(f"Run Directory:      {cfg.run_dir}")
        print("=" * 60 + "\n")
        
        obs, _ = self.env.reset()
        num_envs = obs.shape[0]
        
        n_step_buffers = [NStepBuffer(cfg.n_step, cfg.gamma) for _ in range(num_envs)]
        episode_rewards = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs)
        episode_count = 0
        best_score = 0
        
        loss_accum = 0.0
        loss_steps = 0
        t0 = time.time()
        
        # Graceful shutdown handler
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            
        print(f"Training on {self.agent.device} | run dir: {cfg.run_dir}")
        print(f"Warming up for {cfg.warmup_steps:,} steps...")

        global_step = 0
        steps_since_log = 0  # Counter for accurate steps-per-second (SPS) calculation

        try:
            while global_step < cfg.total_steps:
                # Update learning rate
                current_lr = cfg.learning_rate(global_step)
                for param_group in self.agent.optimizer.param_groups:
                    param_group["lr"] = current_lr
                    
                state_batch = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                
                self.agent.online_net.train()
                self.agent.online_net.reset_noise()
                
                # ---- 1. Select Actions ----
                with torch.no_grad():
                    log_p = self.agent.online_net(state_batch)
                    p = log_p.exp()
                    q_values = (p * self.agent.support).sum(dim=2)
                    actions = q_values.argmax(dim=1).cpu().numpy()
                
                # ---- 2. Environment Step ----
                next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
                dones = terminateds | truncateds
                
                # ---- 3. Process Transitions & Store in Buffer ----
                for i in range(num_envs):
                    global_step += 1
                    steps_since_log += 1
                    
                    episode_rewards[i] += rewards[i]
                    episode_steps[i] += 1
                    
                    if dones[i]:
                        # Handle VectorEnv automatic reset behavior to get the true terminal state
                        true_next_obs = next_obs[i]
                        if isinstance(infos, dict) and "final_observation" in infos and infos["final_observation"] is not None:
                            try:
                                true_next_obs = infos["final_observation"][i]
                            except Exception:
                                pass
                                
                        final_info = None
                        if isinstance(infos, dict) and "final_info" in infos and infos["final_info"] is not None:
                            try:
                                info_item = infos["final_info"][i]
                                if isinstance(info_item, dict):
                                    final_info = info_item
                            except Exception:
                                pass
                                
                        if final_info is None:
                            final_info = {
                                "score": infos["score"][i] if "score" in infos else 0,
                                "snake_length": infos["snake_length"][i] if "snake_length" in infos else 0
                            }
                            
                        # Store terminal transition
                        ready_transitions = n_step_buffers[i].add(
                            (obs[i], actions[i], rewards[i], true_next_obs, True)
                        )
                        for s, a, R, ns, d in ready_transitions:
                            self.agent.replay_buffer.add(s, a, R, ns, d)
                            
                        # Log episode metrics
                        self.writer.add_scalar("episode/reward", episode_rewards[i], global_step)
                        self.writer.add_scalar("episode/length", final_info["snake_length"], global_step)
                        self.writer.add_scalar("episode/score", final_info["score"], global_step)
                        
                        best_score = max(best_score, final_info["score"])
                        episode_count += 1
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0
                    else:
                        # Store normal transition
                        ready_transitions = n_step_buffers[i].add(
                            (obs[i], actions[i], rewards[i], next_obs[i], False)
                        )
                        for s, a, R, ns, d in ready_transitions:
                            self.agent.replay_buffer.add(s, a, R, ns, d)

                # ---- 4. Train Network ----
                if global_step >= cfg.warmup_steps and len(self.agent.replay_buffer) >= cfg.batch_size:
                    for _ in range(cfg.train_freq):
                        loss = self.agent.train_step(cfg.batch_size, cfg.beta(global_step))
                        loss_accum += loss
                        loss_steps += 1

                # ---- 5. Sync Target Network ----
                self.agent.update_target()

                # ---- 6. Logging & TensorBoard ----
                if steps_since_log >= cfg.log_freq and loss_steps > 0:
                    avg_loss = loss_accum / loss_steps
                    sps = steps_since_log / (time.time() - t0)
                    
                    self.writer.add_scalar("train/loss", avg_loss, global_step)
                    self.writer.add_scalar("train/steps_per_sec", sps, global_step)
                    self.writer.add_scalar("train/buffer_size", len(self.agent.replay_buffer), global_step)
                    self.writer.add_scalar("train/lr", current_lr, global_step)
                    
                    print(
                        f"step {global_step:>8,} | ep {episode_count:>5} | "
                        f"loss {avg_loss:.4f} | lr {current_lr:.2e} | "
                        f"best_score {best_score} | {sps:.0f} sps"
                    )
                    
                    loss_accum = 0.0
                    loss_steps = 0
                    steps_since_log = 0
                    t0 = time.time()

                # ---- 7. Evaluation ----
                if global_step % cfg.eval_freq < num_envs:
                    metrics = self.evaluator.evaluate()
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"eval/{k}", v, global_step)
                    print(
                        f"  [eval] mean_score={metrics['mean_score']:.1f} "
                        f"mean_length={metrics['mean_length']:.1f} "
                        f"max_score={metrics['max_score']}"
                    )

                # ---- 8. Checkpointing ----
                if global_step % cfg.checkpoint_freq < num_envs:
                    ckpt_path = cfg.run_path / f"checkpoint_{global_step:08d}.pt"
                    self.agent.save(ckpt_path)
                    print(f"  [ckpt] saved -> {ckpt_path}")
                    
                obs = next_obs

        except KeyboardInterrupt:
            print(f"\nInterrupted at step {global_step:,}. Saving...")
        finally:
            if global_step >= cfg.total_steps:
                final_path = cfg.run_path / "final.pt"
                self.agent.save(final_path)
                print(f"Training Complete! Saved -> {final_path}")
            else:
                interrupt_path = cfg.run_path / f"interrupted_{global_step:08d}.pt"
                self.agent.save(interrupt_path)
                print(f"Saved -> {interrupt_path}")
                
            self.writer.close()