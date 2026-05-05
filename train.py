"""Training script for JAX DrQ with Comet ML logging."""

import argparse
import os
import pickle
import time
from typing import Dict
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import yaml
from comet_ml import Experiment
from flax import serialization

from drq import DRQAgent
from replay_buffer import ReplayBuffer
from utils import make_env, set_seed_everywhere


class VideoRecorder:
    """Simple video recorder."""
    
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        self.frames = []
        self.enabled = False
    
    def init(self, enabled=True):
        self.frames = []
        self.enabled = enabled and self.save_dir is not None
    
    def record(self, obs):
        """Record frame from observation (pixel-based)."""
        if self.enabled:
            # obs is in CHW format (channels, height, width)
            # Convert to HWC format for video
            if len(obs.shape) == 3:
                # Single observation: (C*frame_stack, H, W)
                # Take only the last frame (most recent)
                c, h, w = obs.shape
                if c == 9:  # 3 frames * 3 channels
                    frame = obs[-3:, :, :]  # Last 3 channels (most recent frame)
                elif c == 3:
                    frame = obs
                else:
                    # Take last 3 channels
                    frame = obs[-3:, :, :]
                # Convert from CHW to HWC
                frame = np.transpose(frame, (1, 2, 0))
                self.frames.append(frame)
    
    def save(self, filename):
        if self.enabled and len(self.frames) > 0:
            import imageio
            path = os.path.join(self.save_dir, filename)
            # Use macro_block_size=1 to prevent auto-resizing of non-16-divisible dimensions
            imageio.mimsave(path, self.frames, fps=30, macro_block_size=1)
            self.frames = []
            return path  # Return path for logging to Comet ML
        self.frames = []
        return None


class Workspace:
    """Training workspace for DrQ agent."""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        
        # Setup directories
        self.work_dir = cfg.get('work_dir', './runs')
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Setup device
        device_str = cfg.get('device', 'gpu')
        if device_str.lower() == 'cpu':
            # Force CPU
            jax.config.update('jax_platform_name', 'cpu')
            print(f'Using device: CPU')
        else:
            # Use GPU
            devices = jax.devices('gpu')
            print(f'Using device: {devices[0]}')
        
        # Set seeds
        set_seed_everywhere(cfg['seed'])
        self.rng = jrandom.PRNGKey(cfg['seed'])
        
        # Initialize Comet ML
        self.experiment = Experiment(
            api_key=cfg.get('comet_api_key', os.environ.get('COMET_API_KEY')),
            project_name=cfg.get('comet_project', 'drq-jax'),
            workspace=cfg.get('comet_workspace', None),
        )
        
        # Set experiment name if provided
        experiment_name = cfg.get('comet_experiment_name', None)
        if experiment_name is not None:
            self.experiment.set_name(experiment_name)
        
        self.experiment.log_parameters(cfg)
        
        # Create environment
        env_name = cfg['env']
        if env_name == 'ball_in_cup_catch':
            domain_name, task_name = 'ball_in_cup', 'catch'
        elif env_name == 'point_mass_easy':
            domain_name, task_name = 'point_mass', 'easy'
        else:
            parts = env_name.split('_')
            domain_name = parts[0]
            task_name = '_'.join(parts[1:])
        
        self.env = make_env(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg['seed'],
            frame_stack=cfg['frame_stack'],
            action_repeat=cfg['action_repeat'],
            image_size=cfg['image_size'],
            max_steps=cfg['max_steps']
        )
        
        # Get environment specs
        obs_shape = self.env.observation_space.shape
        action_dim = self.env.action_space.shape[0]
        action_range = (
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        )
        
        # Initialize agent
        self.rng, agent_key = jrandom.split(self.rng)
        self.agent = DRQAgent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            action_range=action_range,
            rng_key=agent_key,
            lr=cfg.get('lr', 1e-3),
            discount=cfg.get('discount', 0.99),
            tau=cfg.get('critic_tau', 0.01),
            init_temperature=cfg.get('init_temperature', 0.1),
            actor_update_freq=cfg.get('actor_update_freq', 2),
            critic_target_update_freq=cfg.get('critic_target_update_freq', 2),
            hidden_dim=cfg.get('hidden_dim', 1024),
            hidden_depth=cfg.get('hidden_depth', 2),
            feature_dim=cfg.get('feature_dim', 50),
            drq_k=cfg.get('drq_k', 2),
            drq_m=cfg.get('drq_m', 2),
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_shape=obs_shape,
            action_shape=(action_dim,),
            capacity=cfg['replay_buffer_capacity'],
            image_pad=cfg['image_pad'],
            use_augmentation=cfg.get('use_augmentation', True)
        )
        
        # Video recorder
        video_dir = os.path.join(self.work_dir, 'videos') if cfg.get('save_video', False) else None
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        self.video_recorder = VideoRecorder(video_dir)
        
        self.step = 0
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(self.work_dir, 'checkpoints')
        if cfg.get('checkpoint_frequency', 0) > 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, filename: str = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f'checkpoint_{self.step}.pkl'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Serialize TrainState objects properly (they contain unpicklable optimizer functions)
        checkpoint = {
            'step': self.step,
            'agent': {
                'actor_state': serialization.to_state_dict(self.agent.actor_state),
                'critic_state': serialization.to_state_dict(self.agent.critic_state),
                'log_alpha': np.array(self.agent.log_alpha),  # Convert to numpy
                'alpha_opt_state': jax.tree_util.tree_map(np.asarray, self.agent.alpha_opt_state),
                'agent_step': self.agent.step,
            },
            'rng': np.array(self.rng),  # Convert RNG to numpy
            'cfg': self.cfg,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f'Saved checkpoint to {checkpoint_path}')
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        print(f'Loading checkpoint from {checkpoint_path}')
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore agent state (convert back from serialized format)
        self.agent.actor_state = serialization.from_state_dict(
            self.agent.actor_state, checkpoint['agent']['actor_state']
        )
        self.agent.critic_state = serialization.from_state_dict(
            self.agent.critic_state, checkpoint['agent']['critic_state']
        )
        self.agent.log_alpha = jnp.array(checkpoint['agent']['log_alpha'])
        self.agent.alpha_opt_state = jax.tree_util.tree_map(
            jnp.asarray, checkpoint['agent']['alpha_opt_state']
        )
        self.agent.step = checkpoint['agent']['agent_step']
        
        # Restore training state
        self.step = checkpoint['step']
        self.rng = jnp.array(checkpoint['rng'])  # Convert back to JAX array
        
        print(f'Resumed from step {self.step}')
    
    def evaluate(self) -> float:
        """Evaluate agent performance."""
        total_reward = 0.0
        
        for episode in range(self.cfg['num_eval_episodes']):
            obs, _ = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            
            done = False
            episode_reward = 0.0
            episode_step = 0
            
            while not done:
                # Select action deterministically
                self.rng, act_key = jrandom.split(self.rng)
                action = self.agent.act(obs, act_key, deterministic=True)
                action = np.array(action)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.video_recorder.record(obs)
                episode_reward += reward
                episode_step += 1
            
            total_reward += episode_reward
            video_path = self.video_recorder.save(f'step_{self.step}_ep_{episode}.mp4')
            
            # Upload video to Comet ML (only for first episode)
            if video_path is not None and episode == 0:
                self.experiment.log_video(video_path, name=f'eval_step_{self.step}', step=self.step)
        
        avg_reward = total_reward / self.cfg['num_eval_episodes']
        
        # Log to Comet ML
        self.experiment.log_metric('eval/episode_reward', avg_reward, step=self.step)
        
        print(f'| eval | S: {self.step} | R: {avg_reward:.4f}')
        
        return avg_reward
    
    def run(self):
        """Main training loop."""
        episode = 0
        episode_reward = 0.0
        episode_step = 0
        
        obs, _ = self.env.reset()
        done = True
        
        start_time = time.time()
        
        while self.step < self.cfg['num_train_steps']:
            # Reset episode
            if done:
                if self.step > 0:
                    # Log episode metrics
                    duration = time.time() - start_time
                    self.experiment.log_metric('train/episode_reward', episode_reward, step=self.step)
                    self.experiment.log_metric('train/episode', episode, step=self.step)
                    self.experiment.log_metric('train/duration', duration, step=self.step)
                    
                    print(f'| train | E: {episode} | S: {self.step} | R: {episode_reward:.4f} | D: {duration:.1f}s')
                    
                    start_time = time.time()
                
                # Evaluate periodically
                if self.step % self.cfg['eval_frequency'] == 0:
                    self.evaluate()
                
                # Save checkpoint periodically
                checkpoint_freq = self.cfg.get('checkpoint_frequency', 0)
                if checkpoint_freq > 0 and self.step % checkpoint_freq == 0:
                    checkpoint_path = self.save_checkpoint()
                    # Upload checkpoint to Comet ML
                    self.experiment.log_model('checkpoint', checkpoint_path)
                
                obs, _ = self.env.reset()
                done = False
                episode_reward = 0.0
                episode_step = 0
                episode += 1
            
            # Sample action
            if self.step < self.cfg['num_seed_steps']:
                action = self.env.action_space.sample()
            else:
                self.rng, act_key = jrandom.split(self.rng)
                action = self.agent.act(obs, act_key, deterministic=False)
                action = np.array(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            done_float = float(done)
            # For proper bootstrapping: don't mask value if episode was truncated due to time limit
            # Check if truncated due to time limit (not task failure)
            is_truncated_timeout = truncated and info.get('TimeLimit.truncated', False)
            done_no_max = 0.0 if is_truncated_timeout else done_float
            
            self.replay_buffer.add(obs, action, reward, next_obs, done_float, done_no_max)
            
            # Update agent
            if self.step >= self.cfg['num_seed_steps']:
                for _ in range(self.cfg['num_train_iters']):
                    self.rng, sample_key, update_key = jrandom.split(self.rng, 3)
                    
                    # Sample batch
                    batch = self.replay_buffer.sample(self.cfg['batch_size'], sample_key)
                    obs_b, action_b, reward_b, next_obs_b, not_done_b, obs_aug_b, next_obs_aug_b = batch
                    
                    # Update agent
                    info = self.agent.update(
                        obs_b, action_b, reward_b, next_obs_b, not_done_b,
                        obs_aug_b, next_obs_aug_b, update_key
                    )
                    
                    # Log training metrics
                    if self.step % self.cfg.get('log_frequency', 1000) == 0:
                        for key, value in info.items():
                            self.experiment.log_metric(f'train/{key}', float(value), step=self.step)
                        self.experiment.log_metric('train/batch_reward', float(reward_b.mean()), step=self.step)
            
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            self.step += 1
        
        print('Training completed!')
        
        # Save final checkpoint
        if self.cfg.get('checkpoint_frequency', 0) > 0:
            final_checkpoint = self.save_checkpoint('checkpoint_final.pkl')
            self.experiment.log_model('final_checkpoint', final_checkpoint)
        
        self.experiment.end()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None, help='Comet ML experiment name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, gpu, gpu:0, gpu:1, etc.')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override with command line args
    if args.env is not None:
        cfg['env'] = args.env
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    if args.experiment_name is not None:
        cfg['comet_experiment_name'] = args.experiment_name
    if args.device is not None:
        cfg['device'] = args.device
    
    # Create workspace and run
    workspace = Workspace(cfg)
    
    # Load checkpoint if provided
    if args.checkpoint is not None:
        workspace.load_checkpoint(args.checkpoint)
    
    workspace.run()


if __name__ == '__main__':
    main()
