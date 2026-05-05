"""Utility functions for JAX DrQ implementation."""

import os
import random
from typing import Any, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from collections import deque
import dm_env
from dm_control import suite
from dm_control.suite.wrappers import pixels
from shimmy.dm_control_compatibility import DmControlCompatibilityV0


def set_seed_everywhere(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    

class FrameStack(gym.Wrapper):
    """Stack frames for temporal information."""
    
    def __init__(self, env, k: int):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=np.uint8
        )
        # Handle _max_episode_steps attribute safely
        if hasattr(env, '_max_episode_steps'):
            self._max_episode_steps = env._max_episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
    """Repeat actions for multiple steps."""
    
    def __init__(self, env, repeat: int):
        super().__init__(env)
        self._repeat = repeat
        # Preserve _max_episode_steps if it exists
        if hasattr(env, '_max_episode_steps'):
            self._max_episode_steps = env._max_episode_steps
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(domain_name: str, task_name: str, seed: int, 
             frame_stack: int, action_repeat: int, image_size: int,
             max_steps: int = 1000):
    """Create DMControl environment wrapped for pixel observations."""
    # Determine camera ID
    camera_id = 2 if domain_name == 'quadruped' else 0
    
    # Create dm_control environment
    dm_env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={'random': seed}
    )
    
    # Wrap with dm_control's pixel wrapper for pixel observations
    dm_env = pixels.Wrapper(
        dm_env,
        pixels_only=True,
        render_kwargs={
            'camera_id': camera_id,
            'height': image_size,
            'width': image_size,
        }
    )
    
    # Convert to Gymnasium with shimmy
    env = DmControlCompatibilityV0(dm_env, render_mode=None)
    
    # Add TimeLimit wrapper to properly handle episode truncation
    # Uses max_steps from config (default 1000, standard for dm_control)
    env = TimeLimit(env, max_episode_steps=max_steps)
    
    # The pixel wrapper returns observations as dict with 'pixels' key
    # We need to extract just the pixels and convert to CHW format
    class PixelExtractor(gym.ObservationWrapper):
        def __init__(self, env, image_size):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, image_size, image_size),
                dtype=np.uint8
            )
        
        def observation(self, obs):
            if isinstance(obs, dict) and 'pixels' in obs:
                pixels = obs['pixels']
            else:
                pixels = obs
            # Transpose from HWC to CHW format
            if len(pixels.shape) == 3 and pixels.shape[-1] == 3:
                pixels = np.transpose(pixels, (2, 0, 1))
            return pixels
    
    env = PixelExtractor(env, image_size)
    
    # Add action repeat wrapper if needed
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    
    # Frame stack
    env = FrameStack(env, k=frame_stack)
    env.action_space.seed(seed)
    
    return env


def soft_update(params: Dict, target_params: Dict, tau: float) -> Dict:
    """Soft update of target network parameters."""
    return jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        params, target_params
    )


def orthogonal_init(scale=1.0):
    """Orthogonal initialization for neural network weights."""
    def init(key, shape, dtype=jnp.float32):
        if len(shape) == 2:
            # For linear layers
            n_rows, n_cols = shape
            flat_shape = (n_rows, n_cols)
        elif len(shape) == 4:
            # For convolutional layers
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[2] * receptive_field_size
            fan_out = shape[3] * receptive_field_size
            flat_shape = (fan_in, fan_out)
        else:
            flat_shape = shape
            
        matrix = jrandom.normal(key, flat_shape)
        q, r = jnp.linalg.qr(matrix)
        q = q * jnp.sign(jnp.diag(r))
        
        if len(shape) == 4:
            q = q.reshape(shape)
        
        return scale * q[:shape[0], :shape[1]] if len(shape) == 2 else scale * q
    
    return init


def random_crop(key, img, padding):
    """Random crop augmentation for images."""
    # img shape: (batch, channels, height, width)
    batch_size, channels, h, w = img.shape
    
    # Pad the image
    img_padded = jnp.pad(
        img,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='edge'
    )
    
    # Random crop
    crop_max = h + 2 * padding - h
    keys = jrandom.split(key, batch_size)
    
    def crop_single(key, img):
        h_start = jrandom.randint(key, (), 0, crop_max + 1)
        w_start = jrandom.randint(key, (), 0, crop_max + 1)
        return jax.lax.dynamic_slice(
            img,
            (0, h_start, w_start),
            (channels, h, w)
        )
    
    cropped = jax.vmap(crop_single)(keys, img_padded)
    return cropped


def preprocess_obs(obs):
    """Preprocess observations by normalizing to [0, 1]."""
    return obs.astype(jnp.float32) / 255.0


class TanhTransform:
    """Tanh transformation for actions."""
    
    @staticmethod
    def forward(x):
        return jnp.tanh(x)
    
    @staticmethod
    def inverse(y):
        # Inverse tanh (arctanh)
        eps = 1e-6
        y = jnp.clip(y, -1 + eps, 1 - eps)
        return 0.5 * jnp.log((1 + y) / (1 - y))
    
    @staticmethod
    def log_det_jacobian(x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


def sample_squashed_normal(key, mu, log_std):
    """Sample from a squashed normal distribution."""
    std = jnp.exp(log_std)
    eps = jrandom.normal(key, mu.shape)
    x = mu + eps * std
    y = jnp.tanh(x)
    
    # Log probability
    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + eps**2)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
    
    # Correct for tanh squashing
    log_prob -= jnp.sum(TanhTransform.log_det_jacobian(x), axis=-1, keepdims=True)
    
    return y, log_prob


def eval_mode(func):
    """Decorator for evaluation mode (deterministic actions)."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs, deterministic=True)
    return wrapper
