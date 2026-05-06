"""Replay buffer for JAX DrQ implementation."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom


class ReplayBuffer:
    """Experience replay buffer with image augmentation support."""
    
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device='cpu', use_augmentation=True):
        self.capacity = capacity
        self.image_pad = image_pad
        self.use_augmentation = use_augmentation
        
        # Initialize buffers
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.full = False
    
    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def add(self, obs, action, reward, next_obs, done, done_no_max):
        """Add a transition to the buffer."""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def sample(self, batch_size, rng_key):
        """Sample a batch of transitions with augmentation."""
        # Sample indices
        idxs = np.random.randint(
            0,
            self.capacity if self.full else self.idx,
            size=batch_size
        )
        
        # Get transitions
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        not_dones_no_max = self.not_dones_no_max[idxs]
        
        # Convert to JAX arrays
        obses = jnp.array(obses)
        next_obses = jnp.array(next_obses)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        not_dones_no_max = jnp.array(not_dones_no_max)
        
        # Apply augmentation (random crop) if enabled (DrQ), otherwise use original images (SAC)
        if self.use_augmentation:
            keys = jrandom.split(rng_key, 2)
            obses_aug = self._augment(obses, keys[0])
            next_obses_aug = self._augment(next_obses, keys[1])
        else:
            # No augmentation - use same observations for both (SAC mode)
            obses_aug = obses
            next_obses_aug = next_obses
        
        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
    
    def _augment(self, imgs, key):
        """Apply random crop augmentation to images."""
        # imgs shape: (batch, channels, height, width)
        batch_size = imgs.shape[0]
        
        # Pad images
        if len(imgs.shape) == 4:
            # (batch, channels, height, width)
            padded = jnp.pad(
                imgs,
                ((0, 0), (0, 0), (self.image_pad, self.image_pad), 
                 (self.image_pad, self.image_pad)),
                mode='edge'
            )
        else:
            raise ValueError(f"Unexpected image shape: {imgs.shape}")
        
        # Random crop
        _, c, h, w = imgs.shape
        crop_max = self.image_pad * 2
        
        keys = jrandom.split(key, batch_size)
        
        def crop_single(key, img):
            h_offset = jrandom.randint(key, (), 0, crop_max + 1)
            w_key = jrandom.split(key)[1]
            w_offset = jrandom.randint(w_key, (), 0, crop_max + 1)
            return jax.lax.dynamic_slice(
                img,
                (0, h_offset, w_offset),
                (c, h, w)
            )
        
        cropped = jax.vmap(crop_single)(keys, padded)
        return cropped


def random_crop_single(key, img, padding, output_size):
    """Random crop a single image."""
    # Pad
    padded = jnp.pad(
        img,
        ((0, 0), (padding, padding), (padding, padding)),
        mode='edge'
    )
    
    # Random crop location
    crop_h = jrandom.randint(key, (), 0, 2 * padding + 1)
    w_key = jrandom.split(key)[1]
    crop_w = jrandom.randint(w_key, (), 0, 2 * padding + 1)
    
    # Crop
    channels = img.shape[0]
    cropped = jax.lax.dynamic_slice(
        padded,
        (0, crop_h, crop_w),
        (channels, output_size, output_size)
    )
    
    return cropped


def random_crop_batch(key, imgs, padding):
    """Apply random crop to a batch of images."""
    batch_size = imgs.shape[0]
    output_size = imgs.shape[2]  # Assuming square images
    
    keys = jrandom.split(key, batch_size)
    
    cropped = jax.vmap(
        lambda k, img: random_crop_single(k, img, padding, output_size)
    )(keys, imgs)
    
    return cropped
