"""Neural network architectures for JAX DrQ implementation."""

from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    
    feature_dim: int = 50
    num_layers: int = 4
    num_filters: int = 32
    
    @nn.compact
    def __call__(self, obs):
        # Normalize observations
        x = obs.astype(jnp.float32) / 255.0
        
        # Convolutional layers
        x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), 
                   strides=(2, 2), kernel_init=orthogonal(jnp.sqrt(2.0)),
                   bias_init=constant(0.0))(x)
        x = nn.relu(x)
        
        for _ in range(self.num_layers - 1):
            x = nn.Conv(features=self.num_filters, kernel_size=(3, 3),
                       strides=(1, 1), kernel_init=orthogonal(jnp.sqrt(2.0)),
                       bias_init=constant(0.0))(x)
            x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Linear projection
        x = nn.Dense(features=self.feature_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = jnp.tanh(x)
        
        return x


class Actor(nn.Module):
    """Actor network for continuous control."""
    
    action_dim: int
    hidden_dim: int = 1024
    hidden_depth: int = 2
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    feature_dim: int = 50
    
    @nn.compact
    def __call__(self, obs):
        # Encode observations
        encoder = Encoder(feature_dim=self.feature_dim)
        x = encoder(obs)
        
        # MLP trunk
        for _ in range(self.hidden_depth):
            x = nn.Dense(features=self.hidden_dim,
                        kernel_init=orthogonal(jnp.sqrt(2.0)),
                        bias_init=constant(0.0))(x)
            x = nn.relu(x)
        
        # Output layer for mean and log_std
        x = nn.Dense(features=2 * self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0))(x)
        
        mu, log_std = jnp.split(x, 2, axis=-1)
        
        # Constrain log_std
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        return mu, log_std


class Critic(nn.Module):
    """Double Q-learning critic network."""
    
    hidden_dim: int = 1024
    hidden_depth: int = 2
    feature_dim: int = 50
    
    @nn.compact
    def __call__(self, obs, action):
        # Encode observations
        encoder = Encoder(feature_dim=self.feature_dim)
        obs_features = encoder(obs)
        
        # Concatenate observation features and action
        x = jnp.concatenate([obs_features, action], axis=-1)
        
        # Q1 network
        q1 = x
        for _ in range(self.hidden_depth):
            q1 = nn.Dense(features=self.hidden_dim,
                         kernel_init=orthogonal(jnp.sqrt(2.0)),
                         bias_init=constant(0.0))(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(features=1,
                     kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(q1)
        
        # Q2 network
        q2 = x
        for _ in range(self.hidden_depth):
            q2 = nn.Dense(features=self.hidden_dim,
                         kernel_init=orthogonal(jnp.sqrt(2.0)),
                         bias_init=constant(0.0))(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(features=1,
                     kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(q2)
        
        return q1, q2


class ActorCriticEncoder(nn.Module):
    """Combined network with shared encoder for actor-critic."""
    
    action_dim: int
    hidden_dim: int = 1024
    hidden_depth: int = 2
    feature_dim: int = 50
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    
    def setup(self):
        self.encoder = Encoder(feature_dim=self.feature_dim)
        
    @nn.compact
    def __call__(self, obs, action=None, mode='actor'):
        if mode == 'encoder':
            return self.encoder(obs)
        elif mode == 'actor':
            return self.forward_actor(obs)
        elif mode == 'critic':
            return self.forward_critic(obs, action)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward_actor(self, obs):
        x = self.encoder(obs)
        
        # MLP trunk
        for _ in range(self.hidden_depth):
            x = nn.Dense(features=self.hidden_dim,
                        kernel_init=orthogonal(jnp.sqrt(2.0)),
                        bias_init=constant(0.0))(x)
            x = nn.relu(x)
        
        # Output layer
        x = nn.Dense(features=2 * self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0))(x)
        
        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        return mu, log_std
    
    def forward_critic(self, obs, action):
        obs_features = self.encoder(obs)
        x = jnp.concatenate([obs_features, action], axis=-1)
        
        # Q1
        q1 = x
        for _ in range(self.hidden_depth):
            q1 = nn.Dense(features=self.hidden_dim,
                         kernel_init=orthogonal(jnp.sqrt(2.0)),
                         bias_init=constant(0.0))(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(features=1,
                     kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(q1)
        
        # Q2
        q2 = x
        for _ in range(self.hidden_depth):
            q2 = nn.Dense(features=self.hidden_dim,
                         kernel_init=orthogonal(jnp.sqrt(2.0)),
                         bias_init=constant(0.0))(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(features=1,
                     kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(q2)
        
        return q1, q2
