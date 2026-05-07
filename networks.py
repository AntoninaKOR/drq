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


class ActorCritic(nn.Module):
    """Combined ActorCritic with shared encoder (like original PyTorch DrQ)."""
    
    action_dim: int
    hidden_dim: int = 1024
    hidden_depth: int = 2
    feature_dim: int = 50
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    
    def setup(self):
        """Setup shared encoder and separate actor/critic heads."""
        # Shared encoder (updated only via critic loss)
        self.encoder = Encoder(feature_dim=self.feature_dim)
        
        # Actor MLP layers
        self.actor_layers = [
            nn.Dense(features=self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0))
            for _ in range(self.hidden_depth)
        ]
        self.actor_out = nn.Dense(features=2 * self.action_dim,
                                  kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0))
        
        # Critic Q1 MLP layers
        self.q1_layers = [
            nn.Dense(features=self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0))
            for _ in range(self.hidden_depth)
        ]
        self.q1_out = nn.Dense(features=1,
                              kernel_init=orthogonal(1.0),
                              bias_init=constant(0.0))
        
        # Critic Q2 MLP layers
        self.q2_layers = [
            nn.Dense(features=self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0))
            for _ in range(self.hidden_depth)
        ]
        self.q2_out = nn.Dense(features=1,
                              kernel_init=orthogonal(1.0),
                              bias_init=constant(0.0))
    
    def __call__(self, obs, action):
        """Forward through both actor and critic."""
        mu, log_std = self.actor(obs, detach_encoder=False)
        q1, q2 = self.critic(obs, action, detach_encoder=False)
        return mu, log_std, q1, q2
    
    def actor(self, obs, detach_encoder=False):
        """Forward pass through actor (returns mu, log_std)."""
        # Encode observations with shared encoder
        features = self.encoder(obs)
        
        # Detach encoder gradients for actor update (like PyTorch version)
        if detach_encoder:
            features = jax.lax.stop_gradient(features)
        
        # Actor MLP
        x = features
        for layer in self.actor_layers:
            x = layer(x)
            x = nn.relu(x)
        x = self.actor_out(x)
        
        # Split into mu and log_std
        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        return mu, log_std
    
    def critic(self, obs, action, detach_encoder=False):
        """Forward pass through critic (returns q1, q2)."""
        # Encode observations with shared encoder
        features = self.encoder(obs)
        
        # Detach encoder gradients for actor update (like PyTorch version)
        if detach_encoder:
            features = jax.lax.stop_gradient(features)
        
        # Concatenate features and action
        x = jnp.concatenate([features, action], axis=-1)
        
        # Q1 network
        q1 = x
        for layer in self.q1_layers:
            q1 = layer(q1)
            q1 = nn.relu(q1)
        q1 = self.q1_out(q1)
        
        # Q2 network
        q2 = x
        for layer in self.q2_layers:
            q2 = layer(q2)
            q2 = nn.relu(q2)
        q2 = self.q2_out(q2)
        
        return q1, q2
