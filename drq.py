"""JAX implementation of DrQ agent."""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
import flax
from flax.training import train_state
from functools import partial

from networks import Actor, Critic
from utils import soft_update, sample_squashed_normal


class TrainState(train_state.TrainState):
    """Extended train state with target network parameters."""
    target_params: flax.core.FrozenDict = None


class DRQAgent:
    """Data-regularized Q-learning agent."""
    
    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        action_range: Tuple[float, float],
        rng_key: jax.random.PRNGKey,
        lr: float = 1e-3,
        discount: float = 0.99,
        tau: float = 0.01,
        init_temperature: float = 0.1,
        actor_update_freq: int = 2,
        critic_target_update_freq: int = 2,
        hidden_dim: int = 1024,
        hidden_depth: int = 2,
        feature_dim: int = 50,
        drq_k: int = 2,
        drq_m: int = 2,
    ):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_range = action_range
        self.discount = discount
        self.tau = tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.drq_k = drq_k  # Q-target averaging: 1 or 2
        self.drq_m = drq_m  # Q-function averaging: 1 or 2
        
        # Split RNG keys
        key1, key2, key3 = jrandom.split(rng_key, 3)
        
        # Initialize networks
        self.actor = Actor(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            feature_dim=feature_dim
        )
        self.critic = Critic(
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            feature_dim=feature_dim
        )
        
        # Initialize parameters
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1, action_dim))
        
        actor_params = self.actor.init(key1, dummy_obs)
        critic_params = self.critic.init(key2, dummy_obs, dummy_action)
        
        # Initialize optimizers
        self.actor_optimizer = optax.adam(learning_rate=lr)
        self.critic_optimizer = optax.adam(learning_rate=lr)
        self.alpha_optimizer = optax.adam(learning_rate=lr)
        
        # Create train states
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=self.actor_optimizer
        )
        
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=self.critic_optimizer,
            target_params=critic_params
        )
        
        # Temperature parameter
        self.log_alpha = jnp.array(jnp.log(init_temperature))
        self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)
        self.target_entropy = -action_dim
        
        # Step counter
        self.step = 0
    
    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)
    
    def act(self, obs: jnp.ndarray, rng_key: jax.random.PRNGKey, 
            deterministic: bool = False) -> jnp.ndarray:
        """Select action given observation."""
        obs = jnp.expand_dims(obs, axis=0)
        mu, log_std = self.actor_state.apply_fn(self.actor_state.params, obs)
        
        if deterministic:
            action = jnp.tanh(mu)
        else:
            action, _ = sample_squashed_normal(rng_key, mu, log_std)
        
        action = jnp.clip(action, self.action_range[0], self.action_range[1])
        return action[0]
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(
        self,
        critic_state: TrainState,
        actor_params: flax.core.FrozenDict,
        obs: jnp.ndarray,
        obs_aug1: jnp.ndarray,
        obs_aug2: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_obs: jnp.ndarray,
        next_obs_aug1: jnp.ndarray,
        next_obs_aug2: jnp.ndarray,
        not_done: jnp.ndarray,
        log_alpha: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[TrainState, Dict]:
        """Update critic networks."""
        
        alpha = jnp.exp(log_alpha)
        
        def critic_loss_fn(params):
            # Compute target Q-values using K augmented next states
            key1, key2 = jrandom.split(rng_key)
            
            # Target for first augmented next observation
            next_mu1, next_log_std1 = self.actor_state.apply_fn(actor_params, next_obs_aug1)
            next_action1, next_log_prob1 = sample_squashed_normal(key1, next_mu1, next_log_std1)
            target_q1_1, target_q2_1 = self.critic.apply(
                critic_state.target_params, next_obs_aug1, next_action1
            )
            target_v1 = jnp.minimum(target_q1_1, target_q2_1) - alpha * next_log_prob1
            target_q = reward + not_done * self.discount * target_v1
            
            # K=2: Average with second augmented next observation
            # K=1: Use only first augmentation
            if self.drq_k == 2:
                next_mu2, next_log_std2 = self.actor_state.apply_fn(
                    actor_params, next_obs_aug2
                )
                next_action2, next_log_prob2 = sample_squashed_normal(
                    key2, next_mu2, next_log_std2
                )
                target_q1_2, target_q2_2 = self.critic.apply(
                    critic_state.target_params, next_obs_aug2, next_action2
                )
                target_v2 = jnp.minimum(target_q1_2, target_q2_2) - alpha * next_log_prob2
                target_q2 = reward + not_done * self.discount * target_v2
                # Average over K=2 augmentations
                target_q = (target_q + target_q2) / 2.0
            
            # Current Q-values using M augmented observations
            # First augmentation
            q1_1, q2_1 = self.critic.apply(params, obs_aug1, action)
            q1_loss = jnp.mean((q1_1 - target_q) ** 2)
            q2_loss = jnp.mean((q2_1 - target_q) ** 2)
            
            # M=2: Average loss over two augmented observations
            # M=1: Use only first augmentation
            if self.drq_m == 2:
                q1_2, q2_2 = self.critic.apply(params, obs_aug2, action)
                q1_loss_2 = jnp.mean((q1_2 - target_q) ** 2)
                q2_loss_2 = jnp.mean((q2_2 - target_q) ** 2)
                # Average over M=2 augmentations (as per DrQ algorithm)
                q1_loss = (q1_loss + q1_loss_2) / 2.0
                q2_loss = (q2_loss + q2_loss_2) / 2.0
            
            total_loss = q1_loss + q2_loss
            
            return total_loss, {
                'critic_loss': total_loss,
                'q1': jnp.mean(q1_1),
                'q2': jnp.mean(q2_1),
            }
        
        (loss, info), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_state.params)
        
        critic_state = critic_state.apply_gradients(grads=grads)
        
        return critic_state, info
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_actor(
        self,
        actor_state: TrainState,
        critic_params: flax.core.FrozenDict,
        obs: jnp.ndarray,
        log_alpha: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[TrainState, Dict]:
        """Update actor network."""
        
        alpha = jnp.exp(log_alpha)
        
        def actor_loss_fn(params):
            mu, log_std = self.actor.apply(params, obs)
            action, log_prob = sample_squashed_normal(rng_key, mu, log_std)
            
            q1, q2 = self.critic.apply(critic_params, obs, action)
            q = jnp.minimum(q1, q2)
            
            actor_loss = jnp.mean(alpha * log_prob - q)
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -jnp.mean(log_prob),
            }
        
        (loss, info), grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor_state.params)
        
        actor_state = actor_state.apply_gradients(grads=grads)
        
        return actor_state, info
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_alpha(
        self,
        log_alpha: jnp.ndarray,
        alpha_opt_state,
        actor_params: flax.core.FrozenDict,
        obs: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, any, Dict]:
        """Update temperature parameter."""
        
        def alpha_loss_fn(log_alpha):
            alpha = jnp.exp(log_alpha)
            mu, log_std = self.actor.apply(actor_params, obs)
            _, log_prob = sample_squashed_normal(rng_key, mu, log_std)
            
            alpha_loss = jnp.mean(alpha * (-log_prob - self.target_entropy))
            
            return alpha_loss, {
                'alpha_loss': alpha_loss,
                'alpha': alpha,
            }
        
        (loss, info), grads = jax.value_and_grad(
            alpha_loss_fn, has_aux=True
        )(log_alpha)
        
        updates, alpha_opt_state = self.alpha_optimizer.update(grads, alpha_opt_state)
        log_alpha = optax.apply_updates(log_alpha, updates)
        
        return log_alpha, alpha_opt_state, info
    
    def update(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_obs: jnp.ndarray,
        not_done: jnp.ndarray,
        obs_aug1: jnp.ndarray,
        obs_aug2: jnp.ndarray,
        next_obs_aug1: jnp.ndarray,
        next_obs_aug2: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> Dict:
        """Update agent."""
        
        key1, key2, key3 = jrandom.split(rng_key, 3)
        
        # Update critic
        self.critic_state, critic_info = self._update_critic(
            self.critic_state,
            self.actor_state.params,
            obs, obs_aug1, obs_aug2, action, reward, next_obs, next_obs_aug1, next_obs_aug2, not_done,
            self.log_alpha,
            key1
        )
        
        info = critic_info
        
        # Update actor and alpha
        if self.step % self.actor_update_freq == 0:
            # Use augmented observations for actor (as per DrQ paper)
            # Use first augmentation for actor update
            self.actor_state, actor_info = self._update_actor(
                self.actor_state,
                self.critic_state.params,
                obs_aug1,
                self.log_alpha,
                key2
            )
            info.update(actor_info)
            
            self.log_alpha, self.alpha_opt_state, alpha_info = self._update_alpha(
                self.log_alpha,
                self.alpha_opt_state,
                self.actor_state.params,
                obs_aug1,
                key3
            )
            info.update(alpha_info)
        
        # Update target critic
        if self.step % self.critic_target_update_freq == 0:
            self.critic_state = self.critic_state.replace(
                target_params=soft_update(
                    self.critic_state.params,
                    self.critic_state.target_params,
                    self.tau
                )
            )
        
        self.step += 1
        
        return info
