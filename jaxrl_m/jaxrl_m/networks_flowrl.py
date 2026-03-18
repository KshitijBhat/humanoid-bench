from jaxrl_m.typing import *

import flax.linen as nn
import jax.numpy as jnp

import flax.linen as nn
import jax.numpy as jnp

import jax
from jaxrl_m.networks import default_init

###############################
#
#  FlowRL-Specific Networks
#
###############################
 
 
class ValueBlock(nn.Module):
    """Value network block matching FlowRL's architecture:
    LayerNorm -> Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear -> LayerNorm
    """
    hidden_dim: int
 
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        return x
 
 
class FlowQNetwork(nn.Module):
    """Twin Q-network with LayerNorm blocks, matching FlowRL's QNetwork.
    Architecture per Q: Linear(in, hidden) -> ValueBlock -> Linear(hidden, 1)
    """
    hidden_dim: int = 512
 
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray):
        x = jnp.concatenate([observations, actions], -1)
 
        # Q1
        x1 = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x1 = ValueBlock(self.hidden_dim)(x1)
        q1 = nn.Dense(1, kernel_init=default_init())(x1)
        q1 = jnp.squeeze(q1, -1)
 
        # Q2
        x2 = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x2 = ValueBlock(self.hidden_dim)(x2)
        q2 = nn.Dense(1, kernel_init=default_init())(x2)
        q2 = jnp.squeeze(q2, -1)
 
        return q1, q2
 
 
class FlowValueNetwork(nn.Module):
    """Value network V(s) for behavior-optimal policy estimation.
    Architecture: Linear(in, hidden) -> ValueBlock -> Linear(hidden, 1)
    """
    hidden_dim: int = 512
 
    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(observations)
        x = ValueBlock(self.hidden_dim)(x)
        x = nn.Dense(1, kernel_init=default_init())(x)
        return jnp.squeeze(x, -1)
 
 
class FlowPolicy(nn.Module):
    """Flow-based policy: velocity field v_theta(s, a_t, t).
    
    Architecture matches the PyTorch FlowRL:
        Linear(state_dim + action_dim + 1, hidden) -> LayerNorm -> ELU
        -> Linear(hidden, hidden) -> LayerNorm -> ELU
        -> Linear(hidden, action_dim)
    
    The policy generates actions by integrating this velocity field from t=0 to t=1
    using the midpoint Euler method, starting from Gaussian noise.
    """
    action_dim: int
    hidden_dim: int = 512
    flow_steps: int = 1
 
    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Predict velocity field v_theta(state, action_t, t)."""
        x = jnp.concatenate([state, action, t], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        return x
 
    def ode_step(self, params, state: jnp.ndarray, action: jnp.ndarray,
                 time_start: jnp.ndarray, time_end: jnp.ndarray) -> jnp.ndarray:
        """Midpoint Euler integration step."""
        velocity_start = self.apply({"params": params}, state, action, time_start)
        dt = time_end - time_start
        intermediate = action + velocity_start * (dt / 2.0)
        time_mid = time_start + dt / 2.0
        velocity_mid = self.apply({"params": params}, state, intermediate, time_mid)
        action_next = action + velocity_mid * dt
        return action_next
 
    def sample_actions(self, params, state: jnp.ndarray, rng: PRNGKey) -> jnp.ndarray:
        """Generate actions by integrating velocity field from t=0 to t=1.
        
        Args:
            params: Network parameters
            state: Observation batch [B, obs_dim]
            rng: Random key for initial noise sampling
            
        Returns:
            actions: Generated actions [B, action_dim], tanh-squashed to [-1, 1]
        """
        batch_size = state.shape[0]
        time_step_size = 1.0 / self.flow_steps
 
        # Sample initial noise and clamp to [-1, 1]
        action = jax.random.normal(rng, (batch_size, self.action_dim))
        action = jnp.clip(action, -1.0, 1.0)
 
        time_start = jnp.zeros((batch_size, 1))
 
        def body_fn(carry, _):
            action, time_start = carry
            time_end = time_start + time_step_size
            action = self.ode_step(params, state, action, time_start, time_end)
            return (action, time_end), None
 
        (action, _), _ = jax.lax.scan(body_fn, (action, time_start), None, length=self.flow_steps)
 
        # Apply tanh squashing to bound actions in [-1, 1]
        action = jnp.tanh(action)
        return action