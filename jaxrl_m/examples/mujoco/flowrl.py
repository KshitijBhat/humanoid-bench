"""FlowRL: Flow-based Policy for Online Reinforcement Learning (JAX implementation).
 
Implements the FlowRL algorithm for HumanoidBench, following:
  - Paper: "Flow-Based Policy for Online Reinforcement Learning" (Lv et al., 2025)
  - Reference: PyTorch implementation from https://github.com/bytedance/FlowRL
 
Key components:
  1. Flow-based policy (velocity field v_theta) with midpoint Euler ODE integration
  2. Twin Q-critics for current policy pi_theta (standard Bellman backup)
  3. Twin Q-critics + Value network for behavior-optimal policy pi_{beta*}
     (expectile regression on V to estimate upper envelope of Q)
  4. Weighted conditional flow matching (CFM) loss with implicit guidance:
     L = -Q(s, pi(s)) + lambda * w(s,a) * ||v_theta(s, a_t, t) - (a - a_0)||^2
     where w(s,a) = relu(Q_buffer(s,a) - Q(s, pi(s))) passed through exp and clamped
"""
 
import functools
from jaxrl_m.typing import *
 
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks_flowrl import FlowQNetwork, FlowValueNetwork, FlowPolicy
 
import flax
import flax.linen as nn
 
# Constants matching PyTorch FlowRL
CFM_MIN = 1e-3
CFM_MAX = 1.0

def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

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
 
 
class FlowRLAgent(flax.struct.PyTreeNode):
    """FlowRL Agent implemented as a JAX PyTree.
    
    Contains:
        - critic / target_critic: Twin Q for current policy pi_theta
        - critic_buffer / target_critic_buffer: Twin Q for behavior-optimal policy pi_{beta*}
        - value_buffer: V network for pi_{beta*} (expectile regression)
        - policy: Flow-based policy (velocity field)
    """
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    critic_buffer: TrainState
    target_critic_buffer: TrainState
    value_buffer: TrainState
    policy: TrainState
    config: dict = nonpytree_field()
 
    @jax.jit
    def update(agent, batch: Batch):
        new_rng, policy_rng, noise_rng, time_rng = jax.random.split(agent.rng, 4)
 
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        masks = batch['masks']
        next_observations = batch['next_observations']
 
        # ============================================================
        # 1. Update critic for current policy (standard Bellman backup)
        # ============================================================
        def critic_loss_fn(critic_params):
            # Next actions from flow policy
            next_actions = agent.policy.model_def.sample_actions(
                agent.policy.params, next_observations[None] if next_observations.ndim == 1 else next_observations,
                policy_rng
            )
            # Target Q values (min of twin Q)
            next_q1, next_q2 = agent.target_critic(next_observations, next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = rewards + agent.config['discount'] * masks * next_q
 
            # Current Q values
            q1, q2 = agent.critic(observations, actions, params=critic_params)
            critic_loss = jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
 
            return critic_loss, {
                'critic_loss': critic_loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'target_q': target_q.mean(),
            }
 
        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=critic_loss_fn, has_aux=True
        )
 
        # ============================================================
        # 2. Update critic_buffer for behavior-optimal policy
        #    Uses V_buffer for bootstrapping instead of policy sampling
        # ============================================================
        def critic_buffer_loss_fn(critic_buffer_params):
            # Bootstrap with V_buffer (not policy-based)
            target_v = agent.value_buffer(next_observations)
            target_q_buffer = rewards + agent.config['discount'] * masks * target_v
 
            q1_buf, q2_buf = agent.critic_buffer(observations, actions, params=critic_buffer_params)
            loss = jnp.mean((q1_buf - target_q_buffer) ** 2) + jnp.mean((q2_buf - target_q_buffer) ** 2)
 
            q_buffer = jnp.minimum(q1_buf, q2_buf)
            return loss, {
                'critic_buffer_loss': loss,
                'q_buffer': q_buffer.mean(),
            }
 
        new_critic_buffer, critic_buffer_info = agent.critic_buffer.apply_loss_fn(
            loss_fn=critic_buffer_loss_fn, has_aux=True
        )
 
        # ============================================================
        # 3. Update V_buffer via expectile regression
        #    V(s) trained with asymmetric loss: tau-weighted for Q > V
        # ============================================================
        def value_buffer_loss_fn(value_params):
            v_pred = agent.value_buffer(observations, params=value_params)
 
            # Use target_critic_buffer for stable Q targets
            q1_target, q2_target = agent.target_critic_buffer(observations, actions)
            q_target = jnp.minimum(q1_target, q2_target)
 
            # Expectile regression: weight errors asymmetrically
            diff = q_target - v_pred
            weight = jnp.where(
                diff < 0,
                1.0 - agent.config['quantile'],  # underestimate weight
                agent.config['quantile'],          # overestimate weight (larger)
            )
            value_loss = jnp.mean(weight * (diff ** 2))
 
            return value_loss, {
                'value_buffer_loss': value_loss,
                'v_buffer': v_pred.mean(),
            }
 
        new_value_buffer, value_info = agent.value_buffer.apply_loss_fn(
            loss_fn=value_buffer_loss_fn, has_aux=True
        )
 
        # ============================================================
        # 4. Update flow policy (delayed, every target_update_interval steps)
        #    Loss = -Q(s, pi(s)) + lambda * weighted_cfm_loss
        # ============================================================
        # Get Q_buffer values for weighting (use current buffer critic, stop gradient)
        q1_buf_sg, q2_buf_sg = agent.critic_buffer(observations, actions)
        q_buffer_sg = jnp.minimum(q1_buf_sg, q2_buf_sg)
 
        def policy_loss_fn(policy_params):
            # Sample actions from flow policy
            pi_actions = agent.policy.model_def.sample_actions(
                policy_params, observations, policy_rng
            )
 
            # Q-value maximization term
            q1_pi, q2_pi = agent.critic(observations, pi_actions)
            min_q_pi = jnp.minimum(q1_pi, q2_pi)
 
            # Weighted CFM loss (implicit guidance)
            # Weight: relu(Q_buffer - Q_pi) -> exp(w - mean(w)) -> clamp
            advantage = jax.lax.stop_gradient(q_buffer_sg - min_q_pi)
            weights = jnp.maximum(advantage, 0.0)  # relu
            weights = jnp.exp(weights - jnp.mean(weights))
            weights = agent.config['lamda'] * weights
            weights = jnp.clip(weights, CFM_MIN, CFM_MAX)
 
            # CFM: sample random time, interpolate action, predict velocity
            batch_size = observations.shape[0]
            action_0 = jax.random.normal(noise_rng, actions.shape)
            action_0 = jnp.clip(action_0, -1.0, 1.0)
 
            t = jax.random.uniform(time_rng, (batch_size, 1))
            action_t = t * actions + (1.0 - t) * action_0
            target_velocity = actions - action_0
 
            predicted_velocity = agent.policy(observations, action_t, t, params=policy_params)
            cfm_loss_per_sample = jnp.mean((predicted_velocity - target_velocity) ** 2, axis=-1)
 
            # Combined loss: -Q + weighted_cfm
            policy_loss = jnp.mean(-min_q_pi + weights * cfm_loss_per_sample)
 
            return policy_loss, {
                'policy_loss': policy_loss,
                'q_pi': min_q_pi.mean(),
                'cfm_loss': cfm_loss_per_sample.mean(),
                'weights_mean': weights.mean(),
            }
 
        # Delayed policy update
        should_update_policy = (agent.critic.step % agent.config['target_update_interval'] == 0)
 
        new_policy, policy_info = jax.lax.cond(
            should_update_policy,
            lambda: agent.policy.apply_loss_fn(loss_fn=policy_loss_fn, has_aux=True),
            lambda: (agent.policy, {
                'policy_loss': 0.0,
                'q_pi': 0.0,
                'cfm_loss': 0.0,
                'weights_mean': 0.0,
            }),
        )
 
        # ============================================================
        # 5. Target network updates (also delayed)
        # ============================================================
        new_target_critic = jax.lax.cond(
            should_update_policy,
            lambda: target_update(new_critic, agent.target_critic, agent.config['tau']),
            lambda: agent.target_critic,
        )
        new_target_critic_buffer = jax.lax.cond(
            should_update_policy,
            lambda: target_update(new_critic_buffer, agent.target_critic_buffer, agent.config['tau']),
            lambda: agent.target_critic_buffer,
        )
 
        return agent.replace(
            rng=new_rng,
            critic=new_critic,
            target_critic=new_target_critic,
            critic_buffer=new_critic_buffer,
            target_critic_buffer=new_target_critic_buffer,
            value_buffer=new_value_buffer,
            policy=new_policy,
        ), {
            **critic_info,
            **critic_buffer_info,
            **value_info,
            **policy_info,
        }
 
    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        """Sample actions using the flow policy.
        
        Args:
            observations: [obs_dim] or [B, obs_dim]
            seed: JAX PRNG key
            temperature: Not used for flow policy (kept for API compatibility).
                         When temperature=0 we still use the same deterministic ODE
                         but with fixed noise seed for evaluation.
        """
        # Ensure batch dimension
        if observations.ndim == 1:
            observations = observations[None]
            squeeze = True
        else:
            squeeze = False
 
        actions = agent.policy.model_def.sample_actions(
            agent.policy.params, observations, seed
        )
        actions = jnp.clip(actions, -1, 1)
 
        if squeeze:
            actions = actions[0]
        return actions
 
 
def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    # Optimizer
    lr: float = 3e-4,
    # Architecture
    hidden_dim: int = 512,
    # RL
    discount: float = 0.99,
    tau: float = 0.95,
    # FlowRL specific
    flow_steps: int = 1,
    quantile: float = 0.9,
    lamda: float = 0.1,
    target_update_interval: int = 2,
    **kwargs,
):
    """Create a FlowRL agent.
    
    Args:
        seed: Random seed
        observations: Example observation batch [1, obs_dim]
        actions: Example action batch [1, action_dim]
        lr: Learning rate for all networks
        hidden_dim: Hidden dimension for all networks
        discount: Discount factor gamma
        tau: Soft target update rate (FlowRL uses 0.95, much faster than SAC's 0.005)
        flow_steps: Number of ODE integration steps (default 1)
        quantile: Expectile factor for V_buffer (0.9 default, 0.8 for unseen tasks)
        lamda: Lagrangian multiplier for CFM regularization weight
        target_update_interval: Delayed policy/target update frequency
    """
    print(f'FlowRL extra kwargs: {kwargs}')
 
    rng = jax.random.PRNGKey(seed)
    rng, critic_key, critic_buf_key, value_key, policy_key = jax.random.split(rng, 5)
 
    action_dim = actions.shape[-1]
 
    # --- Critic for current policy ---
    critic_def = FlowQNetwork(hidden_dim=hidden_dim)
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=lr))
    target_critic = TrainState.create(critic_def, critic_params)
 
    # --- Critic for behavior-optimal policy (buffer) ---
    critic_buffer_def = FlowQNetwork(hidden_dim=hidden_dim)
    critic_buffer_params = critic_buffer_def.init(critic_buf_key, observations, actions)['params']
    critic_buffer = TrainState.create(critic_buffer_def, critic_buffer_params, tx=optax.adam(learning_rate=lr))
    target_critic_buffer = TrainState.create(critic_buffer_def, critic_buffer_params)
 
    # --- Value network for behavior-optimal policy ---
    value_def = FlowValueNetwork(hidden_dim=hidden_dim)
    value_params = value_def.init(value_key, observations)['params']
    value_buffer = TrainState.create(value_def, value_params, tx=optax.adam(learning_rate=lr))
 
    # --- Flow policy ---
    policy_def = FlowPolicy(action_dim=action_dim, hidden_dim=hidden_dim, flow_steps=flow_steps)
    # Initialize with dummy inputs: (state, action, time)
    dummy_t = jnp.zeros((observations.shape[0], 1))
    policy_params = policy_def.init(policy_key, observations, actions, dummy_t)['params']
    policy = TrainState.create(policy_def, policy_params, tx=optax.adam(learning_rate=lr))
 
    config = flax.core.FrozenDict(dict(
        discount=discount,
        tau=tau,
        flow_steps=flow_steps,
        quantile=quantile,
        lamda=lamda,
        target_update_interval=target_update_interval,
    ))
 
    return FlowRLAgent(
        rng=rng,
        critic=critic,
        target_critic=target_critic,
        critic_buffer=critic_buffer,
        target_critic_buffer=target_critic_buffer,
        value_buffer=value_buffer,
        policy=policy,
        config=config,
    )
 
 
def get_default_config():
    """Default FlowRL hyperparameters matching the paper."""
    import ml_collections
 
    return ml_collections.ConfigDict({
        'lr': 3e-4,
        'hidden_dim': 512,
        'discount': 0.99,
        'tau': 0.95,
        'flow_steps': 1,
        'quantile': 0.9,           # 0.8 recommended for unseen HumanoidBench tasks
        'lamda': 0.1,
        'target_update_interval': 2,
    })