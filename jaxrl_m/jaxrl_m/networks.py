"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from jaxrl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

from jaxrl_m.typing import *

import flax.linen as nn
import jax.numpy as jnp

import distrax
import flax.linen as nn
import jax.numpy as jnp

import jax

###############################
#
#  Common Networks
#
###############################


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


###############################
#
#
#  Common RL Networks
#
###############################


class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP((*self.hidden_dims, self.n_actions), activations=self.activations)(
            observations
        )


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


###############################
#
#
#   Meta Networks for Encoders
#
###############################


def get_latent(
    encoder: nn.Module, observations: Union[jnp.ndarray, Dict[str, jnp.ndarray]]
):
    """

    Get latent representation from encoder. If observations is a dict
        a state and image component, then concatenate the latents.

    """
    if encoder is None:
        return observations

    elif isinstance(observations, dict):
        return jnp.concatenate(
            [encoder(observations["image"]), observations["state"]], axis=-1
        )

    else:
        return encoder(observations)


class WithEncoder(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class ActorCritic(nn.Module):
    """Combines FC networks with encoders for actor, critic, and value.

    Note: You can share encoder parameters between actor and critic by passing in the same encoder definition for both.

    Example:

        encoder_def = ImpalaEncoder()
        actor_def = Policy(...)
        critic_def = Critic(...)
        # This will share the encoder between actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': encoder_def},
            networks={'actor': actor_def, 'critic': critic_def}
        )
        # This will have separate encoders for actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': copy.deepcopy(encoder_def)},
            networks={'actor': actor_def, 'critic': critic_def}
        )
    """

    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def actor(self, observations, **kwargs):
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)

    def critic(self, observations, actions, **kwargs):
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)

    def value(self, observations, **kwargs):
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)

    def __call__(self, observations, actions):
        rets = {}
        if "actor" in self.networks:
            rets["actor"] = self.actor(observations)
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets


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