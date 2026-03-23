"""Training launcher for FlowRL on HumanoidBench.
 
Usage:
    python run_flowrl.py --env_name h1hand-walk-v0 --seed 0
    python run_flowrl.py --env_name h1hand-walk-v0 --seed 0 --config.quantile 0.8
    python run_flowrl.py --env_name h1hand-crawl-v0 --max_steps 2000000
"""
 
import os
import pickle
import time
from absl import app, flags
from functools import partial
 
import numpy as np
import jax
import tqdm
import gymnasium as gym
from ml_collections import config_flags
from flax.training import checkpoints
import imageio
from tensorboardX import SummaryWriter
 
import humanoid_bench
import flowrl as learner
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer
 
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'h1hand-walk-v0', 'HumanoidBench environment name.')
flags.DEFINE_string('save_dir', './results', 'Base logging directory.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 4000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 25000, 'Save interval.')
flags.DEFINE_integer('render_interval', 250000, 'Render interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('start_steps', int(5000), 'Number of initial exploration steps.')
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)
 
def render(policy_fn, env) -> tuple:
    """Render one episode and return frames."""
    frames = []
    observation, info = env.reset()
    done = False
 
    episode_return = 0.0
    episode_length = 0
    pbar = tqdm.tqdm(total=1000, desc='render', ncols=80)
    while not done:
        frames.append(env.render())
        action = policy_fn(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1
        done = terminated or truncated
        pbar.update(1)
    pbar.close()
 
    frames = np.array(frames)
    return frames, episode_return, episode_length

def sample_flowrl_batch(replay_buffer, batch_size: int):
    """
    Samples 80% uniformly from the whole buffer and 20% from the most recent 2048 transitions,
    matching the original PyTorch FlowRL implementation.
    """
    buffer_len = replay_buffer.size
    pointer = replay_buffer.pointer
    
    # Calculate 80/20 split
    num_uniform = int(batch_size * 0.8)
    num_recent = batch_size - num_uniform
    
    # 1. Uniform samples
    uniform_indices = np.random.choice(buffer_len, num_uniform, replace=False)
    
    # 2. Recent samples (from a rolling window of max 2048)
    recent_window = min(2048, buffer_len)
    
    if pointer - recent_window >= 0:
        recent_indices_raw = np.arange(pointer - recent_window, pointer)
    else:
        # Handle circular buffer wrap-around
        # When wrapped, the oldest valid data in the window is at the end of the current buffer
        part1 = np.arange(pointer - recent_window + buffer_len, buffer_len)
        part2 = np.arange(0, pointer)
        recent_indices_raw = np.concatenate([part1, part2])
        
    actual_num_recent = min(num_recent, recent_window)
    recent_indices = np.random.choice(recent_indices_raw, actual_num_recent, replace=False)
    
    # 3. Combine indices
    all_indices = np.concatenate([uniform_indices, recent_indices])
    
    # Leverage jaxrl_m's built-in subset sampling
    return replay_buffer.sample(batch_size, indx=all_indices)
 
 
def main(_):
    # Setup config
    # config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)
 
    # Setup directories
    run_id = f"flowrl_{FLAGS.env_name}_{FLAGS.seed}_{int(time.time())}"
    run_dir = os.path.join(os.path.abspath(FLAGS.save_dir), run_id)
    video_dir = os.path.join(run_dir, "videos")
 
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
 
    print(f"Logging experiment to: {run_dir}")
    print(f"Config: {FLAGS.config}")
 
    # TensorBoard
    writer = SummaryWriter(log_dir=run_dir)
 
    # Save config
    config_dict = FLAGS.flag_values_dict()
    try:
        with open(os.path.join(run_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config_dict, f)
    except Exception as e:
        print(f"Warning: Could not pickle config: {e}")
 
    # ============================
    # Environment setup
    # ============================
    kwargs = {}
    if FLAGS.env_name == 'h1hand-package-v0':
        kwargs = {'policy_path': None}
 
    env = EpisodeMonitor(gym.make(FLAGS.env_name, **kwargs))
    eval_env = EpisodeMonitor(gym.make(FLAGS.env_name, **kwargs))
 
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
 
    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )
 
    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))
 
    # ============================
    # Agent creation
    # ============================
    agent = learner.create_learner(
        FLAGS.seed,
        example_transition['observations'][None],
        example_transition['actions'][None],
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )
 
    exploration_metrics = dict()
    obs, _ = env.reset()
    exploration_rng = jax.random.PRNGKey(0)
 
    print(f"\n>>>> Training FlowRL on {FLAGS.env_name}, seed={FLAGS.seed}")
    print(f">>>> Max steps: {FLAGS.max_steps}, batch size: {FLAGS.batch_size}")
    print(f">>>> Flow steps: {FLAGS.config.flow_steps}, quantile: {FLAGS.config.quantile}, "
          f"lambda: {FLAGS.config.lamda}, tau: {FLAGS.config.tau}\n")
 
    # ============================
    # Training loop
    # ============================
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):
 
        # Exploration: random actions for start_steps, then policy
        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs, seed=key)
 
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        mask = float(not terminated or truncated)
 
        replay_buffer.add_transition(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
        ))
        obs = next_obs
 
        if done:
            exploration_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}
            obs, _ = env.reset()
 
        if replay_buffer.size < FLAGS.start_steps:
            continue
 
        # Update agent
        # batch = replay_buffer.sample(FLAGS.batch_size)
        batch = sample_flowrl_batch(replay_buffer, FLAGS.batch_size)
        agent, update_info = agent.update(batch)
 
        # Logging
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            for k, v in train_metrics.items():
                writer.add_scalar(k, v, i)
            for k, v in exploration_metrics.items():
                writer.add_scalar(k, v, i)
            exploration_metrics = dict()
 
        # Evaluation
        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
 
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            eval_metrics["results/return"] = eval_info["episode.return"]
            eval_metrics["results/episode_length"] = eval_info["episode.length"]
 
            eval_success = eval_info.get("success", 0)
            eval_metrics["results/success"] = eval_success
 
            eval_success_subtasks = eval_info.get("success_subtasks", 0)
            eval_metrics["results/success_subtasks"] = eval_success_subtasks
 
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, i)
 
            print(f"\n[Step {i}] eval_return={eval_info['episode.return']:.2f}, "
                  f"eval_length={eval_info['episode.length']:.0f}, "
                  f"success={eval_success:.3f}")
 
        # Video rendering
        if i % FLAGS.render_interval == 0:
            try:
                render_frames, render_return, render_length = render(policy_fn, eval_env)
                video_path = os.path.join(video_dir, f"step_{i}.mp4")
                imageio.mimsave(video_path, render_frames, fps=30)
                print(f"\nSaved video to {video_path} (return={render_return:.3f}, length={render_length})")
            except Exception as e:
                print(f"\nVideo rendering failed: {e}")
 
        # Checkpointing
        if i % FLAGS.save_interval == 0:
            checkpoints.save_checkpoint(run_dir, agent, i)
 
    writer.close()
    print(f"\nTraining complete. Results saved to {run_dir}")
 
 
if __name__ == '__main__':
    app.run(main)