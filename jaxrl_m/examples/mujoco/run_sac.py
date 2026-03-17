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
import sac as learner
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'h1hand-kitchen-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './results', 'Base logging directory.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 4000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 25000, 'Save interval.')
flags.DEFINE_integer('render_interval', 250000, 'Render interval.')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e7), 'Number of training steps.')
flags.DEFINE_integer('start_steps', int(1e4), 'Number of initial exploration steps.')


def render(policy_fn, env) -> tuple:
    frames = []
    observation, info = env.reset()
    done = False

    episode_return = 0.0
    episode_length = 0
    pbar = tqdm.tqdm(total=1000, desc='render', ncols=80)
    while not done:
        frames.append(env.render())  # [H, W, C]
        action = policy_fn(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1
        done = terminated or truncated
        pbar.update(1)
    pbar.close()

    frames = np.array(frames)  # [T, H, W, C]
    # Removed transposition to [T, C, H, W] so imageio can save it natively
    return frames, episode_return, episode_length

def main(_):
    # Setup standard config
    config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)

    # Setup local directories based on environment and timestamp
    run_id = f"sac_{FLAGS.env_name}_{int(time.time())}"
    run_dir = os.path.join(FLAGS.save_dir, run_id)
    video_dir = os.path.join(run_dir, "videos")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Logging experiment to: {run_dir}")

    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=run_dir)

    # Save config dict locally
    config_dict = FLAGS.flag_values_dict()
    # Filter out unpicklable absl flag objects if necessary, or just save standard config
    try:
        with open(os.path.join(run_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config_dict, f)
    except Exception as e:
        print(f"Warning: Could not pickle full config dict: {e}")

    kwargs = {}
    if FLAGS.env_name == 'h1hand-package-v0':
        kwargs = {'policy_path': None}
    
    env = EpisodeMonitor(gym.make(FLAGS.env_name, **kwargs))
    eval_env = EpisodeMonitor(gym.make(FLAGS.env_name, **kwargs))

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    agent = learner.create_learner(FLAGS.seed,
                    example_transition['observations'][None],
                    example_transition['actions'][None],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.config)

    exploration_metrics = dict()
    obs, _ = env.reset()
    exploration_rng = jax.random.PRNGKey(0)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):

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

        batch = replay_buffer.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)

        # Logging to TensorBoard
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            for k, v in train_metrics.items():
                writer.add_scalar(k, v, i)
            for k, v in exploration_metrics.items():
                writer.add_scalar(k, v, i)
            exploration_metrics = dict()

        # Evaluation Logging
        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            eval_metrics["results/return"] = eval_info["episode.return"]
            eval_metrics["results/episode_length"] = eval_info["episode.length"]
            
            eval_success = 0
            if "success" in eval_info:
                eval_success = eval_info["success"]
            eval_metrics["results/success"] = eval_success
            
            eval_success_subtasks = 0
            if "success_subtasks" in eval_info:
                eval_success_subtasks = eval_info["success_subtasks"]
            eval_metrics["results/success_subtasks"] = eval_success_subtasks
            
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, i)

        # Video Rendering
        if i % FLAGS.render_interval == 0:
            render_frames, render_return, render_length = render(policy_fn, eval_env)
            video_path = os.path.join(video_dir, f"step_{i}.mp4")
            imageio.mimsave(video_path, render_frames, fps=30)
            print(f"\nSaved video to {video_path} (return={render_return:.3f}, length={render_length})")

        # Checkpointing
        if i % FLAGS.save_interval == 0:
            checkpoints.save_checkpoint(run_dir, agent, i)

    writer.close()

if __name__ == '__main__':
    app.run(main)