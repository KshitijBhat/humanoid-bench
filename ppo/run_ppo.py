import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

import humanoid_bench
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import imageio
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
# Removed --wandb_entity argument
ARGS = parser.parse_args()


def make_env(
    rank,
    seed=0
):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        
        env = gym.make(ARGS.env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        
        env.action_space.seed(ARGS.seed + rank)
        
        return env

    return _init

class EvalCallback(BaseCallback):
    
    def __init__(self, run_id: str, eval_every: int = 100000, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)

        self.eval_every = eval_every
        self.eval_env = DummyVecEnv([make_env(1)])
        self.video_dir = f"results/{run_id}/videos"
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        
        if self.num_timesteps % self.eval_every == 0:
            self.record_video()

        return True
    
    def record_video(self) -> None:

        print("recording video")
        video = []

        obs = self.eval_env.reset()
        for i in range(1000):
            action = self.model.predict(obs, deterministic=True)[0]
            obs, _, _, _ = self.eval_env.step(action)
            
            # Keep as (H, W, C) for local video saving rather than (C, H, W)
            pixels = self.eval_env.render() 
            video.append(pixels)

        # Save locally using imageio
        video_path = os.path.join(self.video_dir, f"step_{self.num_timesteps}.mp4")
        imageio.mimsave(video_path, video, fps=100)
        print(f"Video saved to {video_path}")


class LogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.aux_rewards = {}
        self.aux_returns = {}
        for key in info_keywords:
            self.aux_rewards[key] = np.zeros(ARGS.num_envs)
            self.aux_returns[key] = deque(maxlen=100)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            for key in self.aux_rewards.keys():
                self.aux_rewards[key][idx] += infos[idx][key]

            if self.locals['dones'][idx]:
                for key in self.aux_rewards.keys():
                    self.aux_returns[key].append(self.aux_rewards[key][idx])
                    self.aux_rewards[key][idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.aux_returns.keys():
            self.logger.record("aux_returns_{}/mean".format(key), np.mean(self.aux_returns[key]))


class EpisodeLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.returns_info = {
            "results/return": [],
            "results/episode_length": [],
            "results/success": [],
            "results/success_subtasks": [],
        }

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            curr_info = infos[idx]
            if "episode" in curr_info:
                self.returns_info["results/return"].append(curr_info["episode"]["r"])
                self.returns_info["results/episode_length"].append(curr_info["episode"]["l"])
                cur_info_success = 0
                if "success" in curr_info:
                    cur_info_success = curr_info["success"]
                self.returns_info["results/success"].append(cur_info_success)
                cur_info_success_subtasks = 0
                if "success_subtasks" in curr_info:
                    cur_info_success_subtasks = curr_info["success_subtasks"]
                self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.returns_info.keys():
            if self.returns_info[key]:
                self.logger.record(key, np.mean(self.returns_info[key]))
                self.returns_info[key] = []


def main(argv):

    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    
    # Generate a unique run ID based on time and environment name
    run_id = f"ppo_{ARGS.env_name}_{int(time.time())}"
    
    # Setup directories for local saving
    tensorboard_log_dir = f"runs/{run_id}"
    model_save_dir = f"models/{run_id}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Use standard SB3 CheckpointCallback instead of WandbCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // ARGS.num_envs, 1), # Account for num_envs
        save_path=model_save_dir,
        name_prefix="model"
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tensorboard_log_dir, 
        learning_rate=float(ARGS.learning_rate), 
        batch_size=512
    )
    
    model.learn(
        total_timesteps=ARGS.max_steps, 
        log_interval=1, 
        callback=[
            checkpoint_callback, 
            EvalCallback(run_id=run_id), 
            LogCallback(info_keywords=[]), 
            EpisodeLogCallback()
        ]
    )
    
    final_model_path = os.path.join(model_save_dir, "ppo_final")
    model.save(final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main(None)