import os
import numpy as np
import jax
import gymnasium as gym
import imageio
import tqdm
from functools import partial
from flax.training import checkpoints

import humanoid_bench
import sac as learner
from jaxrl_m.evaluation import supply_rng, EpisodeMonitor


# =========================
# Config
# =========================
ENV_NAME = "g1-window-v0"
CHECKPOINT_DIR = "/home/kshitij/humanoid-bench/results/sac_g1-window-v0_1774279074"   # <-- change this
CHECKPOINT_STEP = 5000000  # None = latest

VIDEO_PATH = CHECKPOINT_DIR + "/rollout_hd.mp4"
SEED = 0

# High-res settings
WIDTH = 1920
HEIGHT = 1080
FPS = 30


# =========================
# Render function
# =========================
def render(policy_fn, env):
    frames = []
    obs, _ = env.reset()
    done = False

    pbar = tqdm.tqdm(desc="Rendering", ncols=100)

    while not done:
        frame = env.render()  # [H, W, C]
        frames.append(frame)

        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        pbar.update(1)

    pbar.close()
    return np.array(frames)


# =========================
# Main
# =========================
def main():

    # ---- Create HIGH-RES env ----
    kwargs = {}
    if ENV_NAME == 'h1hand-package-v0':
        kwargs = {'policy_path': None}

    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        width=WIDTH,
        height=HEIGHT,
        camera_name="track",  # IMPORTANT for humanoids
        **kwargs
    )

    env = EpisodeMonitor(env)

    # ---- Force resolution (in case env ignores kwargs) ----
    try:
        env.unwrapped.mujoco_renderer.width = WIDTH
        env.unwrapped.mujoco_renderer.height = HEIGHT
    except:
        try:
            env.unwrapped.renderer._width = WIDTH
            env.unwrapped.renderer._height = HEIGHT
        except:
            print("Warning: Could not force renderer resolution")

    # ---- Create dummy input for agent init ----
    obs_sample = env.observation_space.sample()
    act_sample = env.action_space.sample()

    agent = learner.create_learner(
        SEED,
        obs_sample[None],
        act_sample[None],
    )

    # ---- Load checkpoint ----
    agent = checkpoints.restore_checkpoint(
        CHECKPOINT_DIR,
        target=agent,
        step=CHECKPOINT_STEP
    )

    print("Loaded checkpoint.")

    # ---- Deterministic policy (IMPORTANT) ----
    policy_fn = partial(
        supply_rng(agent.sample_actions),
        temperature=0.0  # no exploration noise
    )

    # ---- Render rollout ----
    frames = render(policy_fn, env)

    print("Frames shape:", frames.shape)  # should be [T, 1080, 1920, 3]

    # ---- Save high-quality video ----
    imageio.mimsave(
        VIDEO_PATH,
        frames,
        fps=FPS,
        codec="libx264",
        bitrate="10M",          # HIGH quality
        macro_block_size=None   # prevents resizing
    )

    print(f"Saved HD video to {VIDEO_PATH}")


if __name__ == "__main__":
    main()