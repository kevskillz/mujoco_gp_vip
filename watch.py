import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


import imageio

MODELS_DIR = "models"
VIDEOS_DIR = "videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)

fps = 60
duration_sec = 10
max_steps = fps * duration_sec

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.zip')]
print(f"Found {len(model_files)} model(s):", model_files)


for model_file in model_files:
    model_path = os.path.join(MODELS_DIR, model_file)
    seed_name = os.path.splitext(os.path.basename(model_path))[0]
    gif_filename = os.path.join(VIDEOS_DIR, f"{seed_name}.gif")
    if os.path.exists(gif_filename):
        print(f"Skipping {model_file}: {gif_filename} already exists.")
        continue
    print(f"Processing {model_file} -> {gif_filename}")

    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    venv = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=venv)
    obs = venv.reset()
    frames = []
    step_count = 0
    done_flag = False
    while step_count < max_steps and not done_flag:
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        if isinstance(done, (list, tuple, np.ndarray)):
            done_flag = any(done)
        else:
            done_flag = done
        step_count += 1
    imageio.mimsave(gif_filename, frames, fps=27)
    print(f"Saved 10-second GIF as {gif_filename}")

venv.close()
