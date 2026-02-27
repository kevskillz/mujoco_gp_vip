"""
Evaluate a trained PPO HalfCheetah model and optionally record a video.

Usage:
    # Evaluate only (prints reward stats):
    python evaluate.py

    # Evaluate + record video:
    python evaluate.py --record

    # Use a custom model path:
    python evaluate.py --model legacy/models/halfcheetah_forward_ppo.zip --vecnorm legacy/models/vecnorm_forward.pkl
"""

import os

# Use EGL for headless rendering (must be set BEFORE importing mujoco/gymnasium)
os.environ["MUJOCO_GL"] = "egl"

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate(model, env, n_episodes=10):
    """Run n evaluation episodes and return rewards."""
    rewards = []
    episode_lengths = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Episode {ep+1:2d}: reward = {total_reward:8.2f}  steps = {steps}")
    return rewards, episode_lengths


def record_video(model, env_id, vecnorm_path, output_dir="videos", n_episodes=3):
    """Record video of the agent using gymnasium's RecordVideo wrapper."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a fresh env with rgb_array rendering for recording
    rec_env = gym.make(env_id, render_mode="rgb_array")
    rec_env = gym.wrappers.RecordVideo(rec_env, output_dir, 
                                        episode_trigger=lambda x: True,
                                        name_prefix="halfcheetah_ppo")
    venv = DummyVecEnv([lambda: rec_env])

    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    for ep in range(n_episodes):
        obs = venv.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            total_reward += reward[0]
        print(f"  Recorded episode {ep+1}: reward = {total_reward:.2f}")

    venv.close()
    print(f"\n📹 Videos saved to: {os.path.abspath(output_dir)}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained HalfCheetah PPO model")
    parser.add_argument("--model", type=str, default="legacy/models/halfcheetah_forward_ppo.zip",
                        help="Path to the trained model .zip file")
    parser.add_argument("--vecnorm", type=str, default="legacy/models/vecnorm_forward.pkl",
                        help="Path to VecNormalize .pkl file (optional)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true",
                        help="Record video of the agent")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")
    args = parser.parse_args()

    env_id = "HalfCheetah-v4"

    # --- Load model ---
    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    # Set up environment with VecNormalize if available
    env = gym.make(env_id)
    venv = DummyVecEnv([lambda: env])

    if args.vecnorm and os.path.exists(args.vecnorm):
        print(f"Loading VecNormalize from: {args.vecnorm}")
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model, env=venv)

    # --- Evaluate ---
    print(f"\n{'='*50}")
    print(f"  Evaluating {args.episodes} episodes on {env_id}")
    print(f"{'='*50}")
    rewards, lengths = evaluate(model, venv, n_episodes=args.episodes)

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Mean reward:    {np.mean(rewards):8.2f}")
    print(f"  Std reward:     {np.std(rewards):8.2f}")
    print(f"  Min reward:     {np.min(rewards):8.2f}")
    print(f"  Max reward:     {np.max(rewards):8.2f}")
    print(f"  Mean ep length: {np.mean(lengths):8.1f}")
    print(f"{'='*50}\n")

    # --- Record video ---
    if args.record:
        print("Recording video episodes...")
        record_video(model, env_id, args.vecnorm, 
                     output_dir=args.video_dir, n_episodes=3)

    venv.close()


if __name__ == "__main__":
    main()
