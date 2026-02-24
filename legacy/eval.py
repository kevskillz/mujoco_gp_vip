# eval.py
import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env():
    return gym.make("HalfCheetah-v4")  # or "HalfCheetah-v5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/halfcheetah_forward_ppo")
    p.add_argument("--vecnorm", default="models/vecnorm_forward.pkl")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(args.vecnorm, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model, env=venv)

    ep_returns = []
    for _ in range(args.episodes):
        obs = venv.reset()
        done = False
        total = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)

            total += float(reward[0])   # <-- FIX
            done = bool(dones[0])       # <-- FIX

        ep_returns.append(total)

    print(f"mean_return={np.mean(ep_returns):.2f} std={np.std(ep_returns):.2f}")


if __name__ == "__main__":
    main()
