import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def make_env(seed: int, rank: int):
    def _init():
        env = gym.make("HalfCheetah-v4")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    seed = 0
    n_envs = 8
    timesteps = 500_000  # reduce to 200_000 for a quick smoke test
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(seed, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=timesteps)

    model_path = os.path.join(out_dir, "halfcheetah_forward_ppo")
    model.save(model_path)
    env.save(os.path.join(out_dir, "vecnorm_forward.pkl"))
    env.close()

    print(f"Saved: {model_path}.zip and vecnorm_forward.pkl")


if __name__ == "__main__":
    main()
