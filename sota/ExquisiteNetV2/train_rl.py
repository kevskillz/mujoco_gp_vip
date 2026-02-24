import importlib
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from network import SimpleMLPFeature



# Example training
# model.learn(total_timesteps=10000)

def main(gene_id, timesteps=500000):

    module = importlib.import_module(f"models.network_{gene_id}")

    policy_kwargs = module.get_policy_kwargs()
    ppo_kwargs = module.get_ppo_kwargs()


    env = gym.make("HalfCheetah-v4")

    start_time = time.time()

    model = PPO(
        policy=module.get_policy_kwargs()["policy_class"],  # <-- pass the class object
        env=env,
        policy_kwargs=policy_kwargs,  # other kwargs like net_arch
        **ppo_kwargs
    )

    model.learn(total_timesteps=timesteps)

    # Evaluate
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    train_time = time.time() - start_time

    with open(f"results/{gene_id}_results.txt", "w") as f:
        f.write(f"{mean_reward},{std_reward},{train_time}")