import os
import sys
import argparse
import importlib
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from network import get_policy_kwargs, get_ppo_kwargs   

def main(gene_id, timesteps=500000):

    # module = importlib.import_module(f"network.py")

    policy_kwargs = get_policy_kwargs()
    ppo_kwargs = get_ppo_kwargs()

    # Extract the policy class and remove it from policy_kwargs
    # so it doesn't get passed twice to PPO
    policy_class = policy_kwargs.pop("policy_class", "MlpPolicy")

    env = gym.make("HalfCheetah-v4")

    start_time = time.time()

    # For small smoke-test runs, clamp n_steps so PPO doesn't collect
    # more env steps than total_timesteps (avoids wasting time on login nodes)
    if timesteps <= 2048:
        ppo_kwargs["n_steps"] = min(ppo_kwargs.get("n_steps", 2048), max(timesteps, 64))
        # Also ensure batch_size <= n_steps
        ppo_kwargs["batch_size"] = min(ppo_kwargs.get("batch_size", 64), ppo_kwargs["n_steps"])

    print("running")
    try:
        model = PPO(
            policy=policy_class,
            env=env,
            policy_kwargs=policy_kwargs,  # remaining kwargs like net_arch
            **ppo_kwargs
        )
    except Exception as e:
        # If the LLM-generated architecture is broken, write error results and exit
        print(f"ERROR creating model: {e}")
        train_time = time.time() - start_time
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{gene_id}_results.txt"), "w") as f:
            f.write(f"-999999.0,0.0,{train_time}")
        print(f"Mean reward: -999999.0, Std: 0.0, Time: {train_time:.1f}s")
        print("Job Done")
        return

    model.learn(total_timesteps=timesteps)

    # Evaluate
    if timesteps <= 10000:
        num_eval_episodes = 1
        max_eval_steps = 200  # quick smoke test
    else:
        num_eval_episodes = 10
        max_eval_steps = 1000
    rewards = []
    for _ in range(num_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done and step_count < max_eval_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    train_time = time.time() - start_time

    # Save results under the SOTA_ROOT/results directory (where run_improved.py expects them)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{gene_id}_results.txt"), "w") as f:
        f.write(f"{mean_reward},{std_reward},{train_time}")

    print(f"Mean reward: {mean_reward}, Std: {std_reward}, Time: {train_time:.1f}s")
    print("Job Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent with evolved network")
    # parser.add_argument("-network", type=str, required=True,
    #                     help='Module path like "models.network_XXXX"')
    parser.add_argument("-timesteps", type=int, default=500000,
                        help="Total training timesteps")
    args = parser.parse_args()

    # Extract gene_id from module path: "models.network_XXXX" -> "XXXX"
    # gene_id = args.network.replace("models.network_", "")

    main("", timesteps=500)

