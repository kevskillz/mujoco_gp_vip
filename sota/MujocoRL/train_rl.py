import os
import sys
import json
import argparse
import importlib
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from eval import evaluate_model


def main(
    gene_id,
    timesteps=500000,
    eval_episodes=None,
    eval_max_steps=None,
    model_dir="sota/MujocoRL/trained_models",
    stats_dir="sota/MujocoRL/stats",
):

    module = importlib.import_module(f"models.network_{gene_id}")

    policy_kwargs = module.get_policy_kwargs()
    ppo_kwargs = module.get_ppo_kwargs()

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
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{gene_id}.zip")
    model.save(model_path)


    # Evaluate using eval.py
    if eval_episodes is None:
        num_eval_episodes = 1 if timesteps <= 10000 else 10
    else:
        num_eval_episodes = eval_episodes
    if eval_max_steps is None:
        max_eval_steps = 200 if timesteps <= 10000 else 1000
    else:
        max_eval_steps = eval_max_steps

    mean_reward, std_reward, rewards, metrics = evaluate_model(
        model, env, num_episodes=num_eval_episodes, max_steps=max_eval_steps
    )
    mean_distance = float(metrics.get("mean_distance", 0.0))
    mean_control_cost = float(metrics.get("mean_control_cost", 0.0))
    train_time = time.time() - start_time

    # Save results under the SOTA_ROOT/results directory (where run_improved.py expects them)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{gene_id}_results.txt"), "w") as f:
        # Backward-compatible prefix remains mean_reward,std_reward,train_time.
        f.write(f"{mean_reward},{std_reward},{train_time},{mean_distance},{mean_control_cost}")

    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{gene_id}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gene_id": gene_id,
                "timesteps": timesteps,
                "num_eval_episodes": num_eval_episodes,
                "max_eval_steps": max_eval_steps,
                "train_time_sec": train_time,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_distance": mean_distance,
                "mean_control_cost": mean_control_cost,
                "model_path": model_path,
                "rewards": rewards,
                "distances": metrics.get("distances", []),
                "control_costs": metrics.get("control_costs", []),
            },
            f,
            indent=2,
        )

    print(
        "Mean reward: "
        f"{mean_reward}, Std: {std_reward}, Distance: {mean_distance}, "
        f"CtrlCost: {mean_control_cost}, Time: {train_time:.1f}s"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved stats: {stats_path}")
    print("Job Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent with evolved network")
    parser.add_argument("-network", type=str, required=True,
                        help='Module path like "models.network_XXXX"')
    parser.add_argument("-timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("-eval_episodes", type=int, default=None,
                        help="Optional number of evaluation episodes")
    parser.add_argument("-eval_max_steps", type=int, default=None,
                        help="Optional max steps per evaluation episode")
    parser.add_argument("-model_dir", type=str, default="sota/MujocoRL/trained_models",
                        help="Directory to save trained model zip")
    parser.add_argument("-stats_dir", type=str, default="sota/MujocoRL/stats",
                        help="Directory to save evaluation stats json")
    args = parser.parse_args()

    # Extract gene_id from module path: "models.network_XXXX" -> "XXXX"
    gene_id = args.network.replace("models.network_", "")

    main(
        gene_id,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        model_dir=args.model_dir,
        stats_dir=args.stats_dir,
    )
