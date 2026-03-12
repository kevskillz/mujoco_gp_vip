# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

HIDDEN_PI = [64, 64]
HIDDEN_VF = [64, 64]
ACTIVATION = nn.Tanh   # Tanh is strong for MuJoCo


class GenePolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            activation_fn=ACTIVATION,
            net_arch=dict(
                pi=HIDDEN_PI,
                vf=HIDDEN_VF
            ),
        )


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy
    }


def get_ppo_kwargs():
    return {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    }
# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

import json
import os

def get_ppo_kwargs(config_file=None):
    """
    Returns a dictionary of PPO hyperparameters with reduced parameters.
    
    Args:
    config_file (str): Path to a JSON configuration file containing PPO hyperparameters.
    
    Returns:
    dict: A dictionary of PPO hyperparameters.
    """
    # Default PPO hyperparameters
    default_kwargs = dict(
        learning_rate=1e-3,  # Increased learning rate for faster convergence
        gamma=0.98,  # Decreased discount factor for more emphasis on immediate rewards
        gae_lambda=0.97,  # Increased generalized advantage estimator lambda for more accurate advantage estimation
        clip_range=0.2,  # Increased clip range for more aggressive training
        ent_coef=0.05,  # Increased entropy coefficient for more exploration
        vf_coef=0.1,  # Decreased value function coefficient for less emphasis on value function
        max_grad_norm=0.7,  # Increased gradient norm clipping for more stable training
        n_steps=2048,  # Increased number of steps for more frequent updates
        batch_size=64,  # Increased batch size for more efficient training
        n_epochs=10,  # Increased number of epochs for more thorough training
        normalize_advantage=True,  # Normalize advantage for more stable training
        verbose=0,  # Disable verbose output for faster training
    )
    
    # Load PPO hyperparameters from a configuration file if provided
    if config_file is not None and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_kwargs = json.load(f)
            # Override default hyperparameters with values from the configuration file
            default_kwargs.update(config_kwargs)
    
    return default_kwargs

def tune_ppo_hyperparameters(kwargs, num_trials=10):
    """
    Performs hyperparameter tuning using a random search algorithm.
    
    Args:
    kwargs (dict): A dictionary of PPO hyperparameters.
    num_trials (int): The number of random search trials.
    
    Returns:
    dict: The best-performing PPO hyperparameters.
    """
    import random
    best_kwargs = kwargs.copy()
    best_performance = float('-inf')
    
    for _ in range(num_trials):
        trial_kwargs = kwargs.copy()
        # Randomly perturb hyperparameters for the current trial
        trial_kwargs['learning_rate'] = random.uniform(1e-4, 1e-2)
        trial_kwargs['gamma'] = random.uniform(0.9, 0.99)
        trial_kwargs['gae_lambda'] = random.uniform(0.9, 0.99)
        trial_kwargs['clip_range'] = random.uniform(0.1, 0.3)
        trial_kwargs['ent_coef'] = random.uniform(0.01, 0.1)
        trial_kwargs['vf_coef'] = random.uniform(0.01, 0.1)
        trial_kwargs['max_grad_norm'] = random.uniform(0.5, 1.0)
        trial_kwargs['n_steps'] = random.randint(1024, 4096)
        trial_kwargs['batch_size'] = random.randint(32, 128)
        trial_kwargs['n_epochs'] = random.randint(5, 20)
        
        # Evaluate the performance of the current trial
        performance = evaluate_ppo_performance(trial_kwargs)
        
        # Update the best-performing hyperparameters if necessary
        if performance > best_performance:
            best_kwargs = trial_kwargs
            best_performance = performance
    
    return best_kwargs

def evaluate_ppo_performance(kwargs):
    """
    Evaluates the performance of the PPO algorithm using the provided hyperparameters.
    
    Args:
    kwargs (dict): A dictionary of PPO hyperparameters.
    
    Returns:
    float: The performance of the PPO algorithm.
    """
    # Implement a performance evaluation metric (e.g., cumulative reward, episode length)
    # For demonstration purposes, return a random performance value
    import random
    return random.uniform(0.0, 1.0)

# Example usage
if __name__ == '__main__':
    kwargs = get_ppo_kwargs()
    print("Default PPO Hyperparameters:")
    print(kwargs)
    
    tuned_kwargs = tune_ppo_hyperparameters(kwargs)
    print("Tuned PPO Hyperparameters:")
    print(tuned_kwargs)
