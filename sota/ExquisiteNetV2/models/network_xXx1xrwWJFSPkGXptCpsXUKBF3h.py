
# ========== Start: GeneCrossed

# ========== End:
# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    """
    Returns a dictionary of hyperparameters for the PPO algorithm.
    
    Returns:
        dict: A dictionary of hyperparameters.
    """
    return dict(
        learning_rate=4e-4,  # Moderate learning rate
        gamma=0.99,  # High discount factor
        gae_lambda=0.98,  # High lambda parameter
        clip_range=0.15,  # Moderate clipping range
        ent_coef=0.01,  # Small entropy coefficient
        vf_coef=0.35,  # Moderate value function coefficient
        max_grad_norm=0.75,  # Moderate maximum gradient norm
        n_steps=4096,  # Large number of steps
        batch_size=128,  # Large batch size
        n_epochs=20,  # Large number of epochs
        normalize_advantage=True,  # Normalize advantages
        verbose=0,  # Low verbosity
    )
# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    """
    Returns a dictionary of hyperparameters for the PPO algorithm.
    
    Returns:
        dict: A dictionary of hyperparameters.
    """
    return dict(
        learning_rate=3e-4,  # Learning rate for the optimizer
        gamma=0.99,  # Discount factor for rewards
        gae_lambda=0.95,  # Lambda parameter for GAE
        clip_range=0.2,  # Clipping range for PPO
        ent_coef=0.0,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Maximum gradient norm
        n_steps=2048,  # Number of steps to collect experiences
        batch_size=64,  # Batch size for training
        n_epochs=10,  # Number of epochs for training
        normalize_advantage=True,  # Whether to normalize advantages
        verbose=0,  # Verbosity level
    )
