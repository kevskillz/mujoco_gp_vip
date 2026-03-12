
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
        learning_rate=1e-3,  # Increased learning rate for faster convergence
        gamma=0.98,  # Slightly decreased discount factor for more emphasis on immediate rewards
        gae_lambda=0.92,  # Decreased lambda parameter for GAE to reduce variance
        clip_range=0.25,  # Increased clipping range for PPO to allow for more exploration
        ent_coef=0.01,  # Introduced entropy coefficient to encourage exploration
        vf_coef=0.6,  # Increased value function coefficient to improve value estimation
        max_grad_norm=0.6,  # Increased maximum gradient norm to allow for larger updates
        n_steps=4096,  # Increased number of steps to collect more experiences
        batch_size=128,  # Increased batch size for more stable training
        n_epochs=15,  # Increased number of epochs for more thorough training
        normalize_advantage=True,  # Retained normalization of advantages
        verbose=1,  # Increased verbosity level for more informative output
    )
