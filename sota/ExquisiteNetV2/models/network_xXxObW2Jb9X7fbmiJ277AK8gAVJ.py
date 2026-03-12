
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
        learning_rate=3e-4,  # Learning rate for the optimizer
        gamma=0.99,  # Discount factor for rewards
        gae_lambda=0.95,  # Lambda parameter for GAE
        clip_range=0.2,  # Clipping range for PPO
        ent_coef=0.005,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Maximum gradient norm
        n_steps=2048,  # Number of steps to collect experiences
        batch_size=64,  # Batch size for training
        n_epochs=10,  # Number of epochs for training
        normalize_advantage=True,  # Whether to normalize advantages
        verbose=0,  # Verbosity level
    )

def get_enhanced_ppo_kwargs():
    """
    Returns a dictionary of enhanced hyperparameters for the PPO algorithm.
    
    Returns:
        dict: A dictionary of hyperparameters.
    """
    return dict(
        learning_rate=3e-4,  # Learning rate for the optimizer
        gamma=0.99,  # Discount factor for rewards
        gae_lambda=0.95,  # Lambda parameter for GAE
        clip_range=0.2,  # Clipping range for PPO
        ent_coef=0.005,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Maximum gradient norm
        n_steps=2048,  # Number of steps to collect experiences
        batch_size=64,  # Batch size for training
        n_epochs=10,  # Number of epochs for training
        normalize_advantage=True,  # Whether to normalize advantages
        verbose=0,  # Verbosity level
    )

def get_custom_ppo_kwargs(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
                          ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, n_steps=2048, 
                          batch_size=64, n_epochs=10, normalize_advantage=True, verbose=0):
    """
    Returns a dictionary of custom hyperparameters for the PPO algorithm.
    
    Returns:
        dict: A dictionary of hyperparameters.
    """
    return dict(
        learning_rate=learning_rate,  
        gamma=gamma,  
        gae_lambda=gae_lambda,  
        clip_range=clip_range,  
        ent_coef=ent_coef,  
        vf_coef=vf_coef,  
        max_grad_norm=max_grad_norm,  
        n_steps=n_steps,  
        batch_size=batch_size,  
        n_epochs=n_epochs,  
        normalize_advantage=normalize_advantage,  
        verbose=verbose,  
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
