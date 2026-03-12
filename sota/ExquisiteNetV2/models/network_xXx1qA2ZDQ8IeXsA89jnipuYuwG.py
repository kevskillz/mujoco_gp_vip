# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.

# ===============================
# === Architecture Gene Space ===
# ===============================

import numpy as np
import torch as th
import torch.nn as nn

HIDDEN_PI = [512, 512, 256]  # Increased hidden layer size for pi
HIDDEN_VF = [512, 512, 256]  # Increased hidden layer size for vf
ACTIVATION = nn.LeakyReLU  # Changed activation function to LeakyReLU


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
        "n_steps": 8192,  # Increased number of steps
        "ent_coef": 0.01,  # Adjusted entropy coefficient
        "learning_rate": 0.0005,  # Adjusted learning rate
        "batch_size": 512,  # Increased batch size
        "gamma": 0.99,  # Increased gamma value
        "gae_lambda": 0.99,  # Increased GAE lambda value
        "clip_range": 0.2,  # Adjusted clip range
        "n_epochs": 10,  # Added number of epochs
        "target_kl": 0.02,  # Added target KL divergence
    }
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
