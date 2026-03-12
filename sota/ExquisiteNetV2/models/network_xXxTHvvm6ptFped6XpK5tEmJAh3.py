# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
# Import necessary libraries
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define hyperparameters
HIDDEN_PI = [256, 256]  # Increased hidden layer size for pi network
HIDDEN_VF = [256, 256]  # Increased hidden layer size for value function network
ACTIVATION = nn.GELU  # Changed activation function to GELU

class GenePolicy(ActorCriticPolicy):
    """
    Custom policy class that inherits from ActorCriticPolicy.
    
    Args:
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.
    """
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
    """
    Returns a dictionary with the policy class.
    
    Returns:
    dict: Dictionary containing the policy class.
    """
    return {
        "policy_class": GenePolicy
    }


def get_ppo_kwargs():
    """
    Returns a dictionary with PPO hyperparameters.
    
    Returns:
    dict: Dictionary containing PPO hyperparameters.
    """
    return {
        "n_steps": 4096,  # Increased number of steps for PPO
        "ent_coef": 0.02,  # Increased entropy coefficient
        "learning_rate": 0.00025,  # Decreased learning rate
        "batch_size": 256,  # Increased batch size
        "gamma": 0.995,  # Increased discount factor
        "gae_lambda": 0.98,  # Increased GAE lambda
        "clip_range": 0.15,  # Decreased clip range
        "max_grad_norm": 0.6,  # Increased max grad norm
    }
# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def calculate_learning_rate(base_lr, decay_rate, epoch):
    """
    Calculate the learning rate based on the current epoch.

    Args:
    - base_lr (float): The base learning rate.
    - decay_rate (float): The rate at which the learning rate decays.
    - epoch (int): The current epoch.

    Returns:
    - float: The calculated learning rate.
    """
    return base_lr * (1 - decay_rate * epoch)

def calculate_clip_range(base_clip, clip_decay, epoch):
    """
    Calculate the clip range based on the current epoch.

    Args:
    - base_clip (float): The base clip range.
    - clip_decay (float): The rate at which the clip range decays.
    - epoch (int): The current epoch.

    Returns:
    - float: The calculated clip range.
    """
    return base_clip * (1 - clip_decay * epoch)

def get_ppo_kwargs(epoch=0, base_lr=3e-4, decay_rate=0.01, base_clip=0.2, clip_decay=0.01):
    """
    Get the PPO hyperparameters.

    Args:
    - epoch (int, optional): The current epoch. Defaults to 0.
    - base_lr (float, optional): The base learning rate. Defaults to 3e-4.
    - decay_rate (float, optional): The rate at which the learning rate decays. Defaults to 0.01.
    - base_clip (float, optional): The base clip range. Defaults to 0.2.
    - clip_decay (float, optional): The rate at which the clip range decays. Defaults to 0.01.

    Returns:
    - dict: The PPO hyperparameters.
    """
    learning_rate = calculate_learning_rate(base_lr, decay_rate, epoch)
    clip_range = calculate_clip_range(base_clip, clip_decay, epoch)
    return dict(
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        normalize_advantage=True,
        verbose=0,
    )
