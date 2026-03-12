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

def get_ppo_kwargs():
    """
    Returns a dictionary of PPO hyperparameters with reduced parameters.
    """
    return dict(
        learning_rate=3e-4,  # Reduced learning rate for more stable training
        gamma=0.99,  # Discount factor for reward calculation
        gae_lambda=0.95,  # Generalized advantage estimator lambda
        clip_range=0.1,  # Reduced clip range for more conservative training
        ent_coef=0.01,  # Small entropy coefficient for exploration
        vf_coef=0.25,  # Reduced value function coefficient for more stable training
        max_grad_norm=0.5,  # Gradient norm clipping for stable training
        n_steps=1024,  # Reduced number of steps for more frequent updates
        batch_size=32,  # Reduced batch size for more efficient training
        n_epochs=5,  # Reduced number of epochs for more efficient training
        normalize_advantage=True,  # Normalize advantage for more stable training
        verbose=0,  # Disable verbose output for faster training
    )
