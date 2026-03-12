
# ========== Start: GeneCrossed

# ========== End:

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import ActorCriticPolicy

# ===============================
# === Architecture Gene Space ===
# ===============================

HIDDEN_PI = [256, 256, 128]  # Increased hidden layer size for pi
HIDDEN_VF = [256, 256, 128]  # Increased hidden layer size for vf
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
        "n_steps": 4096,  # Increased number of steps
        "ent_coef": 0.005,  # Adjusted entropy coefficient
        "learning_rate": 0.00025,  # Adjusted learning rate
        "batch_size": 256,  # Increased batch size
        "gamma": 0.995,  # Increased gamma value
        "gae_lambda": 0.98,  # Increased GAE lambda value
        "clip_range": 0.15,  # Adjusted clip range
    }
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
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

HIDDEN_PI = [128, 128]  # Increased hidden layer size
HIDDEN_VF = [128, 128]  # Increased hidden layer size
ACTIVATION = nn.ReLU  # Changed activation function to ReLU


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
        "n_steps": 2048,  # Increased number of steps
        "ent_coef": 0.01,  # Modified entropy coefficient
        "learning_rate": 0.0003,  # Modified learning rate
        "batch_size": 128,  # Modified batch size
        "gamma": 0.99,  # Modified discount factor
        "gae_lambda": 0.95,  # Modified GAE lambda
        "clip_range": 0.2,  # Modified clip range
    }
# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    return dict(
        learning_rate=5e-4,  # Increased learning rate
        gamma=0.98,  # Decreased gamma
        gae_lambda=0.98,  # Increased gae_lambda
        clip_range=0.1,  # Decreased clip_range
        ent_coef=0.01,  # Increased ent_coef
        vf_coef=0.2,  # Decreased vf_coef
        max_grad_norm=1.0,  # Increased max_grad_norm
        n_steps=4096,  # Increased n_steps
        batch_size=128,  # Increased batch_size
        n_epochs=20,  # Increased n_epochs
        normalize_advantage=True,
        verbose=0,
    )
