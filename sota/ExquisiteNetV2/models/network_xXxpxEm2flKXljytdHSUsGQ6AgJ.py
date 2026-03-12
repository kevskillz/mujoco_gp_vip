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

# Define the architecture of the policy
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
        learning_rate=3e-4,  # Decreased learning rate
        gamma=0.99,  # Increased gamma
        gae_lambda=0.95,  # Decreased gae_lambda
        clip_range=0.2,  # Increased clip_range
        ent_coef=0.005,  # Decreased ent_coef
        vf_coef=0.5,  # Increased vf_coef
        max_grad_norm=0.5,  # Decreased max_grad_norm
        n_steps=2048,  # Decreased n_steps
        batch_size=64,  # Decreased batch_size
        n_epochs=10,  # Decreased n_epochs
        normalize_advantage=True,
        verbose=0,
    )
