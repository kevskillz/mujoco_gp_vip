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
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

import torch
import torch.nn as nn
from stable_baselines3 import ActorCriticPolicy

HIDDEN_PI = [128, 128]  # Increased hidden layer size for pi network
HIDDEN_VF = [128, 128]  # Increased hidden layer size for value function network
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
        "n_steps": 2048,  # Increased number of steps for PPO
        "ent_coef": 0.01,  # Increased entropy coefficient
        "learning_rate": 0.0003,  # Decreased learning rate
        "batch_size": 128,  # Increased batch size
        "gamma": 0.99,  # Increased gamma value
        "gae_lambda": 0.95,  # Increased GAE lambda value
        "clip_range": 0.2,  # Decreased clip range
        "max_grad_norm": 0.5,  # Increased max grad norm
    }
# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    return dict(
        learning_rate=1e-3,  # Increased learning rate
        gamma=0.98,  # Decreased gamma
        gae_lambda=0.98,  # Increased gae lambda
        clip_range=0.1,  # Decreased clip range
        ent_coef=0.01,  # Increased entropy coefficient
        vf_coef=0.2,  # Decreased value function coefficient
        max_grad_norm=1.0,  # Increased max grad norm
        n_steps=4096,  # Increased number of steps
        batch_size=128,  # Increased batch size
        n_epochs=20,  # Increased number of epochs
        normalize_advantage=True,
        verbose=0,
    )
