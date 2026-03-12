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

HIDDEN_PI = [256, 256]  # Increased hidden layer size for pi network
HIDDEN_VF = [256, 256]  # Increased hidden layer size for value function network
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

        # Introduce a unique twist: add a custom layer with a sinusoidal activation function
        self.custom_layer = nn.Sequential(
            nn.Linear(256, 128),  # Input size: 256, Output size: 128
            nn.Sigmoid(),  # Sigmoid activation function
            nn.Linear(128, 256)  # Input size: 128, Output size: 256
        )

    def forward(self, obs, deterministic=False):
        # Call the original forward method
        outputs = super().forward(obs, deterministic)

        # Apply the custom layer to the policy output
        outputs[0] = self.custom_layer(outputs[0])

        return outputs


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy
    }


def get_ppo_kwargs():
    return {
        "n_steps": 4096,  # Increased number of steps for PPO
        "ent_coef": 0.005,  # Decreased entropy coefficient
        "learning_rate": 0.00025,  # Decreased learning rate
        "batch_size": 256,  # Increased batch size
        "gamma": 0.995,  # Increased gamma value
        "gae_lambda": 0.98,  # Increased GAE lambda value
        "clip_range": 0.15,  # Decreased clip range
        "clip_range_vf": None,  # Disabled clip range for value function
    }
# --OPTION--

# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    return dict(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        normalize_advantage=True,
        verbose=0,
    )
