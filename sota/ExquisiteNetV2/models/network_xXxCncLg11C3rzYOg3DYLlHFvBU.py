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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Additional feature: Batch normalization
        self.batch_norm_pi = nn.BatchNorm1d(HIDDEN_PI[0])
        self.batch_norm_vf = nn.BatchNorm1d(HIDDEN_VF[0])

        # Additional feature: Dropout
        self.dropout_pi = nn.Dropout(p=0.2)
        self.dropout_vf = nn.Dropout(p=0.2)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        # Additional feature: Input normalization
        obs = F.normalize(obs, p=2, dim=1)

        # Original forward pass
        mean_actions, log_std, values = super().forward(obs, deterministic)

        # Additional feature: Output normalization
        mean_actions = F.normalize(mean_actions, p=2, dim=1)

        return mean_actions, log_std, values

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, deterministic: bool = False) -> tuple:
        # Additional feature: Batch normalization
        latent_pi = self.batch_norm_pi(latent_pi)

        # Additional feature: Dropout
        latent_pi = self.dropout_pi(latent_pi)

        # Original action distribution
        mean_actions, log_std = super()._get_action_dist_from_latent(latent_pi, deterministic)

        return mean_actions, log_std

    def _get_value(self, latent_vf: torch.Tensor) -> torch.Tensor:
        # Additional feature: Batch normalization
        latent_vf = self.batch_norm_vf(latent_vf)

        # Additional feature: Dropout
        latent_vf = self.dropout_vf(latent_vf)

        # Original value function
        values = super()._get_value(latent_vf)

        return values


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
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
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
