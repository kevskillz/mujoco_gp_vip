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

        # Introduce a new layer with a different activation function
        self.additional_layer = nn.Sequential(
            nn.Linear(HIDDEN_PI[-1], HIDDEN_PI[-1]),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        # Use the additional layer to process the input
        features = self.mlp_extractor(obs)
        latent_pi, latent_vf = features
        latent_pi = self.additional_layer(latent_pi)

        # Rest of the forward pass remains the same
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return mean_actions, self.log_std
        log_std = self.log_std
        return mean_actions, log_std

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, deterministic: bool = False) -> tuple:
        # Use the additional layer to process the latent space
        latent_pi = self.additional_layer(latent_pi)
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return mean_actions, self.log_std
        log_std = self.log_std
        return mean_actions, log_std


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
        "clip_range_vf": None,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "verbose": 1,
        "seed": 0,
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
