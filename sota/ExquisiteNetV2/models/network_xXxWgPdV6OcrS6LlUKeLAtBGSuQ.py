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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

HIDDEN_PI = [128, 128]  # Increased hidden layer size for better performance
HIDDEN_VF = [128, 128]  # Increased hidden layer size for better performance
ACTIVATION = nn.ReLU  # ReLU activation function for better performance


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


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self.fc1 = nn.Linear(observation_space.shape[0], 128)  # Added custom feature extractor
        self.fc2 = nn.Linear(128, features_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy,
        "features_extractor_class": CustomFeaturesExtractor  # Added custom feature extractor
    }


def get_ppo_kwargs():
    return {
        "n_steps": 2048,  # Increased number of steps for better performance
        "ent_coef": 0.01,  # Increased entropy coefficient for better exploration
        "learning_rate": 0.0003,  # Decreased learning rate for better convergence
        "batch_size": 128,  # Increased batch size for better performance
        "gamma": 0.99,  # Increased gamma for better long-term rewards
        "gae_lambda": 0.95,  # Increased GAE lambda for better performance
        "clip_range": 0.2,  # Decreased clip range for better stability
        "max_grad_norm": 0.5,  # Decreased max grad norm for better stability
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
