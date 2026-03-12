
# ========== Start: GeneCrossed

# ========== End:

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Define the architecture of the policy
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
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Define the architecture of the policy
HIDDEN_PI = [256, 256]  # Increased hidden layer size
HIDDEN_VF = [256, 256]  # Increased hidden layer size
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
        "ent_coef": 0.02,  # Modified entropy coefficient
        "learning_rate": 0.0005,  # Modified learning rate
        "batch_size": 256,  # Modified batch size
        "gamma": 0.995,  # Modified discount factor
        "gae_lambda": 0.98,  # Modified GAE lambda
        "clip_range": 0.25,  # Modified clip range
        "max_grad_norm": 0.5,  # Added max grad norm
        "vf_coef": 0.5,  # Added value function coefficient
        "use_sde": True,  # Added state-dependent exploration
        "sde_sample_freq": 4,  # Added SDE sample frequency
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
