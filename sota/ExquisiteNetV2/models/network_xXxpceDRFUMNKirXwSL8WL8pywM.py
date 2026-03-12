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
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

HIDDEN_PI = [128, 128]  # Increased hidden layer size for better feature extraction
HIDDEN_VF = [128, 128]  # Increased hidden layer size for better value function estimation
ACTIVATION = nn.ReLU  # ReLU activation function for better gradient flow

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for the GenePolicy.
    """
    def __init__(self, observation_space, features_dim):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.fc1 = nn.Linear(observation_space.shape[0], 128)  # Input layer
        self.fc2 = nn.Linear(128, features_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)
        return x

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
            features_extractor_class=CustomFeaturesExtractor,
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

import numpy as np

def get_default_ppo_kwargs():
    """
    Returns a dictionary of default hyperparameters for the PPO algorithm.
    """
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

def validate_ppo_kwargs(ppo_kwargs):
    """
    Validates the hyperparameters for the PPO algorithm.

    Args:
    - ppo_kwargs (dict): A dictionary of hyperparameters.

    Raises:
    - ValueError: If any hyperparameter is invalid.
    """
    required_keys = [
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "n_steps",
        "batch_size",
        "n_epochs",
        "normalize_advantage",
        "verbose",
    ]

    for key in required_keys:
        if key not in ppo_kwargs:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(ppo_kwargs["learning_rate"], (int, float)) or ppo_kwargs["learning_rate"] <= 0:
        raise ValueError("Learning rate must be a positive number")

    if not isinstance(ppo_kwargs["gamma"], (int, float)) or ppo_kwargs["gamma"] < 0 or ppo_kwargs["gamma"] > 1:
        raise ValueError("Gamma must be a number between 0 and 1")

    if not isinstance(ppo_kwargs["gae_lambda"], (int, float)) or ppo_kwargs["gae_lambda"] < 0 or ppo_kwargs["gae_lambda"] > 1:
        raise ValueError("GAE lambda must be a number between 0 and 1")

    if not isinstance(ppo_kwargs["clip_range"], (int, float)) or ppo_kwargs["clip_range"] < 0:
        raise ValueError("Clip range must be a non-negative number")

    if not isinstance(ppo_kwargs["ent_coef"], (int, float)) or ppo_kwargs["ent_coef"] < 0:
        raise ValueError("Entropy coefficient must be a non-negative number")

    if not isinstance(ppo_kwargs["vf_coef"], (int, float)) or ppo_kwargs["vf_coef"] < 0:
        raise ValueError("Value function coefficient must be a non-negative number")

    if not isinstance(ppo_kwargs["max_grad_norm"], (int, float)) or ppo_kwargs["max_grad_norm"] < 0:
        raise ValueError("Max gradient norm must be a non-negative number")

    if not isinstance(ppo_kwargs["n_steps"], int) or ppo_kwargs["n_steps"] <= 0:
        raise ValueError("Number of steps must be a positive integer")

    if not isinstance(ppo_kwargs["batch_size"], int) or ppo_kwargs["batch_size"] <= 0:
        raise ValueError("Batch size must be a positive integer")

    if not isinstance(ppo_kwargs["n_epochs"], int) or ppo_kwargs["n_epochs"] <= 0:
        raise ValueError("Number of epochs must be a positive integer")

    if not isinstance(ppo_kwargs["normalize_advantage"], bool):
        raise ValueError("Normalize advantage must be a boolean")

    if not isinstance(ppo_kwargs["verbose"], int) or ppo_kwargs["verbose"] < 0:
        raise ValueError("Verbose must be a non-negative integer")

def get_ppo_kwargs(
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    normalize_advantage: bool = True,
