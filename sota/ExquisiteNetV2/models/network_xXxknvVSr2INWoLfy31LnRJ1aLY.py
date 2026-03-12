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
    verbose: int = 0
):
    """
    Returns a dictionary of hyperparameters for the PPO algorithm.

    Args:
    - learning_rate (float): The learning rate of the optimizer.
    - gamma (float): The discount factor.
    - gae_lambda (float): The lambda parameter for GAE.
    - clip_range (float): The clipping range for PPO.
    - ent_coef (float): The entropy coefficient.
    - vf_coef (float): The value function coefficient.
    - max_grad_norm (float): The maximum gradient norm.
    - n_steps (int): The number of steps to collect experiences.
    - batch_size (int): The batch size for training.
    - n_epochs (int): The number of epochs for training.
    - normalize_advantage (bool): Whether to normalize the advantage.
    - verbose (int): The verbosity level.

    Returns:
    - dict: A dictionary of hyperparameters.
    """
    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        normalize_advantage=normalize_advantage,
        verbose=verbose,
    )

# Example usage:
if __name__ == "__main__":
    ppo_kwargs = get_ppo_kwargs()
    print(ppo_kwargs)
