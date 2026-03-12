# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
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
ACTIVATION = nn.ReLU  # ReLU activation function for better non-linearity

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the GenePolicy model.
    """
    def __init__(self, observation_space, features_dim):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.fc1 = nn.Linear(observation_space.shape[0], 128)  # Input layer
        self.fc2 = nn.Linear(128, features_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for the input layer
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
            features_extractor_class=CustomFeaturesExtractor,  # Custom feature extractor
        )


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy
    } 


def get_ppo_kwargs():
    """
    Returns a dictionary of valid PPO hyperparameters.
    """
    return {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
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
