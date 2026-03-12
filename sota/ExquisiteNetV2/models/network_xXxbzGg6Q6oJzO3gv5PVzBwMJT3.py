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
        self.fc1 = nn.Linear(observation_space.shape[0], 64)  # Added custom feature extractor
        self.fc2 = nn.Linear(64, features_dim)

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
        "batch_size": 64,  # Increased batch size for better performance
        "gamma": 0.99,  # Increased gamma for better long-term rewards
        "gae_lambda": 0.95,  # Increased GAE lambda for better performance
        "clip_range": 0.2,  # Decreased clip range for better stability
        "max_grad_norm": 0.5,  # Decreased max grad norm for better stability
    }
# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_default_ppo_kwargs():
    """
    Returns the default PPO hyperparameters.
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

def get_custom_ppo_kwargs(learning_rate, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm, n_steps, batch_size, n_epochs, normalize_advantage, verbose):
    """
    Returns custom PPO hyperparameters.
    
    Args:
    - learning_rate (float): The learning rate of the model.
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

def get_ppo_kwargs(custom=False, **kwargs):
    """
    Returns the PPO hyperparameters.
    
    Args:
    - custom (bool): Whether to use custom hyperparameters. Defaults to False.
    - **kwargs: Custom hyperparameter values.
    """
    if custom:
        return get_custom_ppo_kwargs(**kwargs)
    else:
        return get_default_ppo_kwargs()

def tune_ppo_hyperparameters(learning_rate_range, gamma_range, gae_lambda_range, clip_range_range, ent_coef_range, vf_coef_range, max_grad_norm_range, n_steps_range, batch_size_range, n_epochs_range):
    """
    Tunes the PPO hyperparameters using a grid search.
    
    Args:
    - learning_rate_range (list): A list of learning rates to try.
    - gamma_range (list): A list of discount factors to try.
    - gae_lambda_range (list): A list of lambda parameters to try.
    - clip_range_range (list): A list of clipping ranges to try.
    - ent_coef_range (list): A list of entropy coefficients to try.
    - vf_coef_range (list): A list of value function coefficients to try.
    - max_grad_norm_range (list): A list of maximum gradient norms to try.
    - n_steps_range (list): A list of numbers of steps to try.
    - batch_size_range (list): A list of batch sizes to try.
    - n_epochs_range (list): A list of numbers of epochs to try.
    """
    best_hyperparameters = None
    best_performance = float('-inf')
    
    for learning_rate in learning_rate_range:
        for gamma in gamma_range:
            for gae_lambda in gae_lambda_range:
                for clip_range in clip_range_range:
                    for ent_coef in ent_coef_range:
                        for vf_coef in vf_coef_range:
                            for max_grad_norm in max_grad_norm_range:
                                for n_steps in n_steps_range:
                                    for batch_size in batch_size_range:
                                        for n_epochs in n_epochs_range
