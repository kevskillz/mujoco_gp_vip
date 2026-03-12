# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--

# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

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


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy
    }

# --OPTION--
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

import math

def get_ppo_kwargs():
    # Use prime numbers for learning rate and gamma
    learning_rate = 7e-4  # 7 is a prime number
    gamma = 0.997  # close to 1, but not exactly 1
    
    # Use irrational numbers for gae_lambda and clip_range
    gae_lambda = math.sqrt(2) / 2  # approximately 0.707
    clip_range = math.pi / 6  # approximately 0.524
    
    # Use a non-standard value for ent_coef
    ent_coef = 1 / math.e  # approximately 0.368
    
    # Use a non-standard value for vf_coef
    vf_coef = math.sqrt(3) / 2  # approximately 0.866
    
    # Use a non-standard value for max_grad_norm
    max_grad_norm = math.log(2)  # approximately 0.693
    
    # Use a non-standard value for n_steps
    n_steps = 2049  # a prime number close to 2048
    
    # Use a non-standard value for batch_size
    batch_size = 65  # a prime number close to 64
    
    # Use a non-standard value for n_epochs
    n_epochs = 11  # a prime number close to 10
    
    # Use a non-standard value for normalize_advantage
    normalize_advantage = False  # opposite of the original value
    
    # Use a non-standard value for verbose
    verbose = 1  # a non-zero value
    
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
