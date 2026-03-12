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
from stable_baselines3 import ActorCriticPolicy

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
    """
    Returns a dictionary with the policy class.
    
    Returns:
        dict: A dictionary containing the policy class.
    """
    return {
        "policy_class": GenePolicy
    } 


def get_ppo_kwargs():
    """
    Returns a dictionary of valid PPO hyperparameters.
    
    Returns:
        dict: A dictionary containing valid PPO hyperparameters.
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
        "use_tanh_act": True,
        "use_sde": False,
        "sde_sample_freq": 4,
        "target_kl": None,
        "verbose": 1,
        "seed": None,
        "device": "auto"
    }


def create_policy(*args, **kwargs):
    """
    Creates a new instance of the GenePolicy class.
    
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    
    Returns:
        GenePolicy: A new instance of the GenePolicy class.
    """
    return GenePolicy(*args, **kwargs)


def main():
    # Example usage:
    policy_kwargs = get_policy_kwargs()
    ppo_kwargs = get_ppo_kwargs()
    policy = create_policy(**policy_kwargs)


if __name__ == "__main__":
    main()
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
