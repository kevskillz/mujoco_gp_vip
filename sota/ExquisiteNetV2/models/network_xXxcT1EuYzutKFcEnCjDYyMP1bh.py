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

def calculate_learning_rate(base_lr, decay_rate, epoch):
    """
    Calculate the learning rate based on the current epoch.

    Args:
    - base_lr (float): The base learning rate.
    - decay_rate (float): The rate at which the learning rate decays.
    - epoch (int): The current epoch.

    Returns:
    - float: The calculated learning rate.
    """
    return base_lr * (1 - decay_rate * epoch)

def calculate_clip_range(base_clip, clip_decay, epoch):
    """
    Calculate the clip range based on the current epoch.

    Args:
    - base_clip (float): The base clip range.
    - clip_decay (float): The rate at which the clip range decays.
    - epoch (int): The current epoch.

    Returns:
    - float: The calculated clip range.
    """
    return base_clip * (1 - clip_decay * epoch)

def get_ppo_kwargs(epoch=0, base_lr=3e-4, decay_rate=0.01, base_clip=0.2, clip_decay=0.01):
    """
    Get the PPO hyperparameters.

    Args:
    - epoch (int, optional): The current epoch. Defaults to 0.
    - base_lr (float, optional): The base learning rate. Defaults to 3e-4.
    - decay_rate (float, optional): The rate at which the learning rate decays. Defaults to 0.01.
    - base_clip (float, optional): The base clip range. Defaults to 0.2.
    - clip_decay (float, optional): The rate at which the clip range decays. Defaults to 0.01.

    Returns:
    - dict: The PPO hyperparameters.
    """
    learning_rate = calculate_learning_rate(base_lr, decay_rate, epoch)
    clip_range = calculate_clip_range(base_clip, clip_decay, epoch)
    return dict(
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        normalize_advantage=True,
        verbose=0,
    )
