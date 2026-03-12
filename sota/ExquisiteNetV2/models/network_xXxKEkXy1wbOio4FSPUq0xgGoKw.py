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
    verbose: int = 0,
):
    """
    Returns a dictionary of hyperparameters for the PPO algorithm.

    Args:
    - learning_rate (float): The learning rate for the optimizer.
    - gamma (float): The discount factor for the reward.
    - gae_lambda (float): The lambda parameter for the Generalized Advantage Estimator (GAE).
    - clip_range (float): The clipping range for the policy.
    - ent_coef (float): The entropy coefficient for the policy.
    - vf_coef (float): The value function coefficient.
    - max_grad_norm (float): The maximum gradient norm for the optimizer.
    - n_steps (int): The number of steps to collect before updating the policy.
    - batch_size (int): The batch size for the policy update.
    - n_epochs (int): The number of epochs to update the policy.
    - normalize_advantage (bool): Whether to normalize the advantage.
    - verbose (int): The verbosity level.

    Returns:
    - dict: A dictionary of hyperparameters for the PPO algorithm.
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
