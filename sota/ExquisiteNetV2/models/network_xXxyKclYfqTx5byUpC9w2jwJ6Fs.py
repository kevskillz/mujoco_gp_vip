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

def get_ppo_kwargs():
    return dict(
        learning_rate=1e-3,  # Increased learning rate
        gamma=0.98,  # Decreased gamma
        gae_lambda=0.98,  # Increased gae lambda
        clip_range=0.1,  # Decreased clip range
        ent_coef=0.01,  # Increased entropy coefficient
        vf_coef=0.2,  # Decreased value function coefficient
        max_grad_norm=1.0,  # Increased max grad norm
        n_steps=4096,  # Increased number of steps
        batch_size=128,  # Increased batch size
        n_epochs=20,  # Increased number of epochs
        normalize_advantage=True,
        verbose=0,
    )
