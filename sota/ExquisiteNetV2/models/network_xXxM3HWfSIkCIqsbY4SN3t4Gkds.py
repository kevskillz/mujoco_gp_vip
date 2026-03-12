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
        learning_rate=3.5e-4,  # Increased learning rate for faster convergence
        gamma=0.995,  # Increased discount factor for longer-term rewards
        gae_lambda=0.92,  # Decreased lambda for more conservative advantage estimation
        clip_range=0.22,  # Increased clip range for more aggressive policy updates
        ent_coef=0.01,  # Introduced entropy coefficient for exploration-exploitation trade-off
        vf_coef=0.6,  # Increased value function coefficient for more accurate value estimation
        max_grad_norm=0.6,  # Increased maximum gradient norm for more stable updates
        n_steps=4096,  # Increased number of steps for more diverse experience collection
        batch_size=128,  # Increased batch size for more stable policy updates
        n_epochs=15,  # Increased number of epochs for more thorough policy optimization
        normalize_advantage=True,  # Retained advantage normalization for more stable updates
        verbose=1,  # Increased verbosity for more detailed logging
    )
