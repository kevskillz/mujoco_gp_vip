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
        # Learning rate schedule
        learning_rate=3e-4,
        learning_rate_schedule='linear',
        learning_rate_final=1e-5,
        
        # Gamma and GAE lambda
        gamma=0.99,
        gae_lambda=0.95,
        
        # Clipping and normalization
        clip_range=0.2,
        normalize_advantage=True,
        
        # Entropy regularization
        ent_coef=0.01,
        ent_coef_schedule='linear',
        ent_coef_final=0.001,
        
        # Value function coefficient
        vf_coef=0.5,
        
        # Gradient normalization
        max_grad_norm=0.5,
        
        # Training parameters
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        
        # Verbosity
        verbose=0,
    )

def get_ppo_kwargs_with_momentum():
    kwargs = get_ppo_kwargs()
    kwargs.update(dict(
        # Momentum for the optimizer
        momentum=0.9,
        
        # Nesterov acceleration
        nesterov=True,
    ))
    return kwargs

def get_ppo_kwargs_with_adamw():
    kwargs = get_ppo_kwargs()
    kwargs.update(dict(
        # AdamW optimizer
        optimizer='adamw',
        
        # Weight decay
        weight_decay=0.01,
    ))
    return kwargs

def get_ppo_kwargs_with_cliprange_schedule():
    kwargs = get_ppo_kwargs()
    kwargs.update(dict(
        # Clip range schedule
        clip_range_schedule='linear',
        clip_range_final=0.1,
    ))
    return kwargs
