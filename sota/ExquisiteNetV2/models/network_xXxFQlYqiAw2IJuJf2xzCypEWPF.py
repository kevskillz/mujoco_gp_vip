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

import optuna

def get_ppo_kwargs(trial):
    return dict(
        learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        gamma=trial.suggest_uniform('gamma', 0.9, 0.99),
        gae_lambda=trial.suggest_uniform('gae_lambda', 0.8, 0.99),
        clip_range=trial.suggest_uniform('clip_range', 0.1, 0.3),
        ent_coef=trial.suggest_uniform('ent_coef', 0.0, 0.1),
        vf_coef=trial.suggest_uniform('vf_coef', 0.4, 0.6),
        max_grad_norm=trial.suggest_uniform('max_grad_norm', 0.4, 0.6),
        n_steps=trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
        n_epochs=trial.suggest_categorical('n_epochs', [5, 10, 15]),
        normalize_advantage=trial.suggest_categorical('normalize_advantage', [True, False]),
        verbose=0,
    )

def optimize_ppo_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(get_ppo_kwargs, n_trials=50)
    best_params = study.best_params
    return best_params

best_params = optimize_ppo_hyperparameters()
print(best_params)
