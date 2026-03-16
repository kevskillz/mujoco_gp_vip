
# ========== Start: GeneCrossed

# ========== End:

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Define constants for hidden layers and activation function
HIDDEN_PI = [64, 64]  # Hidden layers for policy network
HIDDEN_VF = [64, 64]  # Hidden layers for value function network
ACTIVATION = nn.Tanh  # Activation function used in the networks

class GenePolicy(ActorCriticPolicy):
    """
    Custom Policy class for Actor-Critic algorithms.
    
    Args:
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super(GenePolicy, self).__init__(
            *args,
            **kwargs,
            activation_fn=ACTIVATION,  # Use Tanh as activation function
            net_arch=dict(
                pi=HIDDEN_PI,  # Policy network architecture
                vf=HIDDEN_VF   # Value function network architecture
            ),
        )

def get_policy_kwargs():
    """
    Returns a dictionary containing the policy class.
    
    Returns:
    dict: Dictionary with "policy_class" key set to GenePolicy.
    """
    return {
        "policy_class": GenePolicy
    }

def get_ppo_kwargs():
    """
    Returns a dictionary of valid PPO hyperparameters.
    
    Returns:
    dict: Dictionary containing PPO hyperparameters.
    """
    # Example hyperparameters, adjust according to your needs
    return {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    }

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
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Define constants for hidden layers and activation function
HIDDEN_PI = [128, 128]  # Increased size for potentially better feature extraction
HIDDEN_VF = [128, 128]  # Increased size for potentially better value estimation
ACTIVATION = nn.ReLU  # Changed to ReLU for potentially better gradient flow

class GenePolicy(ActorCriticPolicy):
    """
    Custom Policy class for Actor-Critic algorithms.
    
    Args:
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super(GenePolicy, self).__init__(
            *args,
            **kwargs,
            activation_fn=ACTIVATION,  # Use ReLU as activation function
            net_arch=dict(
                pi=HIDDEN_PI,  # Policy network architecture
                vf=HIDDEN_VF   # Value function network architecture
            ),
        )

def get_policy_kwargs():
    """
    Returns a dictionary containing the policy class.
    
    Returns:
    dict: Dictionary with "policy_class" key set to GenePolicy.
    """
    return {
        "policy_class": GenePolicy
    }

def get_ppo_kwargs():
    """
    Returns a dictionary of valid PPO hyperparameters.
    
    Returns:
    dict: Dictionary containing PPO hyperparameters.
    """
    # Adjusted hyperparameters for potentially improved performance
    return {
        "n_steps": 4096,  # Increased for more exploration
        "ent_coef": 0.02,  # Slightly increased for better exploration
        "learning_rate": 0.0005,  # Adjusted for potentially faster convergence
        "batch_size": 256,  # Doubled for potentially faster training
        "gamma": 0.995,  # Slightly decreased for less discounting
        "gae_lambda": 0.92,  # Adjusted for potentially better advantage estimation
        "clip_range": 0.15,  # Decreased for potentially better stability
    }

# --OPTION--
import numpy as np

# ===============================
# === Helper Functions ===
# ===============================
def linear_learning_rate_schedule(initial_lr, decay_rate):
    """Linear learning rate schedule."""
    def schedule(x):
        return initial_lr * (1 - x * decay_rate)
    return schedule

def exponential_learning_rate_schedule(initial_lr, decay_rate):
    """Exponential learning rate schedule."""
    def schedule(x):
        return initial_lr * np.exp(-x * decay_rate)
    return schedule

def cosine_annealing_schedule(initial_lr, min_lr, T_max):
    """Cosine annealing learning rate schedule."""
    def schedule(x):
        return min_lr + (initial_lr - min_lr) / 2 * (1 + np.cos(np.pi * x / T_max))
    return schedule

# ===============================
# === PPO Hyperparameter Gene ===
# ===============================
def get_ppo_kwargs(
    learning_rate_schedule=None, 
    entropy_regularization=None, 
    initial_learning_rate=3e-4, 
    decay_rate=0.1, 
    min_learning_rate=1e-6, 
    T_max=1000
):
    """
    Returns a dictionary of PPO hyperparameters.

    Args:
    - learning_rate_schedule (str): The type of learning rate schedule to use (e.g., 'linear', 'exponential', 'cosine_annealing').
    - entropy_regularization (float): The entropy regularization coefficient.
    - initial_learning_rate (float): The initial learning rate.
    - decay_rate (float): The decay rate for the learning rate schedule.
    - min_learning_rate (float): The minimum learning rate for the cosine annealing schedule.
    - T_max (int): The maximum number of iterations for the cosine annealing schedule.

    Returns:
    - A dictionary of PPO hyperparameters.
    """

    # Default hyperparameters
    ppo_kwargs = dict(
        learning_rate=initial_learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0 if entropy_regularization is None else entropy_regularization,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        normalize_advantage=True,
        verbose=0,
    )

    # Apply learning rate scheduling if specified
    if learning_rate_schedule == 'linear':
        ppo_kwargs['learning_rate'] = linear_learning_rate_schedule(initial_learning_rate, decay_rate)
    elif learning_rate_schedule == 'exponential':
        ppo_kwargs['learning_rate'] = exponential_learning_rate_schedule(initial_learning_rate, decay_rate)
    elif learning_rate_schedule == 'cosine_annealing':
        ppo_kwargs['learning_rate'] = cosine_annealing_schedule(initial_learning_rate, min_learning_rate, T_max)

    return ppo_kwargs

