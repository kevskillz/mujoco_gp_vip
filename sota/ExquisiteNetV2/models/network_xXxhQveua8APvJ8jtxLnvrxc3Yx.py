# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--
# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.# -- NOTE --
import torch
import torch.nn as nn
from stable_baselines3 import ActorCriticPolicy

# ===============================
# === Architecture Gene Space ===
# ===============================

HIDDEN_PI = [256, 256]  # Increased hidden layer size
HIDDEN_VF = [256, 256]  # Increased hidden layer size
ACTIVATION = nn.ReLU  # Changed activation function to ReLU

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

def get_ppo_kwargs():
    return {
        "n_steps": 4096,  # Increased number of steps
        "ent_coef": 0.02,  # Modified entropy coefficient
        "learning_rate": 0.00025,  # Modified learning rate
        "batch_size": 256,  # Modified batch size
        "gamma": 0.995,  # Modified discount factor
        "gae_lambda": 0.95,  # Modified GAE lambda
        "clip_range": 0.2,  # Modified clip range
        "n_epochs": 10,  # Added number of epochs
        "target_kl": None,  # Added target KL divergence
    }

def create_neural_network(input_size, output_size):
    """
    Creates a neural network with two hidden layers.
    
    Args:
    input_size (int): The size of the input layer.
    output_size (int): The size of the output layer.
    
    Returns:
    nn.Module: The created neural network.
    """
    return nn.Sequential(
        nn.Linear(input_size, HIDDEN_PI[0]),
        ACTIVATION(),
        nn.Linear(HIDDEN_PI[0], HIDDEN_PI[1]),
        ACTIVATION(),
        nn.Linear(HIDDEN_PI[1], output_size)
    )

def calculate_entropy(policy, actions):
    """
    Calculates the entropy of the policy.
    
    Args:
    policy (GenePolicy): The policy to calculate the entropy for.
    actions (torch.Tensor): The actions to calculate the entropy for.
    
    Returns:
    torch.Tensor: The calculated entropy.
    """
    log_prob = policy.log_prob(actions)
    return -torch.mean(log_prob)

def train_policy(policy, env, num_episodes):
    """
    Trains the policy using PPO.
    
    Args:
    policy (GenePolicy): The policy to train.
    env (gym.Env): The environment to train in.
    num_episodes (int): The number of episodes to train for.
    """
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        rewards = 0.0
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards += reward
        print(f'Episode {episode+1}, Reward: {rewards}')
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
        ent_coef=0.01,  # Fixed to a small value
        vf_coef=0.5,    # Fixed to a value
        max_grad_norm=0.5,  # Fixed to a value
        n_steps=2048,
        batch_size=64,
        n_epochs=4,     # Fixed to a small value
        normalize_advantage=True,  # Fixed to True
        verbose=0,      # Fixed to 0
    )
