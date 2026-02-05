import os
import time
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

MODEL_PATH = "models/halfcheetah_forward_ppo.zip"
VECNORM_PATH = "models/vecnorm_forward.pkl"

print("CWD:", os.getcwd())
print("MODEL exists?", os.path.exists(MODEL_PATH), MODEL_PATH)
print("VECNORM exists?", os.path.exists(VECNORM_PATH), VECNORM_PATH)

env = gym.make("HalfCheetah-v4", render_mode="human")
venv = DummyVecEnv([lambda: env])

venv = VecNormalize.load(VECNORM_PATH, venv)
venv.training = False
venv.norm_reward = False

model = PPO.load(MODEL_PATH, env=venv)

obs = venv.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    # time.sleep(1/60)
    if done:
        obs = venv.reset()

venv.close()
