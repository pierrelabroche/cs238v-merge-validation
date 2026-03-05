from stable_baselines3 import DQN
import gymnasium as gym
import highway_env

env = gym.make("merge-v0")
model = DQN("MlpPolicy", env, verbose=1)
print("DQN loaded successfully")
