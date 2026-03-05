import gymnasium as gym
import highway_env

env = gym.make("merge-v0", render_mode="human")
obs, info = env.reset()

for episode in range(5):
    obs, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = 1  # IDLE - let IDM drive
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Episode {episode+1}: crashed={env.unwrapped.vehicle.crashed}, speed={env.unwrapped.vehicle.speed:.1f}")

env.close()
