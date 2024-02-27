from tm_gym_interface import TMGymInterface
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import numpy as np
from pathlib import Path
from get_game_state import TMInterfaceManager
from stable_baselines3 import DQN

base_dir = Path(__file__).resolve().parents[1]
map_name = "train.npy"
# map_name = "benchmark.npy"

zone_centers = np.load(str(base_dir / "maps" / map_name))

for i in range(3):
    zone_centers = np.vstack(
        (
            zone_centers,
            (2 * zone_centers[-1] - zone_centers[-2])[None, :],
        )
    )

tmi = TMInterfaceManager(base_dir=base_dir, zone_centers=zone_centers)

env = TMGymInterface(tmi)
model = DQN.load(f"final_models/Optimized/model", env=env)

obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)