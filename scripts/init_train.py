from tm_gym_interface import TMGymInterface
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import numpy as np
from pathlib import Path
from get_game_state import TMInterfaceManager
from stable_baselines3 import DQN

base_dir = Path(__file__).resolve().parents[1]
run_name = "0"
map_name = "train.npy"
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
TIMESTEPS = 200000

model = DQN('MultiInputPolicy', env,
            tensorboard_log=f"runs/{run_name}",
            learning_rate=0.00063,
            gamma=0.99,
            batch_size=128,
            target_update_interval=10000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.12,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256, 256]), verbose=1)

model.learn(total_timesteps=TIMESTEPS, tb_log_name=f'DQN_{run_name}_0')
model.save(f"models/{run_name}/0_{TIMESTEPS}")