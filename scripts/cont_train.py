from tm_gym_interface import TMGymInterface
import gymnasium as gym
import numpy as np
from pathlib import Path
from get_game_state import TMInterfaceManager
import time
from stable_baselines3 import DQN

base_dir = Path(__file__).resolve().parents[1]
run_name = "5"
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
custom_objects = {
                    'learning_rate': 6.3e-4, # to decimal 0.00063
                    # 'learning_rate': 0.001,
                    'gamma': 0.99,
                    'batch_size': 128, #128
                    'target_update_interval': 10000,
                    'train_freq': 4,
                    'gradient_steps': 1,
                    'exploration_fraction': 0.12,
                    'exploration_initial_eps': 0.1,
                    'exploration_final_eps': 0.05,
                    'policy_kwargs': dict(net_arch=[256, 256]),
                    'learning_starts': 0,
                    'tensorboard_log': f"runs/{run_name}",
                }
default_objects = {
                    'learning_rate': 0.0001,
                    'gamma': 0.99,
                    'batch_size': 32,
                    'target_update_interval': 10000,
                    'train_freq': 4,
                    'gradient_steps': 1,
                    'exploration_fraction': 0.1,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.05,
                    'policy_kwargs': dict(net_arch=[256, 256]),
                    'learning_starts': 0,
                    'tensorboard_log': f"runs/{run_name}",
                
}

TIMESTEPS = 100000
#tensorboard --logdir runs/
iters = 5
while True:
    model = DQN.load(f"models/{run_name}/{iters}_{TIMESTEPS}", env=env, 
                     custom_objects=custom_objects)
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'DQN_{run_name}_{iters}')
    model.save(f"models/{run_name}/{iters}_{TIMESTEPS}")