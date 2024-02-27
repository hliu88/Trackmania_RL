import numpy as np
from tminterface.interface import Message, MessageType, TMInterface
import gymnasium as gym
from gymnasium import spaces
import time

inputs = [
    {  # 0 Forward
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 1 Forward left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 2 Forward right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": False,
    },
    {  # 3 Nothing
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 4 Nothing left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 5 Nothing right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": False,
    },
    {  # 6 Brake
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 7 Brake left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 8 Brake right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": True,
    },
    {  # 9 Brake and accelerate
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 10 Brake and accelerate left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 11 Brake and accelerate right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": True,
    },
]

class TMGymInterface(gym.Env):
    def __init__(self, tmi) -> None:
        super(TMGymInterface, self).__init__()
        self.tmi = tmi
        n_actions = 12
        self.action_space = spaces.Discrete(n_actions)
        
        vehicle_velocity_space = spaces.Box(low=-100, high=100, shape=(1,), dtype='float32')
        vehicle_turning_rate_space = spaces.Box(low=-1, high=1, shape=(1,), dtype='float32')
        vehicle_lateral_velocity_space = spaces.Box(low=-2, high=2, shape=(1,), dtype='float32')
        vehicle_acceleration_space = spaces.Box(low=-1, high=1, shape=(1,), dtype='float32')
        angle_to_center_line_space = spaces.Box(low=-1, high=1, shape=(1,), dtype='float32')
        angle_to_next_zone_space = spaces.Box(low=-1, high=1, shape=(1,), dtype='float32')
        angle_to_next_next_zone_space = spaces.Box(low=-1, high=1, shape=(1,), dtype='float32')
        angle_to_next_next_next_zone_space = spaces.Box(low=-1, high=1, shape=(1,), dtype='float32')
        distance_to_center_line_space = spaces.Box(low=-100, high=100, shape=(1,), dtype='float32')
        distance_to_next_checkpt_space = spaces.Box(low=-1000, high=1000, shape=(1,), dtype='float32')
        distance_to_next_next_checkpt_space = spaces.Box(low=-1000, high=1000, shape=(1,), dtype='float32')
        distance_to_next_next_next_checkpt_space = spaces.Box(low=-1000, high=1000, shape=(1,), dtype='float32')
        # contact_ground_material_left_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype='float32')
        # contact_ground_material_right_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype='float32')
        # vehicle_wheel_space = spaces.Box(low=0, high=1, shape=(4,), dtype='float32')
        # vehicle_engine_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype='float32')
        # vehicle_gear_space = spaces.Box(low=0, high=10, shape=(1,), dtype='float32')

        #convert to a dict
        self.observation_space = spaces.Dict({
            'vehicle_velocity': vehicle_velocity_space,
            'vehicle_turning_rate': vehicle_turning_rate_space,
            'vehicle_lateral_velocity': vehicle_lateral_velocity_space,
            'vehicle_acceleration': vehicle_acceleration_space,
            'angle_to_center_line': angle_to_center_line_space,
            'angle_to_next_zone': angle_to_next_zone_space,
            'angle_to_next_next_zone': angle_to_next_next_zone_space,
            'angle_to_next_next_next_zone': angle_to_next_next_next_zone_space,
            'distance_to_center_line': distance_to_center_line_space,
            'distance_to_next_checkpt': distance_to_next_checkpt_space,
            'distance_to_next_next_checkpt': distance_to_next_next_checkpt_space,
            'distance_to_next_next_next_checkpt': distance_to_next_next_next_checkpt_space,
            # 'contact_ground_material_left': contact_ground_material_left_space,
            # 'contact_ground_material_right': contact_ground_material_right_space,
            # 'vehicle_wheel': vehicle_wheel_space,
            # 'vehicle_engine': vehicle_engine_space,
            # 'vehicle_gear': vehicle_gear_space,
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs = self.tmi.rollout()

        (vehicle_velocity, vehicle_turning_rate, vehicle_lateral_velocity, vehicle_acceleration, 
        race_time, vehicle_wheel, contact_ground_material_left, contact_ground_material_right,
        vehicle_engine, vehicle_gear, angle_to_center_line, distance_to_center_line, 
        current_zone_idx, reach_end, has_contact, distance_to_next_checkpt, distance_to_next_next_checkpt,
        distance_to_next_next_next_checkpt, angle_to_next_zone, angle_to_next_next_zone,
        angle_to_next_next_next_zone) = obs

        info = {
            "wait_after_reset": True
        }

        obs = {
            'vehicle_velocity': np.array([vehicle_velocity], dtype='float32'),
            'vehicle_turning_rate': np.array([vehicle_turning_rate], dtype='float32'),
            'vehicle_lateral_velocity': np.array([vehicle_lateral_velocity], dtype='float32'),
            'vehicle_acceleration': np.array([vehicle_acceleration], dtype='float32'),
            'angle_to_center_line': np.array([angle_to_center_line], dtype='float32'),
            'angle_to_next_zone': np.array([angle_to_next_zone], dtype='float32'),
            'angle_to_next_next_zone': np.array([angle_to_next_next_zone], dtype='float32'),
            'angle_to_next_next_next_zone': np.array([angle_to_next_next_next_zone], dtype='float32'),
            'distance_to_center_line': np.array([distance_to_center_line], dtype='float32'),
            'distance_to_next_checkpt': np.array([distance_to_next_checkpt], dtype='float32'),
            'distance_to_next_next_checkpt': np.array([distance_to_next_next_checkpt], dtype='float32'),
            'distance_to_next_next_next_checkpt': np.array([distance_to_next_next_next_checkpt], dtype='float32'),
            # 'contact_ground_material_left': np.array([contact_ground_material_left], dtype='float32'),
            # 'contact_ground_material_right': np.array([contact_ground_material_right], dtype='float32'),
            # 'vehicle_wheel': np.array(vehicle_wheel, dtype='float32'),
            # 'vehicle_engine': np.array([vehicle_engine], dtype='float32'),
            # 'vehicle_gear': np.array([vehicle_gear], dtype='float32'),
        }
        return obs, info
    
    def step(self, actions=None):
        if actions is not None:
            actions = int(actions)
            self.tmi.input(list(inputs[actions].values()))
        (vehicle_velocity, vehicle_turning_rate, vehicle_lateral_velocity, vehicle_acceleration, 
        race_time, vehicle_wheel, contact_ground_material_left, contact_ground_material_right,
        vehicle_engine, vehicle_gear, angle_to_center_line, distance_to_center_line, 
        current_zone_idx, reach_end, has_contact, distance_to_next_checkpt, distance_to_next_next_checkpt,
        distance_to_next_next_next_checkpt, angle_to_next_zone, angle_to_next_next_zone,
        angle_to_next_next_next_zone) = self.tmi.get_obs()

        rew = 0
        rew += vehicle_velocity - abs((angle_to_center_line + angle_to_next_zone/2 + angle_to_next_next_zone/6+ angle_to_next_next_next_zone/8)/4*100)

        if actions is not None:
            if actions == 0:
                rew += 10 # 2
                print("W")
            elif actions == 1 or actions == 2: # W+LR
                rew += 5 
                print("W+LR")
            elif actions == 4 or actions == 5: # LR
                rew -= 200
                print("LR")
            elif actions == 6:
                rew -= 20
                print("S")
            elif actions == 7 or actions == 8: #S+LR
                rew -= 6
                print("S+LR")
            elif actions == 9:
                rew -= 5
                print("W+S")
            elif actions == 10 or actions == 11: #W+S+LR
                rew -= 1
                print("W+S+LR")
            elif actions == 3:
                rew -= 100
                print("Nothing")

        if vehicle_velocity < -0.01:
            rew -= 100
        terminated = reach_end

        if terminated:
            time.sleep(0.5)
            mult_fraction = 15000 / race_time
            rew += 500 * (mult_fraction ** 3)
        info = {
            "wait_after_reset": True if race_time < 1000 else False
        }
        truncated = True if race_time > 30000 else False

        if not truncated and race_time > 1000:
            truncated = has_contact
        if not truncated and race_time > 1000 and vehicle_velocity < 0.5:
            truncated = True

        if truncated:
            self.tmi.respawn()
            rew -= 100
            time.sleep(0.5)

        obs = {
            'vehicle_velocity': np.array([vehicle_velocity], dtype='float32'),
            'vehicle_turning_rate': np.array([vehicle_turning_rate], dtype='float32'),
            'vehicle_lateral_velocity': np.array([vehicle_lateral_velocity], dtype='float32'),
            'vehicle_acceleration': np.array([vehicle_acceleration], dtype='float32'),
            'angle_to_center_line': np.array([angle_to_center_line], dtype='float32'),
            'angle_to_next_zone': np.array([angle_to_next_zone], dtype='float32'),
            'angle_to_next_next_zone': np.array([angle_to_next_next_zone], dtype='float32'),
            'angle_to_next_next_next_zone': np.array([angle_to_next_next_next_zone], dtype='float32'),
            'distance_to_center_line': np.array([distance_to_center_line], dtype='float32'),
            'distance_to_next_checkpt': np.array([distance_to_next_checkpt], dtype='float32'),
            'distance_to_next_next_checkpt': np.array([distance_to_next_next_checkpt], dtype='float32'),
            'distance_to_next_next_next_checkpt': np.array([distance_to_next_next_next_checkpt], dtype='float32'),
            # 'contact_ground_material_left': np.array([contact_ground_material_left], dtype='float32'),
            # 'contact_ground_material_right': np.array([contact_ground_material_right], dtype='float32'),
            # 'vehicle_wheel': np.array(vehicle_wheel, dtype='float32'),
            # 'vehicle_engine': np.array([vehicle_engine], dtype='float32'),
            # 'vehicle_gear': np.array([vehicle_gear], dtype='float32'),
        }
        return obs, rew, terminated, truncated, info

    
    def render(self):
        pass

    def close(self):
        pass