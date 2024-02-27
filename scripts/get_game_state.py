import math
import time

import numpy as np
import psutil

from ReadWriteMemory import ReadWriteMemory
from tminterface.interface import Message, MessageType, TMInterface

class TMInterfaceManager:
    def __init__(
        self,
        base_dir,
        running_speed=1,
        run_steps_per_action=10,
        max_overall_duration_ms=2000,
        max_minirace_duration_ms=2000,
        interface_name="TMInterface0",
        zone_centers=None,
        simulation_state=None,
        iface=None,
        current_zone_idx=0,
    ):
        self.base_dir = base_dir
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_ms = max_overall_duration_ms
        self.max_minirace_duration_ms = max_minirace_duration_ms
        self.interface_name = interface_name
        self.zone_centers = zone_centers
        self.current_zone_idx = current_zone_idx
        self.reload = False
        self.iface = None

        remove_fps_cap()

    def rollout(self):
        if not self.iface:
            self.setup()
        if self.reload:
            self.respawn()
            return self.get_obs()
        
        return self.get_obs()

    def get_obs(self):
        zone_centers = self.zone_centers
        current_zone_center = zone_centers[self.current_zone_idx, :]
        next_zone_center = zone_centers[self.current_zone_idx + 1, :]

        vehicle_state = self.iface.get_simulation_state()

        # Vechicle Velocity =============================================== #
        vehicle_orientation_matrix = vehicle_state.dyna.current_state.rotation.to_numpy()
        vehicle_velocity_matrix = np.array(
                                vehicle_state.dyna.current_state.linear_speed,
                                dtype=np.float32,
                            )
        vehicle_velocity_in_car_reference_system_matrix = np.dot(vehicle_velocity_matrix, vehicle_orientation_matrix)
        vehicle_velocity = np.linalg.norm(vehicle_velocity_in_car_reference_system_matrix)
        if vehicle_velocity_in_car_reference_system_matrix[2] < 0:
            vehicle_velocity = -vehicle_velocity

        # Vechicle Turning Rate =============================================== #
        vehicle_turning_rate = vehicle_state.scene_mobil.turning_rate
        
        # Lateral Velocity =============================================== #
        vehicle_lateral_velocity_matrix = vehicle_state.dyna.current_state.angular_speed
        vehicle_lateral_velocity = np.linalg.norm(vehicle_lateral_velocity_matrix)
        if vehicle_lateral_velocity_matrix[1] > 0:
            vehicle_lateral_velocity = -vehicle_lateral_velocity

        # Vechicle Acceleration =============================================== #
        last_vehicle_orientation_matrix = vehicle_state.dyna.previous_state.rotation.to_numpy()
        last_vehicle_velocity_matrix = np.array(
                                vehicle_state.dyna.previous_state.linear_speed,
                                dtype=np.float32,
                            )
        last_vehicle_velocity_in_car_reference_system_matrix = np.dot(last_vehicle_velocity_matrix, last_vehicle_orientation_matrix)
        last_vehicle_velocity = np.linalg.norm(last_vehicle_velocity_in_car_reference_system_matrix)
        if last_vehicle_velocity_in_car_reference_system_matrix[2] < 0:
            last_vehicle_velocity = -last_vehicle_velocity

        vehicle_acceleration = vehicle_velocity - last_vehicle_velocity
        # Race Time =============================================== #
        race_time = vehicle_state.race_time

        # Vehicle Wheel and Engine State =============================================== #
        vehicle_wheel = [
                            vehicle_state.simulation_wheels[0].real_time_state.is_sliding,
                            vehicle_state.simulation_wheels[1].real_time_state.is_sliding,
                            vehicle_state.simulation_wheels[2].real_time_state.is_sliding,
                            vehicle_state.simulation_wheels[3].real_time_state.is_sliding
                        ]
        vehicle_gear = vehicle_state.scene_mobil.engine.gear
        vehicle_engine = vehicle_state.scene_mobil.engine.actual_rpm

        # Vehicle Angle to Center Line =============================================== #
        quat = vehicle_state.dyna.current_state.quat.to_numpy()
        # convert quat to rotation matrix
        R = np.array([[1 - 2*(quat[2]**2 + quat[3]**2), 2*(quat[1]*quat[2] - quat[3]*quat[0]), 2*(quat[1]*quat[3] + quat[2]*quat[0])],
                        [2*(quat[1]*quat[2] + quat[3]*quat[0]), 1 - 2*(quat[1]**2 + quat[3]**2), 2*(quat[2]*quat[3] - quat[1]*quat[0])],
                        [2*(quat[1]*quat[3] - quat[2]*quat[0]), 2*(quat[2]*quat[3] + quat[1]*quat[0]), 1 - 2*(quat[1]**2 + quat[2]**2)]])

        line_vector = next_zone_center - current_zone_center
        transformed_line_vector = np.dot(line_vector, R)
        angle_to_center_line = np.arccos(np.dot(transformed_line_vector, [1, 0, 0]) / (np.linalg.norm(transformed_line_vector) * np.linalg.norm([1, 0, 0])))
        angle_to_center_line = -(np.degrees(angle_to_center_line) - 90) / 90

        # Vehicle Angle to Next Zone =============================================== #
        next_zone_line_vector = zone_centers[self.current_zone_idx + 2].astype(float) - next_zone_center
        transformed_next_zone_line_vector = np.dot(next_zone_line_vector, R)
        angle_to_next_zone = np.arccos(np.dot(transformed_next_zone_line_vector, [1, 0, 0]) / (np.linalg.norm(transformed_next_zone_line_vector) * np.linalg.norm([1, 0, 0])))
        angle_to_next_zone = -(np.degrees(angle_to_next_zone) - 90) / 90

        # Vehicle Angle to Next Next Zone =============================================== #
        next_next_zone_line_vector = zone_centers[self.current_zone_idx + 3].astype(float) - zone_centers[self.current_zone_idx + 2].astype(float)
        transformed_next_next_zone_line_vector = np.dot(next_next_zone_line_vector, R)
        angle_to_next_next_zone = np.arccos(np.dot(transformed_next_next_zone_line_vector, [1, 0, 0]) / (np.linalg.norm(transformed_next_next_zone_line_vector) * np.linalg.norm([1, 0, 0])))
        angle_to_next_next_zone = -(np.degrees(angle_to_next_next_zone) - 90) / 90

        # Vehicle Angle to Next Next Next Zone =============================================== #
        next_next_next_zone_line_vector = zone_centers[self.current_zone_idx + 4].astype(float) - zone_centers[self.current_zone_idx + 3].astype(float)
        transformed_next_next_next_zone_line_vector = np.dot(next_next_next_zone_line_vector, R)
        angle_to_next_next_next_zone = np.arccos(np.dot(transformed_next_next_next_zone_line_vector, [1, 0, 0]) / (np.linalg.norm(transformed_next_next_next_zone_line_vector) * np.linalg.norm([1, 0, 0])))
        angle_to_next_next_next_zone = -(np.degrees(angle_to_next_next_next_zone) - 90) / 90

        # Vehicle Position =============================================== #
        vehicle_position = vehicle_state.dyna.current_state.position.to_numpy()

        # Vehicle Contact =============================================== #
        has_contact = vehicle_state.scene_mobil.has_any_lateral_contact
        
        # Contact Ground Material =============================================== #
        contact_ground_material_left = vehicle_state.simulation_wheels[0].real_time_state.contact_material_id
        contact_ground_material_right = vehicle_state.simulation_wheels[1].real_time_state.contact_material_id

        # Distance to Center Line =============================================== #
        next_zone_center = next_zone_center.astype(float)
        current_zone_center = current_zone_center.astype(float)
        vehicle_position = vehicle_position.astype(float)
        line_vector = next_zone_center - current_zone_center
        line_direction = line_vector / np.linalg.norm(line_vector)
        distance_to_center_line_matrix =np.cross(next_zone_center-current_zone_center, current_zone_center-vehicle_position)/np.linalg.norm(next_zone_center-current_zone_center)
        if distance_to_center_line_matrix[1] >= 0:
            distance_to_center_line = math.sqrt(distance_to_center_line_matrix[0] ** 2 + distance_to_center_line_matrix[1] ** 2)
        elif distance_to_center_line_matrix[1] < 0:
            distance_to_center_line = -math.sqrt(distance_to_center_line_matrix[0] ** 2 + distance_to_center_line_matrix[1] ** 2)

        # Calculate the distance to current zone and next zone =============================================== #
        d1 = np.linalg.norm(next_zone_center - vehicle_position)
        d2 = np.linalg.norm(current_zone_center - vehicle_position)
        d3 = np.linalg.norm(zone_centers[self.current_zone_idx + 2].astype(float) - vehicle_position)
        d4 = np.linalg.norm(zone_centers[self.current_zone_idx + 3].astype(float) - vehicle_position)

        reach_end = False
        if (d1 <= d2 and self.current_zone_idx < len(zone_centers) - 6):
            self.current_zone_idx += 1
            current_zone_center = zone_centers[self.current_zone_idx]
            next_zone_center = zone_centers[1 + self.current_zone_idx]
        elif self.current_zone_idx != 0 and d2 > d1:
            simulation_state = self.iface.get_simulation_state()
            simulation_state.cp_data.cp_times[-1].time = -1 
            # self.rewind_to_state(simulation_state)
            self.respawn()
            reach_end = True
        
        racing_line_rew = (abs((angle_to_center_line + angle_to_next_zone/2 + angle_to_next_next_zone/6+ angle_to_next_next_next_zone/8)/4*100))**2
        return [vehicle_velocity, vehicle_turning_rate, vehicle_lateral_velocity, vehicle_acceleration, race_time, vehicle_wheel, 
                contact_ground_material_left, contact_ground_material_right,
                vehicle_engine, vehicle_gear, angle_to_center_line, distance_to_center_line, self.current_zone_idx, 
                reach_end, has_contact, d1, d3, d4, angle_to_next_zone, angle_to_next_next_zone, 
                angle_to_next_next_next_zone]

    def rewind_to_state(self, state):
        msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
        msg.write_buffer(state.data)
        self.iface._send_message(msg)
        self.iface._wait_for_server_response()
        self.reload = True

    def setup(self):
        self.iface = TMInterface("TMInterface0")
        self.iface.registered = False
        msg = Message(MessageType.C_REGISTER)
        self.iface._send_message(msg)
        self.iface._wait_for_server_response()
        self.iface.registered = True
        while not self.iface._ensure_connected():
            time.sleep(0)
            continue
        if self.iface.mfile is None:
            return
        self.iface.mfile.seek(0)
        msgtype = self.iface._read_int32()

    def input(self, action):
        if not self.iface:
            self.setup()
        w, a, s, d = 0, 0, 0, 0
        if action is not None:
            if action[0] > 0:
                a = 1
            if action[1] > 0:
                d = 1
            if action[2] > 0:
                w = 1
            if action[3] > 0:
                s = 1
        self.iface.set_input_state(brake=s, accelerate=w, left=a, right=d)


    def respawn(self):
        if not self.iface:
            self.setup()
        self.current_zone_idx = 0
        self.iface.respawn()
        self.iface.set_timeout(1)
        self.reload = False
        return self.get_obs()


def remove_fps_cap():
    process = filter(lambda pr: pr.name() == "TmForever.exe", psutil.process_iter())
    rwm = ReadWriteMemory()
    for p in process:
        pid = int(p.pid)
        process = rwm.get_process_by_id(pid)
        process.open()
        process.write(0x005292F1, 4294919657)
        process.write(0x005292F1 + 4, 2425393407)
        process.write(0x005292F1 + 8, 2425393296)
        process.close()
        print(f"Disabled FPS cap of process {pid}")
