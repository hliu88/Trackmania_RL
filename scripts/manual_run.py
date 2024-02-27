import sys
import numpy as np
from pathlib import Path
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

Path("maps").mkdir(exist_ok=True)

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.race_finished = False
        self.raw_position_list = []
        self.period_save_pos_ms = 10
        self.target_distance_between_cp_m = 10
        self.zone_centers = None

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()

        if _time == 0:
            self.raw_position_list = []

        if _time >= 0 and _time % self.period_save_pos_ms == 0:
            if not self.race_finished:
                self.raw_position_list.append(np.array(state.position))


    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        if current == target:
            self.raw_position_list.append(np.array(iface.get_simulation_state().position))
            self.race_finished = True
            self.extract_cp_distance_interval()

    def extract_cp_distance_interval(self):
        a = np.array(self.raw_position_list)
        b = np.linalg.norm(a[:-1] - a[1:], axis=1) 
        c = np.pad(b.cumsum(), (1, 0))  
        number_zones = round(c[-1] / self.target_distance_between_cp_m - 0.5) + 0.5  
        zone_length = c[-1] / number_zones
        index_first_pos_in_new_zone = np.unique(c // zone_length, return_index=True)[1][1:]
        index_last_pos_in_current_zone = index_first_pos_in_new_zone - 1
        w1 = 1 - (c[index_last_pos_in_current_zone] % zone_length) / zone_length
        w2 = (c[index_first_pos_in_new_zone] % zone_length) / zone_length
        self.zone_centers = a[index_last_pos_in_current_zone] + (a[index_first_pos_in_new_zone] - a[index_last_pos_in_current_zone]) * (
            w1 / (1e-4 + w1 + w2)
        ).reshape((-1, 1))
        self.zone_centers = np.vstack(
            (
                client.raw_position_list[0][None, :],
                self.zone_centers,
                (2 * client.raw_position_list[-1] - self.zone_centers[-1])[None, :],
            )
        )
        np.save(base_dir / "maps" / "map.npy", np.array(self.zone_centers).round(1))

        save_path = base_dir / "maps" / "map.npy"
        print(f"map.npy was saved successfully to {save_path}")

base_dir = Path(__file__).resolve().parents[1]
server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
print(f"Connecting to {server_name}...")
client = MainClient()
run_client(client, server_name)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(26, 15))
plt.scatter(client.zone_centers[:, 0], client.zone_centers[:, 2], s=0.5)
plt.scatter(
    -np.array(client.raw_position_list)[:, 0],
    np.array(client.raw_position_list)[:, 2],
    s=0.5,
)
