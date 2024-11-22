import time

import matplotlib.pyplot as plt
import torch
import numpy as np

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")
CONTROL_MODE = "PID"  # PID RL
PID_MODE = "vel"  # vel pos


def main():
    env = HjAviary(gui=True)

    ac = MLPActorCritic(env.observation_space, env.action_space)

    state_dict = torch.load("./data/interim/para_temp.pt",
                            map_location=torch.device(DEVICE))
    ac.load_state_dict(state_dict)

    for i in range(1):
        obs_ma, info = env.reset()
        list_vel_z = []
        for j in range(100):
            if CONTROL_MODE == "RL":
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                action, _, _ = ac.step(obs_tensor)
            elif CONTROL_MODE == "PID":
                obs_12 = obs_ma[:, :12]
                pos = obs_12[:, 0:3]
                rpy = obs_12[:, 3:6]
                vel = obs_12[:, 6:9]
                ang = obs_12[:, 9:12]
                pos_z = pos[0, 2]
                vel_z = vel[0, 2]
                list_vel_z.append(vel_z)
                if PID_MODE == "vel":
                    goal_vel_z = 0.5
                    print("vel_z: ", vel_z)
                    vel_z_bias = vel_z - goal_vel_z
                    action_z = vel_z_bias * -20
                elif PID_MODE == "pos":
                    action_z = 0
                else:
                    raise
                action = [action_z, action_z, action_z, action_z]
            else:
                raise
            action_ma = np.array([action])
            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72
            obs_ma = next_obs_ma

            env.render()
            time.sleep(1 / 30)

            if done:
                break
        plt.plot(list_vel_z)
        plt.show()


if __name__ == "__main__":
    main()
