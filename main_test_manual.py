import time
import wandb

import matplotlib.pyplot as plt
import torch
import numpy as np

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")
CONTROL_MODE = "PID"  # PID RL
PID_MODE = "vel"  # vel pos


def main():
    wandb.init(
        project="project-drone-test-20241122",
    )
    env = HjAviary(gui=True)

    ac = MLPActorCritic(env.observation_space, env.action_space)

    state_dict = torch.load("./data/interim/para_temp.pt",
                            map_location=torch.device(DEVICE))
    ac.load_state_dict(state_dict)

    for i in range(1):
        obs_ma, info = env.reset()
        list_vel_z = []
        list_vel_x = []
        for j in range(1000):
            if CONTROL_MODE == "RL":
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                action, _, _ = ac.step(obs_tensor)
            elif CONTROL_MODE == "PID":
                obs_12 = obs_ma[:, :12]
                pos = obs_12[:, 0:3]
                rpy = obs_12[:, 3:6]
                vel = obs_12[:, 6:9]
                ang = obs_12[:, 9:12]
                ang_my = ang[0, 0]
                ang_x = ang[0, 1]
                ang_z = ang[0, 2]
                vel_x = vel[0, 0]
                vel_y = vel[0, 1]
                vel_z = vel[0, 2]
                pos_z = pos[0, 2]
                wandb.log({"ang/ang_my": ang_my})
                wandb.log({"ang/ang_x": ang_x})
                wandb.log({"ang/ang_z": ang_z})
                wandb.log({"vel/vel_x": vel_x})
                wandb.log({"vel/vel_y": vel_y})
                wandb.log({"vel/vel_z": vel_z})
                if PID_MODE == "vel":
                    # vel
                    goal_vel_z = 0.5
                    vel_z_bias = vel_z - goal_vel_z
                    action_vel_z = vel_z_bias * -20
                    goal_vel_x = 0.5
                    # if vel_x < goal_vel_x:
                    #     goal_ang_x = 0.1
                    # else:
                    #     goal_ang_x = -0.1
                    goal_ang_x = (goal_vel_x - vel_x) * 0.02  # 0.02~0.05
                    # ang
                    action_ang_x = (goal_ang_x - ang_x) * 0.2
                    action_ang_my = 0  # 0.01
                    action_ang_z = 0  # 0.01
                elif PID_MODE == "pos":
                    action_vel_z = 0
                    action_ang_my = 0  # 0.01
                    action_ang_x = 0  # 0.01
                    action_ang_z = 0  # 0.01
                else:
                    raise
                wandb.log({"action/action_vel_z": action_vel_z})
                wandb.log({"action/action_ang_x": action_ang_x})
                wandb.log({"action/action_ang_my": action_ang_my})
                wandb.log({"action/action_ang_z": action_ang_z})
                action = [action_vel_z - action_ang_my - action_ang_x - action_ang_z,
                          action_vel_z - action_ang_my + action_ang_x + action_ang_z,
                          action_vel_z + action_ang_my + action_ang_x - action_ang_z,
                          action_vel_z + action_ang_my - action_ang_x + action_ang_z]
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


if __name__ == "__main__":
    main()
