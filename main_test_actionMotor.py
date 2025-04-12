import time
import wandb

import matplotlib.pyplot as plt
import torch
import numpy as np

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")
CONTROL_MODE = "RL"  # PID RL
PERCENT = 0.1
MAX_EP_LEN = 500
LOAD_PATH = "./data/interim/para_actionMotor_scaling1_bs10000.pt"  # _041212


def analyse_obs(obs_ma):
    obs_12 = obs_ma[:, :12]
    pos = obs_12[:, 0:3]
    rpy = obs_12[:, 3:6]
    vel = obs_12[:, 6:9]
    ang = obs_12[:, 9:12]
    ang_my = ang[0, 0]
    ang_x = ang[0, 1]
    ang_z = ang[0, 2]
    wandb.log({"ang/ang_my": ang_my})
    wandb.log({"ang/ang_x": ang_x})
    wandb.log({"ang/ang_z": ang_z})
    vel_x = vel[0, 0]
    vel_y = vel[0, 1]
    vel_z = vel[0, 2]
    wandb.log({"vel/vel_x": vel_x})
    wandb.log({"vel/vel_y": vel_y})
    wandb.log({"vel/vel_z": vel_z})
    pos_x = pos[0, 0]
    pos_y = pos[0, 1]
    pos_z = pos[0, 2]
    wandb.log({"pos/pos_x": pos_x})
    wandb.log({"pos/pos_y": pos_y})
    wandb.log({"pos/pos_z": pos_z})
    return ang_my, ang_x, pos_z, vel_x, vel_y, vel_z


def generate_action_pid(obs_ma):
    ang_my, ang_x, pos_z, vel_x, vel_y, vel_z = analyse_obs(obs_ma)
    # goal_pos
    goal_pos_z = 2
    wandb.log({"pos/goal_pos_z": goal_pos_z})
    # goal_vel
    goal_vel_z = (goal_pos_z - pos_z) * 0.5
    goal_vel_x = 1
    goal_vel_y = -0.5
    wandb.log({"vel/goal_vel_z": goal_vel_z})
    wandb.log({"vel/goal_vel_x": goal_vel_x})
    wandb.log({"vel/goal_vel_y": goal_vel_y})
    # action_vel
    vel_z_bias = vel_z - goal_vel_z
    action_vel_z = vel_z_bias * -20
    # goal_ang
    goal_ang_x = (goal_vel_x - vel_x) * 0.02  # 0.02~0.05
    goal_ang_my = (goal_vel_y - vel_y) * -0.02
    wandb.log({"ang/goal_ang_x": goal_ang_x})
    wandb.log({"ang/goal_ang_my": goal_ang_my})
    # action_ang
    action_ang_x = (goal_ang_x - ang_x) * 0.2
    action_ang_my = (goal_ang_my - ang_my) * 0.2  # 0.01
    action_ang_z = 0  # 0.01
    # final
    wandb.log({"action/action_vel_z": action_vel_z})
    wandb.log({"action/action_ang_x": action_ang_x})
    wandb.log({"action/action_ang_my": action_ang_my})
    wandb.log({"action/action_ang_z": action_ang_z})
    action = [action_vel_z - action_ang_my - action_ang_x - action_ang_z,
              action_vel_z - action_ang_my + action_ang_x + action_ang_z,
              action_vel_z + action_ang_my + action_ang_x - action_ang_z,
              action_vel_z + action_ang_my - action_ang_x + action_ang_z]
    return action


def main():
    wandb.init(
        project="project-drone-test-20241122",
    )
    env = HjAviary(gui=True)  # , ctrl_freq=10, pyb_freq=100

    ac = MLPActorCritic(env.observation_space, env.action_space)  # , hidden_sizes=(128, 128, 128)

    for i in range(100):
        state_dict = torch.load(LOAD_PATH,
                                map_location=torch.device(DEVICE))
        ac.load_state_dict(state_dict)

        obs_ma, info = env.reset(PERCENT)
        for j in range(MAX_EP_LEN):
            if CONTROL_MODE == "RL":
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                action, _, _ = ac.step(obs_tensor)
            elif CONTROL_MODE == "PID":
                action = generate_action_pid(obs_ma)
            else:
                raise
            action_ma = np.array([action])
            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72
            obs_ma = next_obs_ma

            env.render()
            time.sleep(1 / 30)  #  * 10

            if done:
                break


if __name__ == "__main__":
    main()
