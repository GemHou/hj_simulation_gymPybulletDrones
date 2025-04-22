import time
import wandb

import matplotlib.pyplot as plt
import torch
import numpy as np

from utils_rl import MLPActorCritic
from utils_drone import HjAviaryActionAng

DEVICE = torch.device("cpu")
CONTROL_MODE = "PID"  # PID RL
PERCENT = 1


def analyse_obs(obs_ma):
    obs_12 = obs_ma[:, :12]
    pos = obs_12[:, 0:3]
    rpy = obs_12[:, 3:6]
    vel = obs_12[:, 6:9]
    ang = obs_12[:, 9:12]
    ang_my = ang[0, 0]
    ang_x = ang[0, 1]
    ang_z = ang[0, 2]
    wandb.log({"y/ang_my": ang_my})
    wandb.log({"x/ang_x": ang_x})
    wandb.log({"z/ang_z": ang_z})
    vel_x = vel[0, 0]
    vel_y = vel[0, 1]
    vel_z = vel[0, 2]
    wandb.log({"x/vel_x": vel_x})
    wandb.log({"y/vel_y": vel_y})
    wandb.log({"z/vel_z": vel_z})
    pos_x = pos[0, 0]
    pos_y = pos[0, 1]
    pos_z = pos[0, 2]
    wandb.log({"x/pos_x": pos_x})
    wandb.log({"y/pos_y": pos_y})
    wandb.log({"z/pos_z": pos_z})
    return ang_my, ang_x, pos_z, pos_x, pos_y, vel_x, vel_y, vel_z


def generate_action_pid(obs_ma,
                        goal_pos_z=2.5,
                        goal_vel_x=None,
                        goal_vel_y=None,
                        goal_pos_x=None,
                        goal_pos_y=None):
    ang_my, ang_x, pos_z, pos_x, pos_y, vel_x, vel_y, vel_z = analyse_obs(obs_ma)
    # goal_pos
    # goal_pos_z
    wandb.log({"z/goal_pos_z": goal_pos_z})
    wandb.log({"x/goal_pos_x": goal_pos_x})
    wandb.log({"y/goal_pos_y": goal_pos_y})
    # goal_vel
    goal_vel_z = (goal_pos_z - pos_z) * 0.5
    # goal_vel_x
    if goal_pos_x is not None:
        goal_vel_x = (goal_pos_x - pos_x) * 0.2 - vel_x * 0.1
    # goal_vel_y
    if goal_pos_y is not None:
        goal_vel_y = (goal_pos_y - pos_y) * 0.2 - vel_y * 0.1
    wandb.log({"z/goal_vel_z": goal_vel_z})
    wandb.log({"x/goal_vel_x": goal_vel_x})
    goal_vel_x = np.clip(goal_vel_x, -2, 2)
    wandb.log({"y/goal_vel_y": goal_vel_y})
    goal_vel_y = np.clip(goal_vel_y, -2, 2)
    # goal_ang
    goal_ang_x = (goal_vel_x - vel_x) * 0.2  # 0.02~0.05
    goal_ang_x = np.clip(goal_ang_x, -0.5, 0.5)
    goal_ang_my = (goal_vel_y - vel_y) * -0.02
    goal_ang_my = np.clip(goal_ang_my, -0.05, 0.05)
    wandb.log({"x/goal_ang_x": goal_ang_x})
    wandb.log({"y/goal_ang_my": goal_ang_my})
    action_ang = [goal_ang_x, goal_ang_my, goal_vel_z]

    return action_ang


def main():
    wandb.init(
        project="project-drone-test-20250422",
    )
    env = HjAviaryActionAng(gui=True)

    if CONTROL_MODE == "RL":
        ac = MLPActorCritic(env.observation_space, env.action_space)
        state_dict = torch.load("./data/interim/para_actionAng_temp.pt",
                                map_location=torch.device(DEVICE))
        ac.load_state_dict(state_dict)
    else:
        ac = None

    for i in range(1):
        obs_ma, info = env.reset(PERCENT)
        for j in range(1000):
            if CONTROL_MODE == "RL":
                ang_my, ang_x, pos_z, pos_x, pos_y, vel_x, vel_y, vel_z = analyse_obs(obs_ma)
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                action_ang, _, _ = ac.step(obs_tensor)
            elif CONTROL_MODE == "PID":
                action_ang = generate_action_pid(obs_ma, goal_pos_z=3, goal_pos_x=0, goal_pos_y=0)  # , goal_vel_x=5, goal_vel_y=2.5
            else:
                raise
            # if j % 50 < 20:
            #     action_ang = [2, 2, 2, 0]
            # else:
            #     action_ang = [2, 2, -2, 0]
            action_ma = np.array([action_ang])
            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72
            obs_ma = next_obs_ma

            env.render()
            time.sleep(1 / 30)  # * 10

            if done:
                break


if __name__ == "__main__":
    main()
