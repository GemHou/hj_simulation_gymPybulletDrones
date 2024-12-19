import time
import wandb

import torch
import numpy as np

from utils_rl import MLPActorCritic
from utils_drone import HjAviaryActionAngRes

DEVICE = torch.device("cpu")
CONTROL_MODE = "RL"  # PID RL
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



def main():
    wandb.init(
        project="project-drone-test-20241219",
    )
    env = HjAviaryActionAngRes(gui=True)

    if CONTROL_MODE == "RL":
        ac = MLPActorCritic(env.observation_space, env.action_space)
        state_dict = torch.load("./data/interim/para_actionAngRes_temp.pt",
                                map_location=torch.device(DEVICE))
        ac.load_state_dict(state_dict)
    else:
        ac = None

    for i in range(1):
        obs_ma, info = env.reset(PERCENT)
        for j in range(1000):
            # if CONTROL_MODE == "RL":
            ang_my, ang_x, pos_z, pos_x, pos_y, vel_x, vel_y, vel_z = analyse_obs(obs_ma)
            obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
            action_ang, _, _ = ac.step(obs_tensor)
            # elif CONTROL_MODE == "PID":
            #     action_ang = generate_action_pid(obs_ma, goal_pos_z=3, goal_pos_x=0, goal_pos_y=0)  # , goal_vel_x=5, goal_vel_y=2.5
            # else:
            #     raise
            # action_ang = [0, 0, 0, 0]
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
