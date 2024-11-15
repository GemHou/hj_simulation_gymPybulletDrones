import time

import torch
import numpy as np

from utils_rl import MLPActorCritic
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

DEVICE = torch.device("cpu")


def main():
    env = HoverAviary(gui=True)

    ac = MLPActorCritic(env.observation_space, env.action_space)

    state_dict = torch.load("./data/interim/para_temp.pt",
                            map_location=torch.device(DEVICE))
    ac.load_state_dict(state_dict)

    for i in range(10):
        obs_ma, info = env.reset()
        for j in range(100):
            obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
            action, v, logp = ac.step(obs_tensor)
            action_ma = np.array([action])
            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72
            # obs_12 = next_obs_ma[:, :12]  # [1, 12]  # [pos 3, rpy 3, vel 3, ang 3]
            # pos = obs_12[:, 0:3]
            # rpy = obs_12[:, 3:6]
            # vel = obs_12[:, 6:9]
            # ang = obs_12[:, 9:12]
            # reward = pos[0, 2]
            # env.render()
            # time.sleep(1/30)
            # obs_ma = next_obs_ma
            # if reward < 0.05:
            #     done = True
            if done:
                break


if __name__ == "__main__":
    main()
