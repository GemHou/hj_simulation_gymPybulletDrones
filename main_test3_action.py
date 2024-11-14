import time

import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary


def main():
    env = HoverAviary(gui=True)

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)

    for t in range(200):
        # action = np.zeros((1, 4))
        # action = np.ones((1, 4))
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print("obs: ", obs)
        # print("obs.shape: ", obs.shape)  # [1, 72]           12 + ACTION_BUFFER_SIZE * 4 = 72
        obs_12 = obs[:,:12]  # [pos 3, rpy 3, vel 3, ang 3]
        # print("obs_12.shape: ", obs_12.shape)  # [1, 12]
        pos = obs_12[:,0:3]
        rpy = obs_12[:, 3:6]
        vel = obs_12[:, 6:9]
        ang = obs_12[:, 9:12]
        reward = pos[0, 2]
        env.render()
        time.sleep(0.1)

    print("Finished...")


if __name__ == "__main__":
    main()
