import time

import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary


def main():
    env = HoverAviary(gui=True)

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)

    for t in range(50):
        action = np.ones((1, 4))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        # time.sleep(0.1)

    print("Finished...")


if __name__ == "__main__":
    main()
