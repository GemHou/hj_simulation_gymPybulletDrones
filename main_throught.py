import time
import wandb
import torch
import pybullet
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")
CONTROL_MODE = "RL"  # RL
PERCENT = 1.0
MAX_EP_LEN = 1000
LOAD_PATH = "./data/interim/para_randomTMove_obs81_scenario_39.pt"  # _041212
RENDER = False


def main():
    wandb.init(
        project="project-drone-test-20241122",
    )
    env = HjAviary(gui=RENDER)  # , ctrl_freq=10, pyb_freq=100

    ac = MLPActorCritic(env.observation_space, env.action_space)  # , hidden_sizes=(128, 128, 128)

    for i in tqdm(range(100)):
        start_time_1 = time.time()
        obs_ma, info = env.reset(PERCENT)
        start_time_2 = time.time()
        for j in range(MAX_EP_LEN):
            if CONTROL_MODE == "RL":
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                action, _, _ = ac.step(obs_tensor)
            else:
                raise
            action_ma = np.array([action])
            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72

            # env.IMG_RES = np.array([64, 48])
            # temp = env._getDroneImages(0)
            #
            # for i in range(128*128):
            #     ray_from = [1, 2, 3]
            #     ray_to = [1, 2, 3]
            #     ray_results = pybullet.rayTest(ray_from, ray_to)

            obs_ma = next_obs_ma

        fps_reset = j / (time.time() - start_time_1)
        fps_loop = j / (time.time() - start_time_2)
        print("fps_loop: ", fps_loop, "fps_reset: ", fps_reset)


if __name__ == "__main__":
    main()
