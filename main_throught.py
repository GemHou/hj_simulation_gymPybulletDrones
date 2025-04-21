import time
import wandb
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")
CONTROL_MODE = "RL"  # RL
PERCENT = 1.0
MAX_EP_LEN = 350
LOAD_PATH = "./data/interim/para_randomTMove_obs81_scenario_39.pt"  # _041212
RENDER = False

def main():
    wandb.init(
        project="project-drone-test-20241122",
    )
    env = HjAviary(gui=RENDER)  # , ctrl_freq=10, pyb_freq=100

    ac = MLPActorCritic(env.observation_space, env.action_space)  # , hidden_sizes=(128, 128, 128)

    for i in tqdm(range(100)):
        state_dict = torch.load(LOAD_PATH,
                                map_location=torch.device(DEVICE))
        ac.load_state_dict(state_dict)

        obs_ma, info = env.reset(PERCENT)
        for j in range(MAX_EP_LEN):
            if CONTROL_MODE == "RL":
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                action, _, _ = ac.step(obs_tensor)
            else:
                raise
            action_ma = np.array([action])
            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72
            obs_ma = next_obs_ma

            # env.render()
            time.sleep(1 / 30)  #  * 10

            if done:
                if j > 10:
                    time.sleep(0.1)
                break
            else:
                if j ==0:
                    time.sleep(0.1)


if __name__ == "__main__":
    main()
