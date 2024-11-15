import time

import torch
import numpy as np

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")


def main():
    env = HjAviary(gui=True)

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

            env.render()
            time.sleep(1/30)

            if done:
                break


if __name__ == "__main__":
    main()
