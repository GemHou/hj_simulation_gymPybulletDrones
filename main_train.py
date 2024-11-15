import time
import tqdm
import wandb
import torch
from torch.optim import Adam
# import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from utils_rl import PPOBuffer, MLPActorCritic, collect_experience_once, update

DEVICE = torch.device("cpu")
RESUME_NAME = "hjEnv2-reward-done005-el200-bs500-20241115"


class HjAviary(HoverAviary):
    def _computeReward(self):
        state = self._getDroneStateVector(0)  # pos 3 quat 4 ...
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3]) ** 4)
        pos = state[:3]
        pos_z = pos[2]
        if pos_z < 0.05:
            ret = -10
        else:
            ret = pos_z
        return ret

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[:3]
        pos_z = pos[2]
        if pos_z < 0.05:
            done = True
        else:
            done = False
        return done


def main():
    local_steps_per_epoch = 500
    max_ep_len = 200
    clip_ratio = 0.1
    train_pi_iters = 80
    train_v_iters = 80
    pi_lr = 3e-4
    vf_lr = 1e-3
    target_kl = 0.01

    life_long_time_start = time.time()

    wandb.init(
        # mode="offline",
        project="project-drone-20241115",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    env = HjAviary(gui=False)

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)

    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    replay_buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim, size=local_steps_per_epoch)  # size=int(1e6)

    ac = MLPActorCritic(env.observation_space, env.action_space)

    # state_dict = torch.load("./data/interim/para_temp.pt",
    #                         map_location=torch.device(DEVICE))
    # ac.load_state_dict(state_dict)

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    list_ep_ret = []

    for epoch in tqdm.tqdm(range(1000)):
        wandb.log({"7_1 spup increase/Epoch": (epoch + 1)})
        collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer, list_ep_ret)
        wandb.log({"7_1 spup increase/TotalEnvInteracts": (epoch + 1) * local_steps_per_epoch})
        life_long_time = time.time() - life_long_time_start
        wandb.log({"7_1 spup increase/Time": life_long_time})
        wandb.log({"8 throughout/LifeLongEnvRate": (epoch + 1) * local_steps_per_epoch / life_long_time})
        wandb.log({"8 throughout/LifeLongUpdateRate": (epoch + 1) * train_pi_iters / life_long_time})

        data = replay_buffer.get(device=DEVICE)

        update(data, ac, clip_ratio, train_pi_iters, train_v_iters, pi_optimizer, vf_optimizer, target_kl)

        torch.save(ac.state_dict(), "./data/interim/para_temp.pt")

    print("Finished...")


if __name__ == "__main__":
    main()
