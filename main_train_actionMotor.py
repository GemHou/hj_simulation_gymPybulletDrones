import time
import tqdm
import wandb
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils_drone import HjAviary
from utils_rl import PPOBuffer, MLPActorCritic, collect_experience_once, update

DEVICE = torch.device("cpu")
RESUME_NAME = "5900X_actionMotor_scaling1_bsS2000E20000_20250412"
SAVE_PATH = "./data/interim/para_actionMotor_scaling1_bsS2000E20000.pt"
EPOCH = 200  # 200 1000 5000 2000
LOAD_FROM = None  # None "./data/interim/para_actionMotor_temp.pt"
PERCENT_MODE = False  # True False


def main():
    bs_start = 2000
    bs_end = 20000
    max_ep_len = 500
    clip_ratio = 0.2  # 0.1 0.07 0.2
    train_pi_iters = 80
    train_v_iters = 80
    pi_lr = 2e-4  # 初始学习率  # 3e-4 2e-4
    vf_lr = 1e-3  # 固定学习率
    target_kl = 0.01

    life_long_time_start = time.time()

    wandb.init(
        # mode="offline",
        project="project-drone-20241115",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    env = HjAviary(gui=False)  # , ctrl_freq=10, pyb_freq=100

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)

    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    ac = MLPActorCritic(env.observation_space, env.action_space)  # , hidden_sizes=(64, 128, 128)

    if LOAD_FROM is not None:
        state_dict = torch.load(LOAD_FROM,
                                map_location=torch.device(DEVICE))
        ac.load_state_dict(state_dict)

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    if not PERCENT_MODE:
        # 添加 Cosine Annealing 调度器
        scheduler_pi = CosineAnnealingLR(optimizer=pi_optimizer, T_max=EPOCH, eta_min=1e-5)  # T_max 设为总 epoch 数
        scheduler_vf = CosineAnnealingLR(optimizer=vf_optimizer, T_max=EPOCH, eta_min=1e-5)  # T_max 设为总 epoch 数
    else:
        scheduler_pi = scheduler_vf = None

    list_ep_ret = []

    for epoch in tqdm.tqdm(range(EPOCH)):
        local_steps_per_epoch = int((bs_end - bs_start) * epoch / EPOCH + bs_start)
        replay_buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim, size=local_steps_per_epoch)  # size=int(1e6)

        wandb.log({"7_1 spup increase/Epoch": (epoch + 1)})
        time_start_collect_experience_once = time.time()
        if PERCENT_MODE:
            percent = epoch/EPOCH
        else:
            percent = 1
        collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer, list_ep_ret, percent)
        time_collect_experience_once = time.time() - time_start_collect_experience_once
        wandb.log({"8 throughout/EnvRateWithReset": local_steps_per_epoch / time_collect_experience_once})
        wandb.log({"7_1 spup increase/TotalEnvInteracts": (epoch + 1) * local_steps_per_epoch})
        life_long_time = time.time() - life_long_time_start
        wandb.log({"7_1 spup increase/Time": life_long_time})
        # wandb.log({"8 throughout/LifeLongEnvRate": (epoch + 1) * local_steps_per_epoch / life_long_time})
        wandb.log({"8 throughout/LifeLongUpdateRate": (epoch + 1) * train_pi_iters / life_long_time})

        data = replay_buffer.get(device=DEVICE)

        update(data, ac, clip_ratio, train_pi_iters, train_v_iters, pi_optimizer, vf_optimizer, target_kl)

        if not PERCENT_MODE:
            # 调整学习率
            scheduler_pi.step()
            scheduler_vf.step()

        torch.save(ac.state_dict(), SAVE_PATH)

    print("Finished...")


if __name__ == "__main__":
    main()
