import time
import tqdm
import wandb
import torch
import multiprocessing
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils_drone import HjAviary
from utils_rl import PPOBuffer, MLPActorCritic, collect_experience_once, update

DEVICE = torch.device("cpu")
RESUME_NAME = "5900X_actionMotor_scaling1_bsS2000E20000_init10_rewardV5_20250412"
SAVE_PATH = "./data/interim/para_actionMotor_scaling1_bsS2000E20000_init10_rewardV5.pt"
EPOCH = 300  # 200 1000 5000 2000
LOAD_FROM = None  # None "./data/interim/para_actionMotor_temp.pt"
PERCENT_MODE = False  # True False


def setup_wandb():
    wandb.init(
        # mode="offline",
        project="project-drone-20241115",
        resume=RESUME_NAME  # HjScenarioEnv
    )


def setup_environment():
    env = HjAviary(gui=False)  # , ctrl_freq=10, pyb_freq=100
    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)
    return env


def setup_actor_critic(env):
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    ac = MLPActorCritic(env.observation_space, env.action_space)  # , hidden_sizes=(64, 128, 128)
    if LOAD_FROM is not None:
        state_dict = torch.load(LOAD_FROM, map_location=torch.device(DEVICE))
        ac.load_state_dict(state_dict)
    return ac, obs_dim, act_dim


def setup_optimizers_and_schedulers(ac, pi_lr, vf_lr):
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    if not PERCENT_MODE:
        # 添加 Cosine Annealing 调度器
        scheduler_pi = CosineAnnealingLR(optimizer=pi_optimizer, T_max=EPOCH, eta_min=1e-5)  # T_max 设为总 epoch 数
        scheduler_vf = CosineAnnealingLR(optimizer=vf_optimizer, T_max=EPOCH, eta_min=1e-5)  # T_max 设为总 epoch 数
    else:
        scheduler_pi = scheduler_vf = None
    return pi_optimizer, vf_optimizer, scheduler_pi, scheduler_vf


def collect_data(ac, bs_end, bs_start, env, epoch, list_ep_ret, max_ep_len, train_pi_iters):
    local_steps_per_epoch = int((bs_end - bs_start) * epoch / EPOCH + bs_start)
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    replay_buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim, size=local_steps_per_epoch)  # size=int(1e6)
    wandb.log({"7_1 spup increase/Epoch": (epoch + 1)})
    time_start_collect_experience_once = time.time()
    if PERCENT_MODE:
        percent = epoch / EPOCH
    else:
        percent = 1
    collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer, list_ep_ret, percent)
    time_collect_experience_once = time.time() - time_start_collect_experience_once
    wandb.log({"8 throughout/TimeCollectExperienceOnce": time_collect_experience_once})
    wandb.log({"8 throughout/EnvRateWithReset": local_steps_per_epoch / time_collect_experience_once})
    wandb.log({"7_1 spup increase/TotalEnvInteracts": (epoch + 1) * local_steps_per_epoch})
    life_long_time = time.time() - life_long_time_start
    wandb.log({"7_1 spup increase/Time": life_long_time})
    # wandb.log({"8 throughout/LifeLongEnvRate": (epoch + 1) * local_steps_per_epoch / life_long_time})
    wandb.log({"8 throughout/LifeLongUpdateRate": (epoch + 1) * train_pi_iters / life_long_time})
    data = replay_buffer.get(device=DEVICE)
    return data


def print_hello_world(num, result_queue):
    message = f"hello world {num}"

    print(message)
    result_queue.put(message)  # 将结果放入队列



def run_epoch(epoch, ac, env, pi_optimizer, vf_optimizer, scheduler_pi, scheduler_vf, list_ep_ret,
              bs_start, bs_end, max_ep_len, clip_ratio, train_pi_iters, train_v_iters, target_kl):
    result_queue = multiprocessing.Queue()
    processes = []
    for i in range(1, 5):
        p = multiprocessing.Process(target=print_hello_world, args=(i, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    print("从子进程获取的结果:", results)

    data = collect_data(ac, bs_end, bs_start, env, epoch, list_ep_ret, max_ep_len, train_pi_iters)

    update_time_once = update(data, ac, clip_ratio, train_pi_iters, train_v_iters, pi_optimizer, vf_optimizer, target_kl)
    wandb.log({"8 throughout/TimeUpdateOnce": update_time_once})

    if not PERCENT_MODE:
        # 调整学习率
        scheduler_pi.step()
        scheduler_vf.step()

    torch.save(ac.state_dict(), SAVE_PATH)


def main():
    bs_start = 2000
    bs_end = 30000
    max_ep_len = 500
    clip_ratio = 0.2  # 0.1 0.07 0.2
    train_pi_iters = 80
    train_v_iters = 80
    pi_lr = 2e-4  # 初始学习率  # 3e-4 2e-4
    vf_lr = 1e-3  # 固定学习率
    target_kl = 0.01

    global life_long_time_start
    life_long_time_start = time.time()

    setup_wandb()
    env = setup_environment()
    ac, obs_dim, act_dim = setup_actor_critic(env)
    pi_optimizer, vf_optimizer, scheduler_pi, scheduler_vf = setup_optimizers_and_schedulers(ac, pi_lr, vf_lr)

    list_ep_ret = []

    for epoch in tqdm.tqdm(range(EPOCH)):
        run_epoch(epoch, ac, env, pi_optimizer, vf_optimizer, scheduler_pi, scheduler_vf, list_ep_ret,
                  bs_start, bs_end, max_ep_len, clip_ratio, train_pi_iters, train_v_iters, target_kl)

    print("Finished...")


if __name__ == "__main__":
    main()
