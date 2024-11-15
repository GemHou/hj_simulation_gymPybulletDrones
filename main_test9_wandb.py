import time
import tqdm
import wandb
import torch
import numpy as np
from torch.optim import Adam
# import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from utils_rl import PPOBuffer, MLPActorCritic

DEVICE = torch.device("cpu")


def collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer, list_ep_ret):
    obs_ma, info = env.reset()
    ep_ret, ep_len = 0, 0
    list_epoch_ep_ret = []
    list_epoch_ep_len = []
    list_epoch_rps = []
    list_epoch_reset_time = []
    for t in range(local_steps_per_epoch):
        obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
        action, v, logp = ac.step(obs_tensor)

        action_ma = np.array([action])
        next_obs_ma, reward, done, truncated, info = env.step(action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72

        ep_len += 1
        ep_ret += reward

        obs_12 = next_obs_ma[:, :12]  # [1, 12]  # [pos 3, rpy 3, vel 3, ang 3]
        pos = obs_12[:, 0:3]
        rpy = obs_12[:, 3:6]
        vel = obs_12[:, 6:9]
        ang = obs_12[:, 9:12]
        reward = pos[0, 2]

        replay_buffer.store(obs_tensor, action, reward, v, logp)

        obs_ma = next_obs_ma

        # env.render()
        # time.sleep(0.1)

        timeout = ep_len == max_ep_len
        terminal = done or timeout
        epoch_ended = t == local_steps_per_epoch - 1

        if terminal or epoch_ended:
            if epoch_ended and not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
            if timeout or epoch_ended:
                obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
                _, v, _ = ac.step(obs_tensor)
            else:  # done
                v = 0
            replay_buffer.finish_path(v)
            # print("ep_ret: ", ep_ret)
            list_ep_ret.append(ep_ret)
            # wandb.log({"5 performance/episode return": ep_ret})
            # wandb.log({"5 performance/episode length": ep_len})
            # wandb.log({"5 performance/return per step": ep_ret/ep_len})
            list_epoch_ep_ret.append(ep_ret)
            list_epoch_ep_len.append(ep_len)
            list_epoch_rps.append(ep_ret / ep_len)
            reset_time_start = time.time()
            obs_ma, info = env.reset()
            reset_time = time.time() - reset_time_start
            # wandb.log({"8 throughout/AverageResetTime": reset_time})
            list_epoch_reset_time.append(reset_time)
            ep_ret, ep_len = 0, 0
    wandb.log({"5 performance/episode return": np.mean(list_epoch_ep_ret)})
    wandb.log({"5 performance/episode length": np.mean(list_epoch_ep_len)})
    wandb.log({"5 performance/return per step": np.mean(list_epoch_rps)})
    wandb.log({"8 throughout/AverageResetTime": np.mean(list_epoch_reset_time)})


def compute_loss_pi_with_entropy(data, ac, clip_ratio,
                                 ent_coeff=0.001):  # You can adjust the value of ent_coeff ent_coeff= 0.001~0.02~0.1
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi_policy = -(torch.min(ratio * adv, clip_adv)).mean()

    # Entropy bonus
    ent = pi.entropy().mean()
    loss_pi_entropy = -ent  # Negative sign for gradient ascent
    loss_pi_entropy = ent_coeff * loss_pi_entropy
    loss_pi = loss_pi_policy + loss_pi_entropy  # Adding entropy term to the loss
    # loss_pi = torch.clamp(loss_pi, 0, 0.5)

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent.item(), cf=clipfrac, loss_pi_policy=loss_pi_policy,
                   loss_pi_entropy=loss_pi_entropy)

    return loss_pi, pi_info


# Set up function for computing value loss
def compute_loss_v(data, ac):
    obs, ret = data['obs'], data['ret']
    loss = ((ac.v(obs) - ret) ** 2).mean()  # MSE loss
    return loss.clamp(0, 1)


def update(data, ac, clip_ratio, train_pi_iters, train_v_iters, pi_optimizer, vf_optimizer, target_kl):
    pi_l_old, pi_info_old = compute_loss_pi_with_entropy(data, ac, clip_ratio)  #
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data, ac).item()
    pi_info = None
    list_epoch_loss_pi = []
    list_epoch_loss_v = []
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi_with_entropy(data, ac, clip_ratio)  #
        list_epoch_loss_pi.append(loss_pi.detach().numpy())
        # kl = mpi_avg(pi_info['kl'])
        kl = pi_info['kl']
        if kl > 1.5 * target_kl:
            # print('Early stopping at step %d due to %f reaching max kl.' % (i, kl))
            break
        loss_pi.backward()  # time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # wandb.log({"7 spup/LossPi": loss_pi})
        # mpi_avg_grads(ac.pi)  # average grads across MPI processes
        pi_optimizer.step()
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(data, ac)
        # wandb.log({"7 spup/LossV": loss_v})
        list_epoch_loss_v.append(loss_v.detach().numpy())
        loss_v.backward()
        vf_optimizer.step()
    wandb.log({"7 spup/LossPi": np.mean(list_epoch_loss_pi)})
    wandb.log({"7 spup/LossV": np.mean(list_epoch_loss_v)})

    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    wandb.log({"7 spup/kl": kl})
    wandb.log({"7 spup/ent": ent})
    wandb.log({"7 spup/cf": cf})


def main():
    local_steps_per_epoch = 500
    max_ep_len = 50
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
        resume="bs500-20241115"  # HjScenarioEnv
    )

    env = HoverAviary(gui=False)

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)

    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    replay_buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim, size=local_steps_per_epoch)  # size=int(1e6)

    ac = MLPActorCritic(env.observation_space, env.action_space)
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

    print("Finished...")


if __name__ == "__main__":
    main()
