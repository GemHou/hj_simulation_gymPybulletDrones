import time

import gymnasium
import numpy as np
import scipy
import torch
import wandb
from gym.spaces import Box, Discrete
from torch import nn as nn
from torch.distributions import Normal, Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        if isinstance(obs_dim, str):
            # self.obs_buf = [None] * size
            self.obs_buf = dict()
            self.obs_buf["real_ego"] = np.zeros(combined_shape(size, 29), dtype=np.float32)
            self.obs_buf["real_critic"] = np.zeros(combined_shape(size, 7), dtype=np.float32)
            self.obs_buf["real_static"] = np.zeros((size, 4, 16, 6), dtype=np.float32)
            self.obs_buf["real_other"] = np.zeros((size, 30, 1, 11), dtype=np.float32)
        else:
            self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        if isinstance(self.obs_buf, dict):
            self.obs_buf["real_ego"][self.ptr] = obs["real_ego"]
            self.obs_buf["real_critic"][self.ptr] = obs["real_critic"]
            self.obs_buf["real_static"][self.ptr] = obs["real_static"]
            self.obs_buf["real_other"][self.ptr] = obs["real_other"]
        else:
            self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, device, tensor_flag=True):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick

        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)

        x = np.array(self.adv_buf, dtype=np.float32)
        global_sum, global_n = [np.sum(x), len(x)]
        mean = global_sum / global_n
        global_sum_sq = np.sum((x - mean) ** 2)
        std = np.sqrt(global_sum_sq / global_n)  # compute global std
        adv_mean = mean
        adv_std = std

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        if False:
            if tensor_flag:
                return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}
            else:
                return {k: v for k, v in data.items()}
        else:
            result = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    result[k] = dict()
                    if tensor_flag:
                        result[k]["real_ego"] = torch.as_tensor(v["real_ego"], dtype=torch.float32, device=device)
                        result[k]["real_critic"] = torch.as_tensor(v["real_critic"], dtype=torch.float32, device=device)
                        result[k]["real_static"] = torch.as_tensor(v["real_static"], dtype=torch.float32, device=device)
                        result[k]["real_other"] = torch.as_tensor(v["real_other"], dtype=torch.float32, device=device)
                    else:
                        result[k]["real_ego"] = v["real_ego"]
                        result[k]["real_critic"] = v["real_critic"]
                        result[k]["real_static"] = v["real_static"]
                        result[k]["real_other"] = v["real_other"]

                else:
                    if tensor_flag:
                        tensor = torch.as_tensor(v, dtype=torch.float32, device=device)
                        result[k] = tensor
                    else:
                        result[k] = v

            return result


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[1]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[1], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        elif isinstance(action_space, gymnasium.spaces.box.Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[1], hidden_sizes, activation)
        else:
            print("type(action_space): ", type(action_space))
            raise
        # self.pi.to(DEVICE)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        # self.v.to(DEVICE)

    def trans_tensor_to_numpy(self, a, logp_a, v):
        a_numpy = a.cpu().numpy()
        v_numpy = v.cpu().numpy()
        logp_a_numpy = logp_a.cpu().numpy()
        return a_numpy, logp_a_numpy, v_numpy

    def calc_logp_v(self, a, obs, pi):
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        return logp_a, v

    def step(self, obs):
        with torch.no_grad():
            if isinstance(obs, dict):
                pass
            else:
                pass
                # if obs.device == "cpu":
                #     # raise
                #     print("Warning!!! ")
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a, v = self.calc_logp_v(a, obs, pi)
        a_numpy, logp_a_numpy, v_numpy = self.trans_tensor_to_numpy(a, logp_a, v)
        return a_numpy, v_numpy, logp_a_numpy

    def act(self, obs):
        return self.step(obs)[0]


def collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer, list_ep_ret, percent):
    obs_ma, info = env.reset(percent)
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
            list_ep_ret.append(ep_ret)
            list_epoch_ep_ret.append(ep_ret)
            list_epoch_ep_len.append(ep_len)
            list_epoch_rps.append(ep_ret / ep_len)
            reset_time_start = time.time()
            obs_ma, info = env.reset(percent)
            reset_time = time.time() - reset_time_start
            list_epoch_reset_time.append(reset_time)
            ep_ret, ep_len = 0, 0
    print("np.mean(list_epoch_ep_ret): ", np.mean(list_epoch_ep_ret))
    wandb.log({"5 performance/episode return": np.mean(list_epoch_ep_ret)})
    wandb.log({"5 performance/episode length": np.mean(list_epoch_ep_len)})
    wandb.log({"5 performance/return per step": np.mean(list_epoch_rps)})
    wandb.log({"8 throughout/AverageResetTime": np.mean(list_epoch_reset_time)})


def compute_loss_pi_with_entropy(data, ac, clip_ratio,
                                 ent_coeff=0.000):  # You can adjust the value of ent_coeff ent_coeff= 0.001~0.02~0.1
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


def compute_loss_v(data, ac):
    obs, ret = data['obs'], data['ret']
    loss = ((ac.v(obs) - ret) ** 2).mean()  # MSE loss
    # loss = loss.clamp(0, 500000)
    return loss  # .clamp(0, 20)


def update(data, ac, clip_ratio, train_pi_iters, train_v_iters, pi_optimizer, vf_optimizer, target_kl):
    start_time = time.time()
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
    update_time = time.time()-start_time
    return update_time
