import time
import torch
import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from utils_rl import PPOBuffer, MLPActorCritic


def collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer):
    obs_ma, info = env.reset()
    ep_ret, ep_len = 0, 0
    for t in range(local_steps_per_epoch):
        obs_tensor = torch.tensor(obs_ma[0], dtype=torch.float32)
        action, v, logp = ac.step(obs_tensor)

        action_ma = np.array([action])
        next_obs_ma, reward, done, truncated, info = env.step(action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72

        ep_len += 1

        obs_12 = next_obs_ma[:, :12]  # [1, 12]  # [pos 3, rpy 3, vel 3, ang 3]
        pos = obs_12[:, 0:3]
        rpy = obs_12[:, 3:6]
        vel = obs_12[:, 6:9]
        ang = obs_12[:, 9:12]
        reward = pos[0, 2]

        replay_buffer.store(obs_tensor, action, reward, v, logp)

        obs_ma = next_obs_ma

        env.render()
        time.sleep(0.1)

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
            obs_ma, info = env.reset()
            ep_ret, ep_len = 0, 0


def main():
    env = HoverAviary(gui=True)

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)

    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    replay_buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(500))  # size=int(1e6)

    ac = MLPActorCritic(env.observation_space, env.action_space)

    local_steps_per_epoch = 200
    max_ep_len = 50

    collect_experience_once(ac, env, local_steps_per_epoch, max_ep_len, replay_buffer)

    print("Finished...")


if __name__ == "__main__":
    main()
