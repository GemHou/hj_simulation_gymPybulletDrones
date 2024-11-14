import time

import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def main():
    env = HoverAviary(gui=True)

    print("env.CTRL_FREQ: ", env.CTRL_FREQ)
    print("env.ACTION_BUFFER_SIZE: ", env.ACTION_BUFFER_SIZE)
    print("env.action_space: ", env.action_space)

    obs, info = env.reset()

    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))

    for t in range(50):
        # action = np.zeros((1, 4))
        # action = np.ones((1, 4))
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72
        obs_12 = next_obs[:, :12]  # [1, 12]  # [pos 3, rpy 3, vel 3, ang 3]
        pos = obs_12[:, 0:3]
        rpy = obs_12[:, 3:6]
        vel = obs_12[:, 6:9]
        ang = obs_12[:, 9:12]
        reward = pos[0, 2]

        replay_buffer.store(obs, action, reward, next_obs, terminated)

        obs = next_obs

        env.render()
        time.sleep(0.1)

    print("Finished...")


if __name__ == "__main__":
    main()
