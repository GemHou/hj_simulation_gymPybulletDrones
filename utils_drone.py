from gym_pybullet_drones.envs import HoverAviary


class HjAviary(HoverAviary):
    def _computeReward(self):
        state = self._getDroneStateVector(0)  # pos 3 quat 4 ...
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3]) ** 4)
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        if pos_z < 0.08:
            reward_done = -10
        else:
            reward_done = 0
        # reward_z = pos_z
        # reward_xy = - abs(pos_x) - abs(pos_y)
        target_x = 0.5
        target_y = 0.5
        target_z = 0.5
        reward_target = 5 - ((pos_x - target_x) ** 2 + (pos_y - target_y) ** 2 + (pos_z - target_z) ** 2) * 5
        # reward_target = min(reward_target, 10)
        reward = reward_done + reward_target  # + reward_z + reward_xy
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        if pos_z < 0.08:
            done = True
        else:
            done = False

        return done


class HjAviaryActionAng(HjAviary):
    def step(self,
             action
             ):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
