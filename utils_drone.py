from gym_pybullet_drones.envs import HoverAviary


class HjAviary(HoverAviary):
    def _computeReward(self):
        state = self._getDroneStateVector(0)  # pos 3 quat 4 ...
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3]) ** 4)
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        if pos_z < 0.05:
            reward_done = -10
        else:
            reward_done = 0
        reward_z = pos_z
        reward_xy = - abs(pos_x) - abs(pos_y)
        reward = reward_done + reward_z + reward_xy
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[:3]
        pos_z = pos[2]
        if pos_z < 0.05:
            done = True
        else:
            done = False
        return done
