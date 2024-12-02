import math

import numpy as np

from gym_pybullet_drones.envs import HoverAviary

TARGET_X = 10.5
TARGET_Y = 0.5
TARGET_Z = 0.5


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
        target_x = TARGET_X
        target_y = TARGET_Y
        target_z = TARGET_Z
        dis_target = math.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2 + (pos_z - target_z)**2)
        reward_target = 1 / (dis_target + 1)
        # reward_target = 5 - (abs(pos_x - target_x) + abs(pos_y - target_y) + abs(pos_z - target_z))
        # reward_target = min(reward_target, 10)
        reward = reward_done + reward_target  # + reward_z + reward_xy
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        target_x = TARGET_X
        target_y = TARGET_Y
        target_z = TARGET_Z
        if pos_z < 0.08:
            done = True
        elif abs(pos_x - target_x) > 20:
            done = True
        elif abs(pos_y - target_y) > 5:
            done = True
        elif abs(pos_z - target_z) > 5:
            done = True
        else:
            done = False

        return done


class HjAviaryActionAng(HjAviary):
    def step(self,
             action_ma_aug
             ):
        # analyse action
        goal_ang_x = action_ma_aug[0, 0] * 0.02
        goal_ang_my = action_ma_aug[0, 1] * 0.02
        goal_vel_z = action_ma_aug[0, 2]
        # if goal_vel_z < 0:
        #     goal_vel_z = 0
        # get obs
        state = self._getDroneStateVector(0)  # [pos 3, nth 4, rpy 3, vel 3, ang 3, last_clipped_action 4]
        ang = state[13:16]
        ang_my = ang[0]
        ang_x = ang[1]
        vel = state[10:13]
        vel_z = vel[2]
        # PID controller
        # action_vel
        vel_z_bias = vel_z - goal_vel_z
        action_vel_z = vel_z_bias * -20
        # if action_vel_z > 1:
        #
        # action_ang
        action_ang_x = (goal_ang_x - ang_x) * 0.2
        action_ang_my = (goal_ang_my - ang_my) * 0.2  # 0.01
        action_ang_z = 0  # 0.01
        # final
        # wandb.log({"action/action_vel_z": action_vel_z})
        # wandb.log({"action/action_ang_x": action_ang_x})
        # wandb.log({"action/action_ang_my": action_ang_my})
        # wandb.log({"action/action_ang_z": action_ang_z})
        action_motor = [action_vel_z - action_ang_my - action_ang_x - action_ang_z,
                        action_vel_z - action_ang_my + action_ang_x + action_ang_z,
                        action_vel_z + action_ang_my + action_ang_x - action_ang_z,
                        action_vel_z + action_ang_my - action_ang_x + action_ang_z]
        action_ma_motor = np.array([action_motor])
        obs, reward, terminated, truncated, info = super().step(action_ma_motor)
        return obs, reward, terminated, truncated, info
