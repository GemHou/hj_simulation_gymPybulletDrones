import math
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs import HoverAviary


class HjAviary(HoverAviary):
    def calc_done(self, dis_target, pitch, pos_z, roll):
        if pos_z < 0.2:
            done = True
        elif dis_target > 50 * 1.732:
            done = True
        elif abs(roll) > 60 or abs(pitch) > 60:
            done = True
        else:
            done = False
        return done

    def _computeReward(self):
        state = self._getDroneStateVector(0)  # state = np.hstack([pos, nth, rpy, vel, ang, last_clipped_action])  # 3 4 3 3 3 4
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3]) ** 4)
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        ang_v = state[13:16]
        roll = state[7] * 180 / math.pi
        pitch = state[8] * 180 / math.pi
        dis_target = math.sqrt((pos_x - self.target_x) ** 2 + (pos_y - self.target_y) ** 2 + (pos_z - self.target_z) ** 2)

        done = self.calc_done(dis_target, pitch, pos_z, roll)
        if done:
            reward_done = -10
        else:
            reward_done = 0

        target_x = self.target_x
        target_y = self.target_y
        target_z = self.target_z
        dis_target = math.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2 + (pos_z - target_z)**2)
        reward_target = 1 / (dis_target / 1 + 1)
        # reward_target = 5 - (abs(pos_x - target_x) + abs(pos_y - target_y) + abs(pos_z - target_z))
        # reward_target = min(reward_target, 10)

        reward_ang_v = np.clip(1 / (abs(ang_v[2]) / 1 + 1), 0, 1)

        reward = reward_done + reward_target * 0.7 + reward_ang_v * 0.3  # + reward_z + reward_xy

        reward = np.clip(reward, -1, 2)

        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)  # state = np.hstack([pos, nth, rpy, vel, ang, last_clipped_action])  # 3 4 3 3 3 4
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        roll = state[7] * 180 / math.pi
        pitch = state[8] * 180 / math.pi
        dis_target = math.sqrt((pos_x - self.target_x) ** 2 + (pos_y - self.target_y) ** 2 + (pos_z - self.target_z) ** 2)

        done = self.calc_done(dis_target, pitch, pos_z, roll)

        return done


class HjAviaryActionAng(HjAviary):
    # def _actionSpace(self):
    #     """Returns the action space of the environment.
    #
    #     Returns
    #     -------
    #     spaces.Box
    #         An ndarray of shape (NUM_DRONES, 4) for the commanded velocity vectors.
    #
    #     """
    #     #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
    #     act_lower_bound = np.array([[-1,     -1,     -1] for i in range(self.NUM_DRONES)])
    #     act_upper_bound = np.array([[ 1,      1,      1] for i in range(self.NUM_DRONES)])
    #     return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def step(self,
             action_ma_ang_norm_rl
             ):
        # analyse action
        goal_ang_x = action_ma_ang_norm_rl[0, 0] / 20
        goal_ang_my = action_ma_ang_norm_rl[0, 1] / 20
        goal_vel_z = action_ma_ang_norm_rl[0, 2]
        goal_ang_z = 0  # -0.02 0 0.02
        # if goal_vel_z < 0:
        #     goal_vel_z = 0
        # get obs
        state = self._getDroneStateVector(0)  # [pos 3, nth 4, rpy 3, vel 3, ang 3, last_clipped_action 4]
        ang = state[13:16]
        ang_my = ang[0]
        ang_x = ang[1]
        ang_z = ang[2]
        if abs(ang_z) > 0.02:
            goal_ang_x = 0
            goal_ang_my = 0
        vel = state[10:13]
        vel_z = vel[2]
        # PID controller
        # action_vel
        vel_z_bias = vel_z - goal_vel_z
        action_vel_z = vel_z_bias * -20
        # if action_vel_z > 1:
        #
        # action_ang
        action_ang_x = (goal_ang_x - ang_x) * 0.05
        action_ang_my = (goal_ang_my - ang_my) * 0.05  # 0.01
        action_ang_z = (goal_ang_z - ang_z) * 0.2  # 0.01
        # final
        # wandb.log({"action/action_vel_z": action_vel_z})
        # wandb.log({"action/action_ang_x": action_ang_x})
        # wandb.log({"action/action_ang_my": action_ang_my})
        # wandb.log({"action/action_ang_z": action_ang_z})
        # if abs(ang_z) < 0.02:
        action_motor = [action_vel_z - action_ang_my - action_ang_x - action_ang_z,
                        action_vel_z - action_ang_my + action_ang_x + action_ang_z,
                        action_vel_z + action_ang_my + action_ang_x - action_ang_z,
                        action_vel_z + action_ang_my - action_ang_x + action_ang_z]
        # else:
        #     action_motor = [action_vel_z - action_ang_z,
        #                     action_vel_z + action_ang_z,
        #                     action_vel_z - action_ang_z,
        #                     action_vel_z + action_ang_z]
        action_ma_motor = np.array([action_motor])
        action_ma_motor = np.clip(action_ma_motor, a_min=-10, a_max=10)
        obs, reward, terminated, truncated, info = super().step(action_ma_motor)
        return obs, reward, terminated, truncated, info


class HjAviaryActionAngRes(HjAviaryActionAng):
    def step(self,
             action_ma_ang_norm_rl
             ):
        goal_pos_z = 3
        goal_pos_x = 0
        goal_pos_y = 0

        state = self._getDroneStateVector(0)
        ang = state[13:16]
        ang_my = ang[0]
        ang_x = ang[1]
        ang_z = ang[2]
        pos = state[:3]
        pos_x = pos[0]
        pos_y = pos[1]
        pos_z = pos[2]
        vel = state[10:13]
        vel_x = vel[0]
        vel_y = vel[1]
        vel_z = vel[2]

        # goal_pos
        # goal_pos_z
        # wandb.log({"z/goal_pos_z": goal_pos_z})
        # wandb.log({"x/goal_pos_x": goal_pos_x})
        # wandb.log({"y/goal_pos_y": goal_pos_y})
        # goal_vel
        goal_vel_z = (goal_pos_z - pos_z) * 0.5
        # goal_vel_x
        goal_vel_x = (goal_pos_x - pos_x) * 0.2 - vel_x * 0.1
        # goal_vel_y
        goal_vel_y = (goal_pos_y - pos_y) * 0.2 - vel_y * 0.1
        # wandb.log({"z/goal_vel_z": goal_vel_z})
        # wandb.log({"x/goal_vel_x": goal_vel_x})
        goal_vel_x = np.clip(goal_vel_x, -2, 2)
        # wandb.log({"y/goal_vel_y": goal_vel_y})
        goal_vel_y = np.clip(goal_vel_y, -2, 2)
        # goal_ang
        goal_ang_x = (goal_vel_x - vel_x) * 0.02  # 0.02~0.05
        goal_ang_x = np.clip(goal_ang_x, -0.05, 0.05)
        goal_ang_my = (goal_vel_y - vel_y) * -0.02
        goal_ang_my = np.clip(goal_ang_my, -0.05, 0.05)
        # wandb.log({"x/goal_ang_x": goal_ang_x})
        # wandb.log({"y/goal_ang_my": goal_ang_my})
        action_ang_norm_pid = [goal_ang_x * 20, goal_ang_my * 20, goal_vel_z, 0]
        action_ma_ang_pid_norm = np.array([action_ang_norm_pid])

        # action_ma_ang[0, 0] = action_ma_ang[0, 0] + action_ang_pid[0]
        # action_ma_ang[0, 1] = action_ma_ang[0, 1] + action_ang_pid[1]
        # action_ma_ang[0, 2] = action_ma_ang[0, 2] + action_ang_pid[2]
        action_ma_ang_norm = action_ma_ang_norm_rl + action_ma_ang_pid_norm

        obs, reward, terminated, truncated, info = super().step(action_ma_ang_norm)
        return obs, reward, terminated, truncated, info
