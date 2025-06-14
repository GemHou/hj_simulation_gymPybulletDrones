import numpy as np


def analyze_obs(obs_ma):
    obs_12 = obs_ma[:, :12]
    pos = obs_12[0, 0:3]
    rpy = obs_12[0, 3:6]
    vel = obs_12[0, 6:9]
    ang = obs_12[0, 9:12]
    ang_my = ang[0]
    ang_x = ang[1]
    ang_z = ang[2]
    vel_x = vel[0]
    vel_y = vel[1]
    vel_z = vel[2]
    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]
    return pos, vel_z, vel_x, vel_y, ang_x, ang_my


def calc_pid_vel(drone_pos, small_target_pos):
    # z
    pos_z = drone_pos[2]
    goal_pos_z = small_target_pos[2]
    goal_vel_z = (goal_pos_z - pos_z) * 0.5

    # x
    pos_x = drone_pos[0]
    goal_pos_x = small_target_pos[0]
    if goal_pos_x - pos_x > 0.5:
        goal_vel_x = 1
    elif goal_pos_x - pos_x < -0.5:
        goal_vel_x = -1
    else:
        goal_vel_x = 0
    # goal_vel_x = 0.5

    # y
    pos_y = drone_pos[1]
    goal_pos_y = small_target_pos[1]
    if goal_pos_y - pos_y > 0.5:
        goal_vel_y = 1
    elif goal_pos_y - pos_y < -0.5:
        goal_vel_y = -1
    else:
        goal_vel_y = 0
    # goal_vel_y = 0.5
    return goal_vel_x, goal_vel_y, goal_vel_z


def calc_pid_ang(ang_my, ang_x, goal_vel_x, goal_vel_y, goal_vel_z, vel_x, vel_y, vel_z, acc_x, acc_y):
    vel_z_bias = vel_z - goal_vel_z
    action_vel_z = vel_z_bias * -10
    # print("vel_x: ", vel_x)
    goal_ang_x = (goal_vel_x - vel_x) * 0.02 - acc_x * 1  # 0.02~0.05
    goal_ang_my = (goal_vel_y - vel_y) * -0.02 + acc_y * 1
    action_ang_x = (goal_ang_x - ang_x) * 0.2
    action_ang_my = (goal_ang_my - ang_my) * 0.2  # 0.01
    action_ang_z = 0  # 0.01
    return action_ang_my, action_ang_x, action_ang_z, action_vel_z


def calc_pid_control(obs_ma, small_target_pos, vel_x_last, vel_y_last):
    drone_pos, vel_z, vel_x, vel_y, ang_x, ang_my = analyze_obs(obs_ma)
    acc_x = vel_x - vel_x_last
    acc_y = vel_y - vel_y_last
    vel_x_last = vel_x
    vel_y_last = vel_y
    goal_vel_x, goal_vel_y, goal_vel_z = calc_pid_vel(drone_pos, small_target_pos)
    action_ang_my, action_ang_x, action_ang_z, action_vel_z = calc_pid_ang(ang_my, ang_x, goal_vel_x, goal_vel_y,
                                                                           goal_vel_z, vel_x, vel_y, vel_z, acc_x,
                                                                           acc_y)
    action = [action_vel_z - action_ang_my - action_ang_x - action_ang_z,
              action_vel_z - action_ang_my + action_ang_x + action_ang_z,
              action_vel_z + action_ang_my + action_ang_x - action_ang_z,
              action_vel_z + action_ang_my - action_ang_x + action_ang_z]
    action_ma = np.array([action])
    return action_ma, vel_x_last, vel_y_last, drone_pos
