import time
import numpy as np
import pybullet as p

from utils_drone import HjAviary
from main_a_star import a_star_3d


def vis_point(point, color=None):
    if color is None:
        color = [1, 0, 0, 0.5]

    # 创建可视化描述符（球体）
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.1,
        rgbaColor=color
    )
    # 创建多体对象
    p.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=point,
        baseOrientation=[0, 0, 0, 1],
        baseMass=0
    )


def visualize_path(path_points_pos, env):
    for point in path_points_pos:
        vis_point(point)


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


def search_a_star_pos(dilated_occ_index, drone_pos, target_pos):
    start_index = [int(drone_pos[0] / 0.25 + 128 * 3), int(drone_pos[1] / 0.25 + 128 * 3), int(drone_pos[2] / 0.25)]
    target_index = [int(target_pos[0] / 0.25 + 128 * 3), int(target_pos[1] / 0.25 + 128 * 3),
                    int(target_pos[2] / 0.25)]
    start_time = time.time()
    path_index = a_star_3d(tuple(start_index), tuple(target_index), dilated_occ_index)
    print("search time: ", time.time() - start_time)
    if path_index is not None:
        path_points_pos = []
        for point_index in path_index:
            path_points_pos.append(
                [(point_index[0] - 128 * 3) * 0.25, (point_index[1] - 128 * 3) * 0.25, point_index[2] * 0.25])
    else:
        raise
    return path_points_pos


def calc_pid_vel(drone_pos, small_target_pos):
    # z
    pos_z = drone_pos[2]
    goal_pos_z = small_target_pos[2]
    goal_vel_z = (goal_pos_z - pos_z) * 0.5

    # x
    pos_x = drone_pos[0]
    goal_pos_x = small_target_pos[0]
    # if goal_pos_x - pos_x > 0.5:
    #     goal_vel_x = 0.2
    # elif goal_pos_x - pos_x < -0.5:
    #     goal_vel_x = -0.2
    # else:
    #     goal_vel_x = 0
    goal_vel_x = 0.5

    # y
    pos_y = drone_pos[1]
    goal_pos_y = small_target_pos[1]
    # if goal_pos_y - pos_y > 0.5:
    #     goal_vel_y = 0.2
    # elif goal_pos_y - pos_y < -0.5:
    #     goal_vel_y = -0.2
    # else:
    #     goal_vel_y = 0
    goal_vel_y = 0.5
    return goal_vel_x, goal_vel_y, goal_vel_z


def calc_pid_ang(ang_my, ang_x, goal_vel_x, goal_vel_y, goal_vel_z, vel_x, vel_y, vel_z, acc_x, acc_y):
    vel_z_bias = vel_z - goal_vel_z
    action_vel_z = vel_z_bias * -10
    print("vel_x: ", vel_x)
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
                                                                           goal_vel_z, vel_x, vel_y, vel_z, acc_x, acc_y)
    action = [action_vel_z - action_ang_my - action_ang_x - action_ang_z,
              action_vel_z - action_ang_my + action_ang_x + action_ang_z,
              action_vel_z + action_ang_my + action_ang_x - action_ang_z,
              action_vel_z + action_ang_my - action_ang_x + action_ang_z]
    action_ma = np.array([action])
    return action_ma, vel_x_last, vel_y_last


def main():
    print("Load env...")
    env = HjAviary(gui=True)  # , ctrl_freq=10, pyb_freq=100

    print("Load a-star...")
    dilated_occ_file_path = "./data/dilated_occ_index.npy"
    dilated_occ_index = np.load(dilated_occ_file_path)

    print("Looping...")
    if True:
        obs_ma, info = env.reset(percent=1)

        drone_pos, vel_z, vel_x, vel_y, ang_x, ang_my = analyze_obs(obs_ma)
        vel_x_last = vel_x
        vel_y_last = vel_y
        target_pos = [env.target_x, env.target_y, env.target_z]

        path_points_pos = search_a_star_pos(dilated_occ_index, drone_pos, target_pos)

        visualize_path(path_points_pos, env)  # 调用可视化函数

        small_target_pos = path_points_pos[3]

        vis_point(small_target_pos, color=[0, 1, 1, 0.5])

        while True:
            # path_points_pos = search_a_star_pos(dilated_occ_index, drone_pos, target_pos)
            # small_target_pos = path_points_pos[3]

            action_ma, vel_x_last, vel_y_last = calc_pid_control(obs_ma, small_target_pos, vel_x_last, vel_y_last)

            next_obs_ma, reward, done, truncated, info = env.step(
                action_ma)  # obs [1, 72] 12 + ACTION_BUFFER_SIZE * 4 = 72

            # SAVE next_obs_ma data

            obs_ma = next_obs_ma

            env.render()
            time.sleep(1 / 30)  # * 10

            if done:
                break

    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()
