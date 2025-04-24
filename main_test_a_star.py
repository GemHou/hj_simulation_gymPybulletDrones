import time
import numpy as np
import pybullet as p
from datetime import datetime

from utils_drone import HjAviary
from main_a_star import a_star_3d

RENDER = False


def vis_point(point, color=None):
    if color is None:
        color = [1, 0, 1, 0.5]

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


def visualize_path(path_points_pos):
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
    start_time = time.time()
    start_index = [int(drone_pos[0] / 0.25 + 128 * 3), int(drone_pos[1] / 0.25 + 128 * 3), int(drone_pos[2] / 0.25)]
    target_index = [int(target_pos[0] / 0.25 + 128 * 3), int(target_pos[1] / 0.25 + 128 * 3),
                    int(target_pos[2] / 0.25)]
    target_index[0] = np.clip(target_index[0], 0, 128 * 4 - 1)
    target_index[1] = np.clip(target_index[1], 0, 128 * 4 - 1)
    target_index[2] = np.clip(target_index[2], 0, 128 - 1)
    start_time = time.time()
    path_index = a_star_3d(tuple(start_index), tuple(target_index), dilated_occ_index)
    # print("search time: ", time.time() - start_time)
    if path_index is not None:
        path_points_pos = []
        for point_index in path_index:
            path_points_pos.append(
                [(point_index[0] - 128 * 3) * 0.25, (point_index[1] - 128 * 3) * 0.25, point_index[2] * 0.25])
    else:
        path_points_pos = None
    search_time = time.time() - start_time
    return path_points_pos, search_time


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


def initialize_environment():
    print("Load env...")
    env = HjAviary(gui=RENDER)
    print("Load a-star...")
    dilated_occ_file_path = "./data/dilated_occ_index.npy"
    dilated_occ_index = np.load(dilated_occ_file_path)
    return env, dilated_occ_index


def reset_environment(env):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(current_time, "reset...")
    save_flag = False
    obs_ma, info = env.reset(percent=1)
    list_drone_pos = []
    list_target_pos = []
    drone_pos, vel_z, vel_x, vel_y, ang_x, ang_my = analyze_obs(obs_ma)
    vel_x_last = vel_x
    vel_y_last = vel_y
    target_pos = [env.target_x, env.target_y, env.target_z]
    list_drone_pos.append(drone_pos)
    list_target_pos.append(target_pos)
    return obs_ma, save_flag, list_drone_pos, list_target_pos, drone_pos, vel_x_last, vel_y_last, target_pos


def perform_path_search(dilated_occ_index, drone_pos, target_pos):
    path_points_pos, search_time = search_a_star_pos(dilated_occ_index, drone_pos, target_pos)
    if search_time > 0.5:
        print("search_time > 0.5")
        return None
    if path_points_pos is None:
        print("start end point problem")
        return None
    visualize_path(path_points_pos)
    if len(path_points_pos) < 4:
        print("len(path_points_pos) < 4")
        return None
    small_target_pos = path_points_pos[3]
    vis_point(small_target_pos, color=[0, 1, 1, 0.5])
    return path_points_pos, small_target_pos


def control_loop(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last, vel_y_last, list_drone_pos, list_target_pos):
    ep_len = 0
    save_flag = False
    while True:
        ep_len += 1
        target_pos = [env.target_x, env.target_y, env.target_z]
        path_points_pos, search_time = search_a_star_pos(dilated_occ_index, drone_pos, target_pos)
        if search_time > 0.5:
            print("search_time > 0.5")
            break
        if path_points_pos is None:
            print("path_points_pos is None")
            break
        if len(path_points_pos) < 4:
            if ep_len > 30 * 5:
                save_flag = True
            print("len(path_points_pos) < 4")
            break
        small_target_pos = path_points_pos[3]
        action_ma, vel_x_last, vel_y_last, drone_pos = calc_pid_control(obs_ma, small_target_pos, vel_x_last,
                                                                        vel_y_last)
        target_pos = [env.target_x, env.target_y, env.target_z]
        list_drone_pos.append(drone_pos)
        list_target_pos.append(target_pos)
        next_obs_ma, reward, done, truncated, info = env.step(action_ma)
        obs_ma = next_obs_ma
        if RENDER:
            env.render()
            time.sleep(1 / 30)
        if done:
            print("done")
            break
    return save_flag, list_drone_pos, list_target_pos


def save_data(save_flag, list_drone_pos, list_target_pos):
    if save_flag:
        drone_pos_array = np.array(list_drone_pos)
        target_pos_array = np.array(list_target_pos)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'./data/data_raw_0_2/data_raw_{current_time}.npz'
        np.savez(file_name, drone_pos_array=drone_pos_array, target_pos_array=target_pos_array)
        print("Data_raw saved as numpy array.")


def main():
    env, dilated_occ_index = initialize_environment()
    print("Looping...")
    while True:
        obs_ma, save_flag, list_drone_pos, list_target_pos, drone_pos, vel_x_last, vel_y_last, target_pos = reset_environment(
            env)
        path_result = perform_path_search(dilated_occ_index, drone_pos, target_pos)
        if path_result is None:
            continue
        path_points_pos, small_target_pos = path_result
        save_flag, list_drone_pos, list_target_pos = control_loop(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last,
                                                                  vel_y_last, list_drone_pos, list_target_pos)
        save_data(save_flag, list_drone_pos, list_target_pos)
    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()
