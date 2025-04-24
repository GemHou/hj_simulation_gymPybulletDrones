import time
import numpy as np
from datetime import datetime

from utils_drone import HjAviary
from main_a_star import a_star_3d
from utils_pid import analyze_obs, calc_pid_control
from utils_vis import vis_point, visualize_path

RENDER_PYBULLET = False


def search_a_star_pos(dilated_occ_index, drone_pos, target_pos):
    start_index = [int(drone_pos[0] / 0.25 + 128 * 3), int(drone_pos[1] / 0.25 + 128 * 3), int(drone_pos[2] / 0.25)]
    target_index = [int(target_pos[0] / 0.25 + 128 * 3), int(target_pos[1] / 0.25 + 128 * 3),
                    int(target_pos[2] / 0.25)]
    target_index[0] = np.clip(target_index[0], 0, 128 * 4 - 1)
    target_index[1] = np.clip(target_index[1], 0, 128 * 4 - 1)
    target_index[2] = np.clip(target_index[2], 0, 128 - 1)
    start_time = time.time()
    path_index = a_star_3d(tuple(start_index), tuple(target_index), dilated_occ_index)
    if path_index is not None:
        path_points_pos = []
        for point_index in path_index:
            path_points_pos.append(
                [(point_index[0] - 128 * 3) * 0.25, (point_index[1] - 128 * 3) * 0.25, point_index[2] * 0.25])
    else:
        path_points_pos = None
    search_time = time.time() - start_time
    return path_points_pos, search_time


def initialize_environment(render_pybullet):
    print("Load env...")
    env = HjAviary(gui=render_pybullet)
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


def control_loop_astar(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last, vel_y_last, list_drone_pos, list_target_pos,
                 render_pybullet):
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
        if render_pybullet:
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
    env, dilated_occ_index = initialize_environment(RENDER_PYBULLET)
    print("Looping...")
    while True:
        obs_ma, save_flag, list_drone_pos, list_target_pos, drone_pos, vel_x_last, vel_y_last, target_pos = reset_environment(
            env)
        path_result = perform_path_search(dilated_occ_index, drone_pos, target_pos)
        if path_result is None:
            continue
        save_flag, list_drone_pos, list_target_pos = control_loop_astar(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last,
                                                                  vel_y_last, list_drone_pos, list_target_pos,
                                                                  RENDER_PYBULLET)
        save_data(save_flag, list_drone_pos, list_target_pos)
    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()
