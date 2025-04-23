import time
import numpy as np

from utils_drone import HjAviary
from main_a_star import a_star_3d


def analyse_obs(obs_ma):
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
    return pos


def search_a_star_pos(dilated_occ_index, drone_pos, target_pos):
    start_index = [int(drone_pos[0] / 0.25 + 128 * 3), int(drone_pos[1] / 0.25 + 128 * 3), int(drone_pos[2] / 0.25)]
    target_index = [int(target_pos[0] / 0.25 + 128 * 3), int(target_pos[1] / 0.25 + 128 * 3),
                    int(target_pos[2] / 0.25)]
    start_time = time.time()
    path_index = a_star_3d(tuple(start_index), tuple(target_index), dilated_occ_index)
    print("search time: ", time.time() - start_time)
    path_points_pos = []
    for point_index in path_index:
        path_points_pos.append(
            [(point_index[0] - 128 * 3) * 0.25, (point_index[1] - 128 * 3) * 0.25, point_index[2] * 0.25])
    return path_points_pos


def main():
    print("Load env...")
    env = HjAviary(gui=True)  # , ctrl_freq=10, pyb_freq=100

    print("Load a-star...")
    dilated_occ_file_path = "./data/dilated_occ_index.npy"
    dilated_occ_index = np.load(dilated_occ_file_path)

    print("Looping...")
    if True:
        obs_ma, info = env.reset(percent=1)

        drone_pos = analyse_obs(obs_ma)
        target_pos = [env.target_x, env.target_y, env.target_z]

        path_points_pos = search_a_star_pos(dilated_occ_index, drone_pos, target_pos)

        print("path_points_pos: ", path_points_pos)

        if True:
            env.render()
            time.sleep(1 / 30)  # * 10

    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()