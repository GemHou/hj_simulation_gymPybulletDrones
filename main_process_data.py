import glob
import numpy as np
from tqdm import tqdm
from datetime import datetime


def main():
    print("Loading...")
    # 获取所有轨迹文件路径
    traj_file_paths = glob.glob("./data/data_raw_0_2/data_raw_*.npz")
    traj_file_paths.sort()

    # print("debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # traj_file_paths = traj_file_paths[:1000]

    drone_trajs = []
    target_trajs = []

    for traj_file_path in tqdm(traj_file_paths):
        # print("traj_file_path: ", traj_file_path)
        drone_trajs.append(np.load(traj_file_path)["drone_pos_array"])
        target_trajs.append(np.load(traj_file_path)["target_pos_array"])

    print("Processing...")
    list_drone_pos = []
    list_target_pos = []
    list_future_traj = []
    for traj_i in tqdm(range(len(drone_trajs))):
        for point_j in range(len(drone_trajs[traj_i]) - 29):
            list_drone_pos.append(drone_trajs[traj_i][point_j])
            list_target_pos.append(target_trajs[traj_i][point_j])
            list_future_traj.append(drone_trajs[traj_i][point_j:point_j + 30])

    print("Saving...")
    array_drone_pos = np.array(list_drone_pos)
    array_target_pos = np.array(list_target_pos)
    array_future_traj = np.array(list_future_traj)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'./data/data_processed_0_1/data_processed_{current_time}.npz'
    np.savez(file_name,
             array_drone_pos=array_drone_pos,
             array_target_pos=array_target_pos,
             array_future_traj=array_future_traj)
    print("Data_raw saved as numpy array.")

    print("Finished...")


if __name__ == "__main__":
    main()
