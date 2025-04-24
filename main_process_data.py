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

    # 按7:2:1比例切分数据集
    num_samples = len(array_drone_pos)
    train_size = int(num_samples * 0.7)
    val_size = int(num_samples * 0.2)
    test_size = num_samples - train_size - val_size

    train_drone_pos = array_drone_pos[:train_size]
    train_target_pos = array_target_pos[:train_size]
    train_future_traj = array_future_traj[:train_size]

    val_drone_pos = array_drone_pos[train_size:train_size + val_size]
    val_target_pos = array_target_pos[train_size:train_size + val_size]
    val_future_traj = array_future_traj[train_size:train_size + val_size]

    test_drone_pos = array_drone_pos[train_size + val_size:]
    test_target_pos = array_target_pos[train_size + val_size:]
    test_future_traj = array_future_traj[train_size + val_size:]

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_file_name = f'./data/data_processed_0_1/train_data_processed_{current_time}.npz'
    val_file_name = f'./data/data_processed_0_1/val_data_processed_{current_time}.npz'
    test_file_name = f'./data/data_processed_0_1/test_data_processed_{current_time}.npz'

    np.savez(train_file_name,
             array_drone_pos=train_drone_pos,
             array_target_pos=train_target_pos,
             array_future_traj=train_future_traj)
    np.savez(val_file_name,
             array_drone_pos=val_drone_pos,
             array_target_pos=val_target_pos,
             array_future_traj=val_future_traj)
    np.savez(test_file_name,
             array_drone_pos=test_drone_pos,
             array_target_pos=test_target_pos,
             array_future_traj=test_future_traj)

    print("Data split and saved as numpy arrays.")

    print("Finished...")


if __name__ == "__main__":
    main()
