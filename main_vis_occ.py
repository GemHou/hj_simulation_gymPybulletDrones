import numpy as np
import open3d as o3d
import glob
from tqdm import tqdm


def transfer_points(dilated_occ_index):
    # 假设 occ_index 是一个三维布尔数组
    # 获取满足条件的索引
    indices = np.argwhere(dilated_occ_index == 1)
    # 转换坐标
    points = np.zeros((indices.shape[0], 3))
    points[:, 0] = (indices[:, 0] - 128 * 3) * 0.25
    points[:, 1] = (indices[:, 1] - 128 * 3) * 0.25
    points[:, 2] = indices[:, 2] * 0.25
    return points


def load_data():
    dilated_occ_file_path = "./data/dilated_occ_index.npy"
    dilated_occ_index = np.load(dilated_occ_file_path)

    # 获取所有轨迹文件路径
    traj_file_paths = glob.glob("./data/data_raw_0_2/data_raw_*.npz")
    drone_trajs = [np.load(traj_file_path)["drone_pos_array"] for traj_file_path in traj_file_paths]
    target_trajs = [np.load(traj_file_path)["target_pos_array"] for traj_file_path in traj_file_paths]

    return dilated_occ_index, drone_trajs, target_trajs


def main():
    dilated_occ_index, drone_trajs, target_trajs = load_data()

    print("Loaded trajectories:")
    for i, (drone_traj, target_traj) in enumerate(zip(drone_trajs, target_trajs)):
        print(f"Trajectory {i + 1}: Drone {drone_traj.shape}, Target {target_traj.shape}")

    points = transfer_points(dilated_occ_index)

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 创建轨迹线
    line_sets = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]  # 预定义颜色

    for i, (drone_traj, target_traj) in enumerate(zip(drone_trajs, target_trajs)):
        # 创建无人机轨迹线
        drone_line_set = o3d.geometry.LineSet()
        drone_line_set.points = o3d.utility.Vector3dVector(drone_traj)
        drone_line_indices = np.arange(len(drone_traj) - 1).reshape(-1, 1)
        drone_line_indices = np.hstack((drone_line_indices, drone_line_indices + 1))
        drone_line_set.lines = o3d.utility.Vector2iVector(drone_line_indices)
        drone_line_set.colors = o3d.utility.Vector3dVector([colors[i % len(colors)] for _ in range(len(drone_traj) - 1)])
        line_sets.append(drone_line_set)

        # 创建目标轨迹线
        target_line_set = o3d.geometry.LineSet()
        target_line_set.points = o3d.utility.Vector3dVector(target_traj)
        target_line_indices = np.arange(len(target_traj) - 1).reshape(-1, 1)
        target_line_indices = np.hstack((target_line_indices, target_line_indices + 1))
        target_line_set.lines = o3d.utility.Vector2iVector(target_line_indices)
        target_line_set.colors = o3d.utility.Vector3dVector([colors[i % len(colors)] for _ in range(len(target_traj) - 1)])
        line_sets.append(target_line_set)

    # 可视化点云和轨迹
    o3d.visualization.draw_geometries(
        [pcd] + line_sets,
        window_name="3D Occupancy and Trajectories Visualization",
        width=800,
        height=600
    )

    print("Finished...")


if __name__ == "__main__":
    main()