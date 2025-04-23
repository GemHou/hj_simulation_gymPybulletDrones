import numpy as np
import open3d as o3d
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
    traj_file_path_1 = "./data/drone_positions_20250423_175041.npy"
    traj_1 = np.load(traj_file_path_1)
    traj_file_path_2 = "./data/drone_positions_20250423_175306.npy"
    traj_2 = np.load(traj_file_path_2)
    return dilated_occ_index, traj_1, traj_2


def main():
    dilated_occ_index, traj_1, traj_2 = load_data()

    print("traj_1: ", traj_1)
    print("traj_2: ", traj_2)

    points = transfer_points(dilated_occ_index)

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 创建轨迹线
    line_set_1 = o3d.geometry.LineSet()
    line_set_1.points = o3d.utility.Vector3dVector(traj_1)
    line_indices_1 = np.arange(len(traj_1) - 1).reshape(-1, 1)
    line_indices_1 = np.hstack((line_indices_1, line_indices_1 + 1))
    line_set_1.lines = o3d.utility.Vector2iVector(line_indices_1)
    line_set_1.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(traj_1) - 1)])  # 红色

    line_set_2 = o3d.geometry.LineSet()
    line_set_2.points = o3d.utility.Vector3dVector(traj_2)
    line_indices_2 = np.arange(len(traj_2) - 1).reshape(-1, 1)
    line_indices_2 = np.hstack((line_indices_2, line_indices_2 + 1))
    line_set_2.lines = o3d.utility.Vector2iVector(line_indices_2)
    line_set_2.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(traj_2) - 1)])  # 绿色

    # 可视化点云和轨迹
    o3d.visualization.draw_geometries(
        [pcd, line_set_1, line_set_2],
        window_name="3D Occupancy and Trajectories Visualization",
        width=800,
        height=600
    )

    print("Finished...")


if __name__ == "__main__":
    main()