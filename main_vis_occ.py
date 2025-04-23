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
    traj_file_paths = glob.glob("./data/data_raw_0_1/drone_positions_*.npy")
    trajs = [np.load(traj_file_path) for traj_file_path in traj_file_paths]

    return dilated_occ_index, trajs


def main():
    dilated_occ_index, trajs = load_data()

    print("Loaded trajectories:")
    for i, traj in enumerate(trajs):
        print(f"Trajectory {i + 1}: {traj.shape}")

    points = transfer_points(dilated_occ_index)

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 创建轨迹线
    line_sets = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]  # 预定义颜色
    for i, traj in enumerate(trajs):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(traj)
        line_indices = np.arange(len(traj) - 1).reshape(-1, 1)
        line_indices = np.hstack((line_indices, line_indices + 1))
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector([colors[i % len(colors)] for _ in range(len(traj) - 1)])
        line_sets.append(line_set)

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