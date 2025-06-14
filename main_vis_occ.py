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


def load_occ_data():
    dilated_occ_file_path = "./data/dilated_occ_index.npy"
    dilated_occ_index = np.load(dilated_occ_file_path)

    # 获取所有轨迹文件路径
    traj_file_paths = glob.glob("./data/data_raw_0_2/data_raw_*.npz")
    traj_file_paths.sort()

    drone_trajs = []
    target_trajs = []

    for traj_file_path in tqdm(traj_file_paths):
        # print("traj_file_path: ", traj_file_path)
        drone_trajs.append(np.load(traj_file_path)["drone_pos_array"])
        target_trajs.append(np.load(traj_file_path)["target_pos_array"])

    return dilated_occ_index, drone_trajs, target_trajs


def print_trajectory_info(drone_trajs, target_trajs):
    total_frames = 0
    print("Loaded trajectories:")
    for i, (drone_traj, target_traj) in enumerate(zip(drone_trajs, target_trajs)):
        print(f"Trajectory {i + 1}: Drone {drone_traj.shape}, Target {target_traj.shape}")
        frames_in_trajectory = drone_traj.shape[0]
        total_frames += frames_in_trajectory
    print(f"Total frames: {total_frames}")


def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_line_set(traj, base_color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj)
    line_indices = np.arange(len(traj) - 1).reshape(-1, 1)
    line_indices = np.hstack((line_indices, line_indices + 1))
    colors = []
    for j in range(len(traj) - 1):
        factor = j / (len(traj) - 1)
        new_color = [1 - (1 - base_color[k]) * factor for k in range(3)]
        colors.append(new_color)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_all_line_sets(drone_trajs, target_trajs):
    line_sets = []
    base_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]  # 预定义颜色

    for i in tqdm(range(len(drone_trajs))):
        drone_traj = drone_trajs[i]
        target_traj = target_trajs[i]
        base_color = base_colors[i % len(base_colors)]

        # 创建无人机轨迹线
        drone_line_set = create_line_set(drone_traj, base_color)
        line_sets.append(drone_line_set)

        # 创建目标轨迹线
        target_line_set = create_line_set(target_traj, base_color)
        line_sets.append(target_line_set)

    return line_sets


def main():
    dilated_occ_index, drone_trajs, target_trajs = load_occ_data()
    print_trajectory_info(drone_trajs, target_trajs)

    drone_trajs = drone_trajs[:5000]
    target_trajs = target_trajs[:5000]

    print("transfer_points...")
    points = transfer_points(dilated_occ_index)

    # 创建 Open3D 点云
    print("o3d point")
    pcd = create_point_cloud(points)

    # 创建轨迹线
    print("o3d line")
    line_sets = create_all_line_sets(drone_trajs, target_trajs)

    # 可视化点云和轨迹
    print("o3d vis")
    o3d.visualization.draw_geometries(
        [pcd] + line_sets,
        window_name="3D Occupancy and Trajectories Visualization",
        width=800,
        height=600
    )

    print("Finished...")


if __name__ == "__main__":
    main()
