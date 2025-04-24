import time
import torch
import open3d as o3d
import numpy as np

from main_test_a_star import initialize_environment, perform_path_search, reset_environment, control_loop, save_data
from utils_model import TrajectoryPredictor

RENDER_OPEN3D = False
RENDER_PYBULLET = True
DEVICE = torch.device("cpu")
print("DEVICE: ", DEVICE)


def load_model(model_path):
    """加载训练好的模型"""
    print("Loading model...")
    model = TrajectoryPredictor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def visualize_outputs(outputs, pcd_occ, drone_pos):
    """使用 Open3D 可视化 outputs"""
    print("Visualizing outputs...")
    # 假设 outputs 是一个形状为 (batch_size, seq_len, 3) 的张量，表示轨迹点的 (x, y, z) 坐标
    outputs = outputs.detach().cpu().numpy()  # 转换为 NumPy 数组
    if len(outputs.shape) == 3:
        outputs = outputs.squeeze(0)  # 去掉 batch 维度

    outputs = outputs + drone_pos

    # 创建点云
    pcd_traj = o3d.geometry.PointCloud()
    pcd_traj.points = o3d.utility.Vector3dVector(outputs)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Trajectory Visualization")

    # o3d.visualization.draw_geometries(
    #     [pcd],
    #     window_name="3D Occupancy and Trajectories Visualization",
    #     width=800,
    #     height=600
    # )

    # 添加点云到窗口
    vis.add_geometry(pcd_traj)
    vis.add_geometry(pcd_occ)

    # 设置视图参数（可选）
    ctr = vis.get_view_control()
    ctr.set_front([0.5, 0.5, -0.5])  # 设置观察方向
    ctr.set_lookat([0, 0, 0])  # 设置观察目标点
    ctr.set_up([0, -1, 0])  # 设置向上方向
    ctr.set_zoom(0.5)  # 设置缩放比例

    # 运行可视化
    vis.run()
    vis.destroy_window()


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


def create_point_cloud(points):
    pcd_occ = o3d.geometry.PointCloud()
    pcd_occ.points = o3d.utility.Vector3dVector(points)
    return pcd_occ


def main():
    env, dilated_occ_index = initialize_environment(RENDER_PYBULLET)
    model_path = './data/models/model_epoch_10.pth'  # 替换为实际的模型文件路径
    model = load_model(model_path)
    if RENDER_OPEN3D:
        points = transfer_points(dilated_occ_index)
        pcd_occ = create_point_cloud(points)
    print("Looping...")
    while True:
        obs_ma, save_flag, list_drone_pos, list_target_pos, drone_pos, vel_x_last, vel_y_last, target_pos = reset_environment(
            env)
        path_result = perform_path_search(dilated_occ_index, drone_pos, target_pos)
        if path_result is None:
            continue
        tensor_drone_pos = torch.tensor([drone_pos], dtype=torch.float32)
        tensor_target_pos = torch.tensor([target_pos], dtype=torch.float32)
        outputs = model(tensor_drone_pos, tensor_target_pos)
        if RENDER_OPEN3D:
            visualize_outputs(outputs, pcd_occ, drone_pos)  # 调用可视化函数
        save_flag, list_drone_pos, list_target_pos = control_loop(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last,
                                                                  vel_y_last, list_drone_pos, list_target_pos,
                                                                  RENDER_PYBULLET)
        save_data(save_flag, list_drone_pos, list_target_pos)
    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()
