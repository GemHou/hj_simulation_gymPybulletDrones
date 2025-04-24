import time
import torch
import open3d as o3d
import numpy as np

from main_test_a_star import initialize_environment, perform_path_search, reset_environment, control_loop, save_data
from utils_model import TrajectoryPredictor

RENDER = True  # 设置为 True 以启用可视化
DEVICE = torch.device("cpu")
print("DEVICE: ", DEVICE)


def load_model(model_path):
    """加载训练好的模型"""
    print("Loading model...")
    model = TrajectoryPredictor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def visualize_outputs(outputs):
    """使用 Open3D 可视化 outputs"""
    print("Visualizing outputs...")
    # 假设 outputs 是一个形状为 (batch_size, seq_len, 3) 的张量，表示轨迹点的 (x, y, z) 坐标
    outputs = outputs.detach().cpu().numpy()  # 转换为 NumPy 数组
    if len(outputs.shape) == 3:
        outputs = outputs.squeeze(0)  # 去掉 batch 维度

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(outputs)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Trajectory Visualization")

    # 添加点云到窗口
    vis.add_geometry(pcd)

    # 设置视图参数（可选）
    ctr = vis.get_view_control()
    ctr.set_front([0.5, 0.5, -0.5])  # 设置观察方向
    ctr.set_lookat([0, 0, 0])  # 设置观察目标点
    ctr.set_up([0, -1, 0])  # 设置向上方向
    ctr.set_zoom(0.5)  # 设置缩放比例

    # 运行可视化
    vis.run()
    vis.destroy_window()


def main():
    env, dilated_occ_index = initialize_environment()
    model_path = './data/models/model_epoch_2.pth'  # 替换为实际的模型文件路径
    model = load_model(model_path)
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
        if RENDER:
            visualize_outputs(outputs)  # 调用可视化函数
        save_flag, list_drone_pos, list_target_pos = control_loop(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last,
                                                                  vel_y_last, list_drone_pos, list_target_pos)
        save_data(save_flag, list_drone_pos, list_target_pos)
    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()