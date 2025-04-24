import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader  # 确保导入 DataLoader

from utils_dataset import TrajectoryDataset
from utils_model import TrajectoryPredictor

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
VIS_COUNT = 5  # 可视化轨迹的数量


def load_tst_dataset(tst_npz_file, batch_size):
    """加载测试集"""
    print("Loading tst dataset...")
    tst_dataset = TrajectoryDataset(tst_npz_file)
    tst_dataloader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)
    return tst_dataloader


def load_model(model_path):
    """加载训练好的模型"""
    print("Loading model...")
    model = TrajectoryPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def tst_model(model, tst_dataloader):
    """测试模型并进行可视化"""
    with torch.no_grad():
        vis_count = 0
        for drone_pos, target_pos, future_traj in tqdm(tst_dataloader):
            drone_pos = drone_pos.to(device)
            target_pos = target_pos.to(device)
            future_traj = future_traj.to(device)

            outputs = model(drone_pos, target_pos)

            # 可视化预测轨迹和真实轨迹
            for i in range(len(outputs)):
                if vis_count >= VIS_COUNT:
                    break
                predicted_traj = outputs[i].cpu().numpy()
                true_traj = future_traj[i].cpu().numpy()

                # 创建 Open3D 点云对象
                pcd_predicted = o3d.geometry.PointCloud()
                pcd_predicted.points = o3d.utility.Vector3dVector(predicted_traj)
                pcd_predicted.paint_uniform_color([1, 0, 0])  # 红色表示预测轨迹

                pcd_true = o3d.geometry.PointCloud()
                pcd_true.points = o3d.utility.Vector3dVector(true_traj)
                pcd_true.paint_uniform_color([0, 1, 0])  # 绿色表示真实轨迹

                # 合并点云
                combined_pcd = pcd_predicted + pcd_true

                # 可视化
                o3d.visualization.draw_geometries([combined_pcd])

                # 保存为 PLY 文件
                o3d.io.write_point_cloud(f'./data/tst_vis/tst_combined_{vis_count}.ply', combined_pcd)

                vis_count += 1


def main():
    # 数据文件路径
    tst_npz_file = './data/data_processed_0_1/test_data_processed_20250424_163317.npz'
    model_path = './data/models/model_epoch_1.pth'  # 替换为实际的模型文件路径
    batch_size = 32 * 32

    # 加载测试集
    tst_dataloader = load_tst_dataset(tst_npz_file, batch_size)

    # 加载模型
    model = load_model(model_path)

    # 测试模型并进行可视化
    tst_model(model, tst_dataloader)


if __name__ == "__main__":
    main()