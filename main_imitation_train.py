import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d

from utils_dataset import TrajectoryDataset
from utils_model import TrajectoryPredictor

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
VIS_COUNT = 5


def load_datasets(train_npz_file, val_npz_file, batch_size):
    """加载训练集和验证集"""
    print("Loading training dataset...")
    train_dataset = TrajectoryDataset(train_npz_file)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Loading validation dataset...")
    val_dataset = TrajectoryDataset(val_npz_file)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def initialize_model():
    """初始化模型、损失函数和优化器"""
    print("Initializing model...")
    model = TrajectoryPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, train_dataloader, writer, epoch):
    """训练模型"""
    model.train()
    running_loss = 0.0
    running_ade = 0.0
    batch_idx = 0

    for drone_pos, target_pos, future_traj in tqdm(train_dataloader):
        batch_idx += 1
        drone_pos = drone_pos.to(device)
        target_pos = target_pos.to(device)
        future_traj = future_traj.to(device)

        optimizer.zero_grad()
        outputs = model(drone_pos, target_pos)
        loss = criterion(outputs, future_traj)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 计算训练 ADE
        distances = torch.norm(outputs - future_traj, dim=2)
        ade = torch.mean(distances)
        running_ade += ade.item()

        # 每个批次记录损失到 TensorBoard
        writer.add_scalar('Training Loss per Batch', loss.item(), epoch * len(train_dataloader) + batch_idx)
        writer.add_scalar('Training ADE per Batch', ade.item(), epoch * len(train_dataloader) + batch_idx)

    # 每个 epoch 记录平均训练损失到 TensorBoard
    train_avg_loss = running_loss / len(train_dataloader)
    writer.add_scalar('Training Loss per Epoch', train_avg_loss, epoch)

    # 每个 epoch 记录平均训练 ADE 到 TensorBoard
    train_avg_ade = running_ade / len(train_dataloader)
    writer.add_scalar('Training ADE per Epoch', train_avg_ade, epoch)

    print(f'Epoch {epoch + 1}, Training Loss: {train_avg_loss}, Training ADE: {train_avg_ade}')
    return train_avg_loss, train_avg_ade


def validate_model(model, criterion, val_dataloader, writer, epoch):
    """验证模型"""
    model.eval()
    val_running_loss = 0.0
    val_running_ade = 0.0

    with torch.no_grad():
        vis_count = 0
        for drone_pos, target_pos, future_traj in tqdm(val_dataloader):
            drone_pos = drone_pos.to(device)
            target_pos = target_pos.to(device)
            future_traj = future_traj.to(device)

            outputs = model(drone_pos, target_pos)
            loss = criterion(outputs, future_traj)
            val_running_loss += loss.item()

            # 计算验证 ADE
            distances = torch.norm(outputs - future_traj, dim=2)
            ade = torch.mean(distances)
            val_running_ade += ade.item()

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

                # 保存为 PLY 文件，文件名标注训练回合数
                o3d.io.write_point_cloud(f'./data/train_vis/epoch_{epoch}_combined_{vis_count}.ply', combined_pcd)

                vis_count += 1

    # 每个 epoch 记录平均验证损失到 TensorBoard
    val_avg_loss = val_running_loss / len(val_dataloader)
    writer.add_scalar('Validation Loss per Epoch', val_avg_loss, epoch)

    # 每个 epoch 记录平均验证 ADE 到 TensorBoard
    val_avg_ade = val_running_ade / len(val_dataloader)
    writer.add_scalar('Validation ADE per Epoch', val_avg_ade, epoch)

    print(f'Epoch {epoch + 1}, Validation Loss: {val_avg_loss}, Validation ADE: {val_avg_ade}')
    return val_avg_loss, val_avg_ade


def main():
    # 数据文件路径
    train_npz_file = './data/data_processed_0_1/train_data_processed_20250424_163317.npz'
    val_npz_file = './data/data_processed_0_1/val_data_processed_20250424_163317.npz'
    batch_size = 32 * 32

    # 加载数据集
    train_dataloader, val_dataloader = load_datasets(train_npz_file, val_npz_file, batch_size)

    # 初始化模型、损失函数和优化器
    model, criterion, optimizer = initialize_model()

    # 初始化 TensorBoard 的 SummaryWriter
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=('./runs/trajectory_experiment' + current_time))

    # 训练和验证模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train_avg_loss, train_avg_ade = train_model(model, criterion, optimizer, train_dataloader, writer, epoch)
        val_avg_loss, val_avg_ade = validate_model(model, criterion, val_dataloader, writer, epoch)

        # 在每个回合保存模型
        torch.save(model.state_dict(), f'./data/models/model_epoch_{epoch + 1}.pth')

    # 关闭 TensorBoard 的 SummaryWriter
    writer.close()


if __name__ == "__main__":
    main()