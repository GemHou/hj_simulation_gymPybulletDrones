import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 的 SummaryWriter
import open3d as o3d  # 导入 Open3D

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# 自定义数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, npz_file):
        """
        Args:
            npz_file (str): Path to the npz file containing the dataset.
        """
        # 加载数据
        data = np.load(npz_file)
        self.drone_positions = torch.tensor(data['array_drone_pos'], dtype=torch.float32)
        self.target_positions = torch.tensor(data['array_target_pos'], dtype=torch.float32)
        self.future_trajectories = torch.tensor(data['array_future_traj'], dtype=torch.float32)

    def __len__(self):
        return len(self.drone_positions)

    def __getitem__(self, idx):
        return self.drone_positions[idx], self.target_positions[idx], self.future_trajectories[idx]


# 定义神经网络模型
class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super(TrajectoryPredictor, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 30 * 3)
        self.output = nn.Unflatten(1, (30, 3))

    def forward(self, drone_pos, target_pos):
        x = torch.cat((drone_pos, target_pos), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


def main():
    # 创建训练集和数据加载器
    print("Loading training dataset...")
    train_npz_file = './data/data_processed_0_1/train_data_processed_20250424_163317.npz'  # 训练数据文件路径
    train_dataset = TrajectoryDataset(train_npz_file)
    train_dataloader = DataLoader(train_dataset, batch_size=32 * 32, shuffle=True)

    # 创建验证集和数据加载器
    print("Loading validation dataset...")
    val_npz_file = './data/data_processed_0_1/val_data_processed_20250424_163317.npz'  # 验证数据文件路径
    val_dataset = TrajectoryDataset(val_npz_file)
    val_dataloader = DataLoader(val_dataset, batch_size=32 * 32, shuffle=False)

    # 初始化模型、损失函数和优化器
    print("Initializing model...")
    model = TrajectoryPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 初始化 TensorBoard 的 SummaryWriter
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(
        log_dir=('./runs/trajectory_experiment' + current_time))  # 指定日志保存路径 tensorboard --logdir=./runs

    # 训练模型
    print("Training...")
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_ade = 0.0
        batch_idx = 0
        model.train()  # 设置模型为训练模式
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
            # 每个批次记录 ADE 到 TensorBoard
            writer.add_scalar('Training ADE per Batch', ade.item(), epoch * len(train_dataloader) + batch_idx)

        # 每个 epoch 记录平均训练损失到 TensorBoard
        train_avg_loss = running_loss / len(train_dataloader)
        writer.add_scalar('Training Loss per Epoch', train_avg_loss, epoch)

        # 每个 epoch 记录平均训练 ADE 到 TensorBoard
        train_avg_ade = running_ade / len(train_dataloader)
        writer.add_scalar('Training ADE per Epoch', train_avg_ade, epoch)

        print(f'Epoch {epoch + 1}, Training Loss: {train_avg_loss}, Training ADE: {train_avg_ade}')

        # 验证模型
        model.eval()  # 设置模型为评估模式
        val_running_loss = 0.0
        val_running_ade = 0.0
        with torch.no_grad():
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
                    predicted_traj = outputs[i].cpu().numpy()
                    true_traj = future_traj[i].cpu().numpy()

                    # 创建 Open3D 点云对象
                    pcd_predicted = o3d.geometry.PointCloud()
                    pcd_predicted.points = o3d.utility.Vector3dVector(predicted_traj)
                    pcd_predicted.paint_uniform_color([1, 0, 0])  # 红色表示预测轨迹

                    pcd_true = o3d.geometry.PointCloud()
                    pcd_true.points = o3d.utility.Vector3dVector(true_traj)
                    pcd_true.paint_uniform_color([0, 1, 0])  # 绿色表示真实轨迹

                    # 可视化
                    o3d.visualization.draw_geometries([pcd_predicted, pcd_true])

        # 每个 epoch 记录平均验证损失到 TensorBoard
        val_avg_loss = val_running_loss / len(val_dataloader)
        writer.add_scalar('Validation Loss per Epoch', val_avg_loss, epoch)

        # 每个 epoch 记录平均验证 ADE 到 TensorBoard
        val_avg_ade = val_running_ade / len(val_dataloader)
        writer.add_scalar('Validation ADE per Epoch', val_avg_ade, epoch)

        print(f'Epoch {epoch + 1}, Validation Loss: {val_avg_loss}, Validation ADE: {val_avg_ade}')

    # 关闭 TensorBoard 的 SummaryWriter
    writer.close()


if __name__ == "__main__":
    main()