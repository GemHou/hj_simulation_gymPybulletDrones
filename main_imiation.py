import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 的 SummaryWriter

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
    # 创建数据集和数据加载器
    print("Loading dataset...")
    npz_file = './data/data_processed_0_1/data_processed_20250424_110042.npz'  # 数据文件路径
    dataset = TrajectoryDataset(npz_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型、损失函数和优化器
    print("Initialing model...")
    model = TrajectoryPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 初始化 TensorBoard 的 SummaryWriter
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=('./runs/trajectory_experiment'+current_time))  # 指定日志保存路径 tensorboard --logdir=./runs

    # 训练模型
    print("Training...")
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_idx = 0
        for drone_pos, target_pos, future_traj in tqdm(dataloader):
            batch_idx += 1
            optimizer.zero_grad()
            outputs = model(drone_pos, target_pos)
            loss = criterion(outputs, future_traj)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 每个批次记录损失到 TensorBoard
            writer.add_scalar('Training Loss per Batch', loss.item(), epoch * len(dataloader) + batch_idx)

        # 每个 epoch 记录平均损失到 TensorBoard
        avg_loss = running_loss / len(dataloader)
        writer.add_scalar('Training Loss per Epoch', avg_loss, epoch)

        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    # 关闭 TensorBoard 的 SummaryWriter
    writer.close()


if __name__ == "__main__":
    main()