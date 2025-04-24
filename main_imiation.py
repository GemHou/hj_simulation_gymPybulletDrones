import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 自定义数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, npz_file):
        """
        Args:
            npz_file (str): Path to the npz file containing the dataset.
        """
        # 加载数据
        data = np.load(npz_file)
        self.drone_positions = torch.tensor(data['drone_positions'], dtype=torch.float32)
        self.target_positions = torch.tensor(data['target_positions'], dtype=torch.float32)
        self.future_trajectories = torch.tensor(data['future_trajectories'], dtype=torch.float32)

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

# 数据文件路径
npz_file = 'path_to_your_dataset.npz'  # 替换为你的数据文件路径

# 创建数据集和数据加载器
dataset = TrajectoryDataset(npz_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = TrajectoryPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for drone_pos, target_pos, future_traj in dataloader:
        optimizer.zero_grad()
        outputs = model(drone_pos, target_pos)
        loss = criterion(outputs, future_traj)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')