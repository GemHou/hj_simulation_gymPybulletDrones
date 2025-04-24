import numpy as np
import torch
from torch.utils.data import Dataset


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
