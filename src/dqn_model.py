import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    深度 Q 网络
    输入：状态 (归一化坐标 + 8方向障碍物) → 10 维
    输出：4 个动作的 Q 值（上下左右）
    """

    def __init__(self, state_dim=10, action_dim=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
