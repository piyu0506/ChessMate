import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 12 channels (6 White, 6 Black), 8x8 grid
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy Head (Predicts from 4096 possible from-to square moves)
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        # Value Head (Predicts a score between -1 and 1)
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x is (batch, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy Head Logic
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2 * 8 * 8)
        p = self.policy_fc(p)
        
        # Value Head Logic
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v