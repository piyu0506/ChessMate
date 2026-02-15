import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=4):
        super(ChessNet, self).__init__()
        # Input: 12 channels (6 white pieces, 6 black pieces)
        self.conv_input = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        self.res_tower = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])
        
        # Policy Head (Actor)
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        # Value Head (Critic)
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # FIX: Reshape the flat 768 input into a 12x8x8 board
        x = x.view(-1, 12, 8, 8) 
        
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_tower(x)
        
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x))).view(-1, 2 * 8 * 8)
        policy = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x))).view(-1, 1 * 8 * 8)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        
        return policy, value