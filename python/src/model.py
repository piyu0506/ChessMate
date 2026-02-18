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
    def __init__(self, num_res_blocks=6):
        super(ChessNet, self).__init__()
        
        # --- BODY ---
        self.conv_input = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        self.res_tower = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])
        
        # --- POLICY HEAD (73 Planes) ---
        # We output 73 channels (spatial), then flatten.
        self.policy_conv = nn.Conv2d(128, 73, kernel_size=1) 
        self.policy_bn = nn.BatchNorm2d(73)
        # 73 channels * 8 * 8 = 4672 moves
        
        # --- VALUE HEAD ---
        self.value_conv = nn.Conv2d(128, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 12, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_tower(x)
        
        # Policy
        p = self.policy_conv(x)      # [Batch, 73, 8, 8]
        p = self.policy_bn(p)
        p = p.view(p.size(0), -1)    # Flatten to [Batch, 4672]
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v