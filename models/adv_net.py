# models/adv_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # --- Encoder (下采样路径) ---
        # Block 1: 保持尺寸 (256 -> 256)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: 下采样 (256 -> 128)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: 下采样 (128 -> 64) - 瓶颈层
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- Decoder (上采样路径 + Skip Connection) ---
        
        # Up 1: 64 -> 128
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # 融合层 1: 接收 up1 的输出(128) + enc2 的跳跃连接(128) = 256
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Up 2: 128 -> 256
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # 融合层 2: 接收 up2 的输出(64) + enc1 的跳跃连接(64) = 128
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # --- Output ---
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()  # 输出归一化到 [-1, 1]
        )

    def forward(self, x):
        # --- Encoding ---
        e1 = self.enc1(x)       # (B, 64, H, W)   <-- 记住这个特征图 (Skip 1)
        e2 = self.enc2(e1)      # (B, 128, H/2, W/2) <-- 记住这个特征图 (Skip 2)
        bottleneck = self.enc3(e2) # (B, 256, H/4, W/4)

        # --- Decoding with Skips ---
        
        # Step 1: 上采样 + 拼接 Skip 2
        u1 = self.up1(bottleneck) # (B, 128, H/2, W/2)
        # 在通道维度拼接: 128 + 128 = 256
        cat1 = torch.cat([u1, e2], dim=1) 
        d1 = self.dec1(cat1)
        
        # Step 2: 上采样 + 拼接 Skip 1
        u2 = self.up2(d1)         # (B, 64, H, W)
        # 在通道维度拼接: 64 + 64 = 128
        cat2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(cat2)
        
        # Output
        out = self.out_conv(d2)
        return out