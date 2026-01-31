# models/robust_protection.py
from models.unet import UNet
import torch.nn as nn
import torch


class NoiseEncoder(nn.Module):
    """
    噪声编码 g_n：Conv-BN-ReLU 堆叠，压缩 η_b 到 1x1x128
    保持你原本的实现即可，这部分符合论文 [cite: 200]
    """

    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, eta_b):
        return self.encoder(eta_b)


class RobustProtection(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 128
        self.eps = 0.1

        # 1. g1：图像特征提取 (U-net)
        # 输入: Image (3), 输出: Feature z (128)
        self.g1 = UNet(n_channels=3, n_classes=self.C)

        # 2. gn：噪声编码
        self.gn = NoiseEncoder(in_channels=3, out_channels=self.C)

        # 3. g2：Stage-2 融合模块 (注意：这也是一个 U-Net)
        # 输入通道计算:
        #   z (来自 g1) = 128
        #   n_s (来自 gn 重复) = 128
        #   eta_b (来自 Basic) = 3
        #   Total = 128 + 128 + 3 = 259
        self.g2 = UNet(n_channels=259, n_classes=3)

    def forward(self, x, eta_b):
        """
        x: (B, 3, 256, 256)
        eta_b: (B, 3, 256, 256)
        """
        # Step 1: g1 提取图像特征 z
        z = self.g1(x)  # (B, 128, 256, 256)

        # Step 2: gn 编码基础噪声 -> n
        n = self.gn(eta_b)  # (B, 128, 1, 1)

        # Step 3: 空间重复 n 到 n_s
        n_s = n.repeat(1, 1, x.shape[2], x.shape[3])  # (B, 128, 256, 256)

        # Step 4: 特征融合 (拼接 z, n_s, 和 eta_b)
        # 论文 Eq (6): cat(z, n_s, eta_b)
        fused = torch.cat([z, n_s, eta_b], dim=1)  # (B, 259, 256, 256)

        # Step 5: g2 生成最终噪声 (U-Net 处理)
        eta = self.g2(fused)  # (B, 3, 256, 256)

        # L_inf 约束
        eta = torch.clamp(eta, min=-self.eps, max=self.eps)
        return eta