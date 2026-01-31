import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CBAM 注意力模块 (高大上的核心)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享感知层 MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道，提取空间特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x) # 通道加权
        result = out * self.sa(out) # 空间加权
        return result

# ==========================================
# 2. ResBlock + CBAM (替换原来的 DoubleConv)
# ==========================================
class ResCBAMBlock(nn.Module):
    """
    结构: [Conv-BN-ReLU-Conv-BN] + [Shortcut] -> ReLU -> [CBAM]
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接 (如果维度变了，用1x1卷积调整)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 注意力模块
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        # 在残差块输出后加上注意力机制
        out = self.cbam(out)
        
        return out

# ==========================================
# 3. 整体架构 (Attention-ResUNet)
# ==========================================
# ... 前面的 ResCBAMBlock 等代码保持不变 ...

# ==========================================
# 3. 整体架构 (ARS)
# ==========================================
class ARS(nn.Module):
    # 👇 改动1：增加 epsilon 参数，默认值可以设为 50.0 (0.2)
    def __init__(self, n_channels=4, n_classes=3, epsilon=50.0):
        super(ARS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epsilon = epsilon  # 👇 保存 epsilon
        
        # Encoder (Downsampling)
        self.inc = ResCBAMBlock(n_channels, 64)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResCBAMBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResCBAMBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResCBAMBlock(256, 512))
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = ResCBAMBlock(512, 256) 
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = ResCBAMBlock(256, 128) 
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = ResCBAMBlock(128, 64)  
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        # 1. 拼接输入: Image (3) + Mask (1) = 4 channels
        x_in = torch.cat([x, mask], dim=1) 
        
        # 2. Encoder
        x1 = self.inc(x_in)      
        x2 = self.down1(x1)      
        x3 = self.down2(x2)       
        x4 = self.down3(x3)       
        
        # 3. Decoder
        x_up1 = self.up1(x4)
        if x_up1.shape != x3.shape:
            x_up1 = F.interpolate(x_up1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x_cat1 = torch.cat([x3, x_up1], dim=1)
        x_dec1 = self.conv_up1(x_cat1)
        
        x_up2 = self.up2(x_dec1)
        if x_up2.shape != x2.shape:
            x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x_cat2 = torch.cat([x2, x_up2], dim=1)
        x_dec2 = self.conv_up2(x_cat2)
        
        x_up3 = self.up3(x_dec2)
        if x_up3.shape != x1.shape:
            x_up3 = F.interpolate(x_up3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x_cat3 = torch.cat([x1, x_up3], dim=1)
        x_dec3 = self.conv_up3(x_cat3)
        
        # 4. Output
        eta = self.outc(x_dec3)
        
        # 👇 改动2：使用 self.epsilon 进行缩放
        # 这样 RobustARS 传入的 50.0 就会在这里生效
        eta = self.tanh(eta) * (self.epsilon / 255.0) 
        
        # 生成带攻击的样本
        x_adv = x + eta
        x_adv = torch.clamp(x_adv, -1, 1) 
        
        return eta, x_adv