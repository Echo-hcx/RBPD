import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ars_attention import ARS  

class SEBlock(nn.Module):
    """ 
    论文 提到的 SE-Block，用于自适应重校准通道响应
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DilatedDiffusionBlock(nn.Module):
    """
    【对齐论文 MAD 模块 】
    集成多尺度空洞卷积与 SE-Block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        # Branch 1-3: 多尺度空洞卷积 (对应论文 d=1, 2, 4) 
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2), nn.BatchNorm2d(mid_channels), nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=4, dilation=4), nn.BatchNorm2d(mid_channels), nn.ReLU(True))
        
        # Branch 4: 全局上下文
        self.branch4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, mid_channels, 1), nn.BatchNorm2d(mid_channels), nn.ReLU(True))
        
        # 融合与 SE-Block 校准 
        self.fusion = nn.Conv2d(mid_channels * 4, out_channels, 1)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        b1, b2, b3 = self.branch1(x), self.branch2(x), self.branch3(x)
        b4 = F.interpolate(self.branch4(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        out = self.fusion(torch.cat([b1, b2, b3, b4], dim=1))
        return self.se(out)

class RobustARS(nn.Module):
    """ 对应论文 Stage II 架构: DFE + MAD """
    def __init__(self, basic_model_path=None, device='cuda'):
        super().__init__()
        self.basic_protection = ARS(n_channels=4, epsilon=7.0).to(device) # epsilon 对齐论文 
        if basic_model_path: self.load_weights(basic_model_path, device)
        
        # DFE: 双流融合编码器 
        self.g1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.g_n = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        
        # MAD: 多尺度聚合解码器 
        self.diffusion = DilatedDiffusionBlock(128, 128)
        self.g2 = nn.Sequential(nn.Conv2d(128 + 3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                                nn.Conv2d(64, 3, 1), nn.Tanh())
        self.epsilon_robust = 0.5 

    def load_weights(self, path, device):
        ckpt = torch.load(path, map_location=device)
        self.basic_protection.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        for p in self.basic_protection.parameters(): p.requires_grad = False
        self.basic_protection.eval()

    def forward(self, x, mask):
        with torch.no_grad(): eta_b, _ = self.basic_protection(x, mask)
        z_fused = self.diffusion(torch.cat([self.g1(x), self.g_n(eta_b)], dim=1))
        eta_robust = self.g2(torch.cat([z_fused, eta_b], dim=1)) * self.epsilon_robust
        return eta_robust, torch.clamp(x + eta_robust, -1.0, 1.0), eta_b