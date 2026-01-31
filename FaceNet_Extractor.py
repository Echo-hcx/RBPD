# FaceNet_Extractor.py (修正版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class FaceNetExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 移除 MTCNN (CelebA-HQ 已经是通过 MTCNN 对齐过的，无需再次对齐)
        
        # 2. 加载 FaceNet 主干
        # pretrained='vggface2' 意味着模型期望输入在 [-1, 1] 左右
        self.model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

        # 3. 冻结参数 (确保不更新 FaceNet 权重)
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.to(self.device)

    def forward(self, x):
        """
        输入 x: [B, 3, 256, 256], 范围 [-1, 1] (来自 ARS Generator)
        输出 emb: [B, 512], 归一化后的身份向量
        """
        # 1. 可微分 Resize (代替 MTCNN/PIL 操作)
        # FaceNet (InceptionResnetV1) 默认输入通常是 160x160
        if x.shape[2] != 160 or x.shape[3] != 160:
            x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        # 2. 直接喂给模型 (不要加 torch.no_grad() !)
        # 虽然参数冻结了，但我们需要梯度流过模型传回 x
        emb = self.model(x)
        
        # 3. 归一化输出
        emb = F.normalize(emb, p=2, dim=1)
        
        return emb