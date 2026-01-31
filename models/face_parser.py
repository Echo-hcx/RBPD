import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import torchvision.transforms as T  # <--- [新增] 引入 transforms

# -----------------------------------------------------------------------------
# 1. 基础模块 (Basic Blocks) - 保持不变
# -----------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = models.resnet18()
        del self.resnet.fc 
        
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.resnet.conv1(x)
        feat = self.resnet.bn1(feat)
        feat = self.resnet.relu(feat)
        feat = self.resnet.maxpool(feat) # Stride 4

        feat8 = self.resnet.layer1(feat) 
        feat16 = self.resnet.layer2(feat8) 
        feat32 = self.resnet.layer3(feat16) 
        feat_res4 = self.resnet.layer4(feat32) 
        
        avg = F.avg_pool2d(feat_res4, feat_res4.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=feat_res4.size()[2:], mode='nearest')

        feat32_arm = self.arm32(feat_res4)
        feat32_sum = feat32_arm + avg_up
        
        # Stride 32 -> 16
        feat32_up = F.interpolate(feat32_sum, size=feat32.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        # Stride 16 -> 8
        feat16_arm = self.arm16(feat32)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat16.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, size=(H, W), mode='bilinear', align_corners=True)
        return feat_out

# -----------------------------------------------------------------------------
# 4. 封装接口 (FaceParser) - 已修改
# -----------------------------------------------------------------------------
class FaceParser(nn.Module):
    def __init__(self, device='cuda', weights_path=None):
        super().__init__()
        self.device = device
        # 初始化 BiSeNet (19个类别)
        self.net = BiSeNet(n_classes=19) 
        self.net.to(device)
        
        # === 语义权重配置 (保持你原有的配置) ===
        self.mask_weights = {
            0: 0.0,  # 背景
            1: 0.4,  # 皮肤
            2: 1.0,  # 左眉
            3: 1.0,  # 右眉
            4: 1.0,  # 左眼
            5: 1.0,  # 右眼
            6: 0.5,  # 眼镜
            7: 0.5,  # 左耳
            8: 0.5,  # 右耳
            9: 0.5,  # 耳环
            10: 1.0, # 鼻子
            11: 1.0, # 嘴内
            12: 1.0, # 上嘴唇
            13: 1.0, # 下嘴唇
            14: 0.0, # 颈部
            15: 0.0, # 项链
            16: 0.0, # 衣服
            17: 0.8, # 头发
            18: 0.0  # 帽子
        }

        # 如果传入了路径，立即加载
        if weights_path:
            self.load_weights(weights_path)
            
        self.net.eval()

    def load_weights(self, path):
        """
        鲁棒的权重加载函数：同时处理 'module.' 前缀和 'spatial_path'->'sp' 命名不匹配问题
        """
        if not os.path.isfile(path):
            print(f"[Warning] FaceParser weights not found at {path}")
            return
            
        print(f"Loading FaceParser weights from {path}...")
        state_dict = torch.load(path, map_location=self.device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # 1. 去掉 DDP 带来的 'module.' 前缀
            name = k[7:] if k.startswith('module.') else k
            
            # 2. 关键修复：映射 BiSeNet 的变量名
            # 你的模型里叫 sp/cp，权重文件里通常叫 spatial_path/context_path
            if 'spatial_path' in name:
                name = name.replace('spatial_path', 'sp')
            elif 'context_path' in name:
                name = name.replace('context_path', 'cp')
                
            new_state_dict[name] = v
            
        # 尝试加载
        try:
            self.net.load_state_dict(new_state_dict, strict=True)
            print("✅ FaceParser weights loaded successfully (strict=True).")
        except RuntimeError as e:
            # 如果还有极少数不匹配（比如 aux loss headers），退回 strict=False
            print(f"[Info] Strict loading mismatch (expected if ignoring aux heads). Error: {e}")
            self.net.load_state_dict(new_state_dict, strict=False)
            print("✅ FaceParser weights loaded with strict=False.")

    def get_mask(self, x):
        """
        输入图像 x: (B, 3, H, W)
        输出掩码 mask: (B, 1, H, W) -> 单通道权重图
        """
        H, W = x.shape[2], x.shape[3]
        
        # 1. 预处理：BiSeNet 需要 512x512 的输入
        # 使用 bilinear 插值调整大小
        x_in = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        
        # 建议：这里最好加上标准化，因为 BiSeNet 训练时通常用了 ImageNet 的 mean/std
        # 如果你之前的代码没加且效果还好，可以注释掉下面这行
        # x_in = (x_in - torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        
        with torch.no_grad():
            # === 修复点 ===
            # 原代码: out = self.net(x_in)[0] 
            # 你的 BiSeNet forward 只返回 feat_out 一个变量，所以不能加 [0]
            out = self.net(x_in) # 输出形状 (B, 19, 512, 512)
            
            # 此时 out 应该是 4 维张量。如果误加了 [0]，它会变成 3 维，导致后续报错。
            parsing = out.argmax(1) # 输出形状 (B, 512, 512)

        # 2. 类别映射到权重
        # 创建一个全 0 的 mask
        mask = torch.zeros_like(parsing, dtype=torch.float32)
        
        # 遍历字典赋值
        for class_id, weight in self.mask_weights.items():
            # 只处理权重不为 0 的，节省一点计算
            if weight > 0:
                mask[parsing == class_id] = weight

        # 3. 后处理：插值回原图尺寸
        # 必须先 unsqueeze 变成 (B, 1, 512, 512) 才能送入 interpolate
        mask = mask.unsqueeze(1) 
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        # mask 保持 (B, 1, H, W) 输出，方便后续直接和图片相乘
        return mask