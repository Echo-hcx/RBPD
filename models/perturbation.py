# models/perturbation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiablePerturbation(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # --- 1. 初始化高斯模糊核 (Blur Kernels) ---
        # 使用 ParameterDict 并设为不可训练，这样它们会自动随模型移动到 GPU
        self.blur_kernels = nn.ParameterDict({
            '3': nn.Parameter(self._create_gaussian_kernel(3, sigma=1.0), requires_grad=False),
            '5': nn.Parameter(self._create_gaussian_kernel(5, sigma=1.5), requires_grad=False)
        })

        # --- 2. 初始化 DCT 权重 (用于 JPEG 模拟) ---
        # 使用 register_buffer 注册为缓冲区，不参与梯度更新，但随模型保存/移动
        self.register_buffer('dct_weight', self._get_dct_weights())

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """生成高斯卷积核"""
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = kernel_size // 2
        variance = sigma ** 2

        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape 为 (1, 1, k, k) -> (3, 1, k, k) 以适配 RGB 分组卷积
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    def _get_dct_weights(self):
        """生成 8x8 DCT 基函数权重"""
        dct_mat = torch.zeros(8, 8)
        for u in range(8):
            for v in range(8):
                if u == 0:
                    a_u = 1.0 / math.sqrt(8)
                else:
                    a_u = math.sqrt(2.0 / 8)
                dct_mat[u, v] = a_u * math.cos((2 * v + 1) * u * math.pi / 16)

        weights = torch.zeros(64, 1, 8, 8)
        for i in range(8):
            for j in range(8):
                basis = torch.outer(dct_mat[i, :], dct_mat[j, :])
                weights[i * 8 + j, 0, :, :] = basis
        return weights

    def p_drop(self, x, keep_prob=0.9):
        """
        随机丢弃像素 (Dropout)，模拟噪声干扰或信息丢失。
        将像素置为 0 (黑色)。
        """
        mask = torch.bernoulli(torch.ones_like(x) * keep_prob)
        return x * mask

    def p_resize(self, x, low=0.6, high=1.0):
        """
        随机缩放再恢复。模拟图像被压缩分辨率。
        注意：F.interpolate 对 input 是可导的，所以梯度可以回传。
        """
        B, C, H, W = x.shape
        # 随机生成缩放比例
        scale = torch.rand(1, device=self.device) * (high - low) + low
        
        new_H = (H * scale).int().item()
        new_W = (W * scale).int().item()
        
        # 兜底防止尺寸过小
        new_H = max(16, new_H)
        new_W = max(16, new_W)

        # 下采样
        x_down = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=True)
        # 上采样回原尺寸 (以便进入后续网络)
        x_recon = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=True)
        return x_recon

    def p_blur(self, x):
        """高斯模糊"""
        # 随机选择核大小
        k_key = '3' if torch.rand(1) < 0.5 else '5'
        kernel = self.blur_kernels[k_key]
        padding = int(k_key) // 2
        # groups=3 保证 RGB 通道独立卷积
        return F.conv2d(x, kernel, padding=padding, groups=3)

    def p_jpeg(self, x, drop_prob=0.5):
        """
        DCT 频域丢弃，模拟 JPEG 压缩带来的高频信息丢失。
        """
        B, C, H, W = x.shape
        
        # 1. Padding 到 8 的倍数
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        else:
            x_pad = x
            
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

        # 2. Reshape 为 (B*C, 1, H, W) 进行卷积
        inputs = x_pad.reshape(B * C, 1, H_pad, W_pad)
        
        # 3. DCT 变换 (卷积实现)
        dct_coeff = F.conv2d(inputs, self.dct_weight, stride=8)

        # 4. 随机频域 Mask (模拟量化丢失)
        # 总是保留 DC 分量 (左上角, index 0)
        mask = torch.bernoulli(torch.ones_like(dct_coeff) * (1 - drop_prob))
        mask[:, 0, :, :] = 1.0 
        dct_coeff = dct_coeff * mask

        # 5. IDCT 逆变换 (转置卷积实现)
        recon = F.conv_transpose2d(dct_coeff, self.dct_weight, stride=8)
        
        # 6. 恢复形状并 Crop 回原尺寸
        x_out = recon.reshape(B, C, H_pad, W_pad)
        return x_out[..., :H, :W]

    def forward(self, x):
            B = x.shape[0]
            out = x.clone() # 复制一份，避免原地修改
            
            # 1. 生成一个 Batch 级别的掩码 (B, 1, 1, 1)
            # 50% 的概率为 1 (做增强)，50% 为 0 (不做增强)
            # 这样每个 Batch 里大约有一半图片会被增强，一半保持原样
            do_aug_mask = torch.bernoulli(torch.ones(B, 1, 1, 1, device=self.device) * 0.5)
            
            # --- 定义临时的增强结果 ---
            x_aug = x.clone()
            
            # 依次应用各种增强（这里逻辑不变）
            if torch.rand(1) < 0.5: x_aug = self.p_resize(x_aug)
            if torch.rand(1) < 0.3: x_aug = self.p_drop(x_aug)
            if torch.rand(1) < 0.3: x_aug = self.p_blur(x_aug)
            if torch.rand(1) < 0.5: x_aug = self.p_jpeg(x_aug)
            
            # --- 关键一步：混合 ---
            # 只有 mask 为 1 的图片会被替换成增强后的 x_aug
            # mask 为 0 的图片保持原样 out
            final_out = out * (1 - do_aug_mask) + x_aug * do_aug_mask
            
            return final_out