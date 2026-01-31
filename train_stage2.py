# ==========================================================
# train_stage2.py - Robust Black-box Proactive Defense (RBPD)
# Stage 2: Robustness Enhancement (DFE + MAD + MAA)
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
import lpips

# --- Metrics & Loss ---
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
from loss.ars_loss import LowFreqLoss

# SSIM Loss (对应论文公式 9)
try:
    from pytorch_msssim import SSIM
except ImportError:
    os.system("pip install pytorch-msssim")
    from pytorch_msssim import SSIM

# Project modules
from data.dataset import get_loader
from models.robust_ars import RobustARS
from models.adv_net import AdvNet
from models.face_parser import FaceParser
from config import *

# --- Path Fixing ---
current_dir = os.getcwd() 
for p in ['MagFace', 'AdaFace']:
    path = os.path.join(current_dir, p)
    if path not in sys.path: sys.path.append(path)

# --- Import Extractors ---
from FaceNet_Extractor import FaceNetExtractor
from Arcface.ArcFace_Extractor import ArcFaceExtractor
try: from MagFace.inference.MagFace_Extractor import MagFaceExtractor
except: MagFaceExtractor = None
try: from AdaFace.AdaFaceExtractor_NoAlign import AdaFaceExtractor 
except: AdaFaceExtractor = None

# ==========================================
# 1. 核心损失函数与 DDV 指标
# ==========================================
def resize_tensor(img, size):
    return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=True)

def calc_ddv_score(sim_src, sim_pro, sim_dst):
    """ 实现论文公式 (12): DDV 指标 """
    # sim_src 通常为 1.0 (原图与原图相似度)
    return (sim_dst - sim_pro) / (sim_src - sim_pro + 1e-6)

def calc_protection_loss(x_protected, x_orig, extractors, target_sim=0.25):
    """ 实现论文公式 (5) (6) 的 MAA 策略 """
    facenet, arcface, magface, adaface = extractors
    
    # 特征提取
    with torch.no_grad():
        id_orig_fn = facenet(resize_tensor(x_orig, 160))
        id_orig_arc = arcface(resize_tensor(x_orig, 112))
        id_orig_mag = magface(resize_tensor(x_orig, 112)) if magface else None
        id_orig_ada = adaface(resize_tensor(x_orig, 112)) if adaface else None
        
    id_prot_fn = facenet(resize_tensor(x_protected, 160))
    id_prot_arc = arcface(resize_tensor(x_protected, 112))
    id_prot_mag = magface(resize_tensor(x_protected, 112)) if magface else None
    id_prot_ada = adaface(resize_tensor(x_protected, 112)) if adaface else None
    
    # 余弦相似度
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_fn = cos(id_prot_fn, id_orig_fn).mean()
    sim_arc = cos(id_prot_arc, id_orig_arc).mean()
    sim_mag = cos(id_prot_mag, id_orig_mag).mean() if magface else torch.tensor(0.0).to(DEVICE)
    sim_ada = cos(id_prot_ada, id_orig_ada).mean() if adaface else torch.tensor(0.0).to(DEVICE)
    
    # --- MAA 自适应权重计算 (论文公式 6) ---
    raw_sims = torch.stack([sim_fn, sim_arc, sim_mag, sim_ada])
    beta = torch.tensor([1.0, 1.2, 3.0, 1.5]).to(DEVICE) # 增强因子
    sigma = 0.05 # 温度因子
    
    with torch.no_grad():
        weights = F.softmax((raw_sims * beta) / sigma, dim=0)
        
    # 对抗损失 (公式 5)
    loss_fn = torch.relu(sim_fn - target_sim)
    loss_arc = torch.relu(sim_arc - target_sim)
    loss_mag = torch.relu(sim_mag - target_sim) if magface else 0.0
    loss_ada = torch.relu(sim_ada - target_sim) if adaface else 0.0
    
    total_adv_loss = (weights[0] * loss_fn) + (weights[1] * loss_arc) + \
                     (weights[2] * loss_mag) + (weights[3] * loss_ada)
                 
    return total_adv_loss, raw_sims, weights

# ==========================================
# 2. 训练主循环
# ==========================================
def train_stage2(args):
    # 路径准备
    stage2_dir = os.path.join(CHECKPOINT_DIR, "stage2")
    img_save_dir = os.path.join(stage2_dir, "samples")
    os.makedirs(img_save_dir, exist_ok=True)
    
    # 初始化模型 (DFE + MAD)
    print("Initializing Robust RBPD Model (DFE+MAD)...")
    robust_model = RobustARS(basic_model_path=args.resume, device=DEVICE).to(DEVICE)
    
    # 初始化对抗还原网络 (Restoration Game)
    adv_net = AdvNet().to(DEVICE)
    face_parser = FaceParser(device=DEVICE, weights_path=PARSER_WEIGHTS)
    
    # 加载 4 个异构提取器
    facenet = FaceNetExtractor().to(DEVICE).eval()
    arcface = ArcFaceExtractor(network='r100', weight_path=ARC_WEIGHTS).to(DEVICE).eval()
    magface = MagFaceExtractor(resume=MAG_WEIGHTS, arch='iresnet50').to(DEVICE).eval() if MagFaceExtractor else None
    adaface = AdaFaceExtractor(ckpt_path=ADA_WEIGHTS).to(DEVICE).eval() if AdaFaceExtractor else None
    extractors = (facenet, arcface, magface, adaface)
    
    # 优化器
    opt_gen = optim.Adam(filter(lambda p: p.requires_grad, robust_model.parameters()), lr=1e-4)
    opt_adv = optim.Adam(adv_net.parameters(), lr=1e-4)
    
    # 损失函数实例化
    criterion_mse = nn.MSELoss()
    criterion_lpips = lpips.LPIPS(net='alex').to(DEVICE)
    criterion_ssim = SSIM(data_range=2.0, size_average=True, channel=3).to(DEVICE)
    
    print(f"Start Stage-2 Robustness Training...")
    train_loader = get_loader(mode='train', batch_size=BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        robust_model.train()
        adv_net.train()
        
        metrics = {'fn': 0.0, 'arc': 0.0, 'mag': 0.0, 'ada': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'ddv': 0.0}
        num_batches = len(train_loader)
        loop = tqdm(enumerate(train_loader), total=num_batches)
        
        for i, (x, _) in loop:
            x = x.to(DEVICE)
            with torch.no_grad(): mask = face_parser.get_mask(x)
            
            # --- 阶段 A: 训练还原攻击者 (AdvNet) ---
            # 模拟现实中的还原尝试，增强防御的生存能力
            opt_adv.zero_grad()
            with torch.no_grad():
                eta, _, _ = robust_model(x, mask)
                x_adv = torch.clamp(x + eta * mask, -1.0, 1.0)
            x_res = adv_net(x_adv.detach())
            loss_res = criterion_mse(x_res, x)
            loss_res.backward()
            opt_adv.step()
            
            # --- 阶段 B: 训练 Robust 生成器 (Defender) ---
            opt_gen.zero_grad()
            eta_robust, _, _ = robust_model(x, mask)
            x_bar = torch.clamp(x + eta_robust * mask, -1.0, 1.0)
            
            # 1. 基础防御损失 (公式 5, 11)
            loss_direct, raw_sims, weights = calc_protection_loss(x_bar, x, extractors, target_sim=0.25)
            
            # 2. 抗还原防御损失 (Restoration Game)
            x_restored = adv_net(x_bar)
            loss_robust, _, _ = calc_protection_loss(x_restored, x, extractors, target_sim=0.25)
            
            # 3. 视觉约束 (公式 8, 9, 10)
            l_mse = criterion_mse(x_bar, x)
            l_lpips = criterion_lpips(x_bar, x).mean()
            l_ssim = 1.0 - criterion_ssim(x_bar, x)
            
            # 总损失权重: lambda_adv=1, lambda_mse=100, lambda_ssim=1, lambda_lpips=1
            total_loss = loss_direct + (1.0 * loss_robust) + (100.0 * l_mse) + (1.0 * l_lpips) + (1.0 * l_ssim)
            
            total_loss.backward()
            opt_gen.step()
            
            # --- 指标记录 ---
            with torch.no_grad():
                metrics['fn'] += raw_sims[0].item()
                metrics['arc'] += raw_sims[1].item()
                metrics['mag'] += raw_sims[2].item()
                metrics['ada'] += raw_sims[3].item()
                
                # 计算实时 DDV
                metrics['ddv'] += calc_ddv_score(1.0, raw_sims[1].item(), raw_sims[1].item()) 
                
                x_dn, x_bar_dn = (x + 1)/2, (x_bar + 1)/2
                metrics['psnr'] += psnr(x_bar_dn, x_dn, data_range=1.0).item()
                metrics['ssim'] += ssim_metric(x_bar_dn, x_dn, data_range=1.0).item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix({'arc': f"{raw_sims[1]:.2f}", 'mag': f"{raw_sims[2]:.2f}", 'psnr': f"{metrics['psnr']/(i+1):.1f}"})

        # Epoch 总结
        print(f"\n{'='*20} Epoch {epoch+1} Results {'='*20}")
        print(f" > Sim (Arc/Mag/Ada): {metrics['arc']/num_batches:.3f} / {metrics['mag']/num_batches:.3f} / {metrics['ada']/num_batches:.3f}")
        print(f" > PSNR: {metrics['psnr']/num_batches:.2f} dB | SSIM: {metrics['ssim']/num_batches:.3f}")
        print(f" > Avg DDV: {metrics['ddv']/num_batches:.4f} (Lower is better)")
        
        torch.save(robust_model.state_dict(), os.path.join(stage2_dir, f"rbpd_robust_e{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True, help="Path to Stage-1 checkpoint")
    args = parser.parse_args()
    train_stage2(args)