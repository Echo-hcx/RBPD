# train.py
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

# Metrics library
from torchmetrics.functional import peak_signal_noise_ratio as psnr

# Project modules
from data.dataset import get_loader
from models.ars_attention import ARS
from config import *
from models.face_parser import FaceParser
from models.perturbation import DifferentiablePerturbation

# --- Path Fixing for External Modules ---
current_dir = os.getcwd() 
magface_path = os.path.join(current_dir, 'MagFace')
if magface_path not in sys.path: sys.path.append(magface_path)
adaface_path = os.path.join(current_dir, 'AdaFace')
if adaface_path not in sys.path: sys.path.append(adaface_path)

# ==========================================
# 1. Import Identity Extractors
# ==========================================
try: from FaceNet_Extractor import FaceNetExtractor
except: raise ImportError("Missing FaceNet_Extractor")
try: from Arcface.ArcFace_Extractor import ArcFaceExtractor
except: raise ImportError("Missing ArcFace_Extractor")
try: from MagFace.inference.MagFace_Extractor import MagFaceExtractor
except: MagFaceExtractor = None
try: from AdaFace.AdaFaceExtractor_NoAlign import AdaFaceExtractor 
except: AdaFaceExtractor = None

# ==========================================
# 2. Utility Functions
# ==========================================
def resize_tensor(img, size):
    return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=True)

def load_extractors():
    print("Loading Identity Extractors...")
    facenet = FaceNetExtractor().to(DEVICE).eval()
    for p in facenet.parameters(): p.requires_grad = False

    arcface = ArcFaceExtractor(network='r100', weight_path='/disk/disk4/hcx/ARS_pert/Arcface/ms1mv3_arcface_r100_fp16/backbone.pth').to(DEVICE).eval()
    for p in arcface.parameters(): p.requires_grad = False

    magface = None
    if MagFaceExtractor:
        try:
            magface = MagFaceExtractor(resume='/disk/disk4/hcx/ARS_pert/MagFace/pretrained/magface_iresnet50_MS1MV2_ddp_fp32.pth', arch='iresnet50').to(DEVICE).eval()
            for p in magface.parameters(): p.requires_grad = False
        except Exception as e:
            print(f"[WARN] MagFace loaded failed: {e}")

    adaface = None
    if AdaFaceExtractor:
        try:
            # 这里的路径请根据你的实际情况确认
            adaface = AdaFaceExtractor(ckpt_path='/disk/disk4/hcx/ARS_pert/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt').to(DEVICE).eval()
            for p in adaface.parameters(): p.requires_grad = False
        except Exception as e:
            print(f"[WARN] AdaFace loaded failed: {e}")

    return facenet, arcface, magface, adaface

def save_sample_images(clean, adv, mask, epoch, save_dir, num_samples=8):
    os.makedirs(save_dir, exist_ok=True)
    current = min(clean.size(0), num_samples)
    clean, adv, mask = clean[:current], adv[:current], mask[:current]
    
    clean = (clean + 1) / 2
    adv = (adv + 1) / 2
    mask_vis = mask.repeat(1, 3, 1, 1) # Mask 变 3 通道方便显示
    
    combined = torch.cat([clean, mask_vis, adv], dim=0)
    save_image(combined, os.path.join(save_dir, f"sample_epoch_{epoch + 1}.png"), nrow=current, padding=2)

# ==========================================
# 3. Main Training Loop
# ==========================================
def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    img_save_dir = os.path.join(CHECKPOINT_DIR, "samples")
    os.makedirs(img_save_dir, exist_ok=True)

    # 1. Init Data & Models
    train_loader = get_loader(
        image_dir=os.path.join(DATA_ROOT, TRAIN_SPLIT), 
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        mode='train', num_workers=16, num_samples=MAX_TRAIN_IMAGES
    )
    
    # FaceParser (用于生成 Mask)
    FACE_PARSER_PATH = '/disk/disk4/hcx/ARS_pert/checkpoints/79999_iter.pth'
    print(f"Initializing FaceParser from {FACE_PARSER_PATH}...")
    face_parser = FaceParser(device=DEVICE, weights_path=FACE_PARSER_PATH)

    # ARS Model (Stage 1)
    print("Initializing ARS Model (Semantic-Guided)...")
    # 注意：这里的 ARS 必须是你修改过接受 4 通道输入的版本
    ars_model = ARS().to(DEVICE)

    # Diff Aug (用于鲁棒性训练)
    print("Initializing Differentiable Augmentation...")
    diff_aug = DifferentiablePerturbation(device=DEVICE).to(DEVICE)
    diff_aug.eval() 

    # Resume Logic
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading weights: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        ars_model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    else:
        print("[INFO] No resume path provided or file not found. Training from scratch.")
    
    # Extractors
    facenet, arcface, magface, adaface = load_extractors()
    
    # 打印活跃的模型，方便调试
    active_models = ['facenet', 'arcface']
    if magface: active_models.append('magface')
    if adaface: active_models.append('adaface')
    print(f"[INFO] Active Extractor Models: {active_models}")

    # Optimizers
    optimizer = optim.Adam(ars_model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)

    # Loss Functions
    criterion_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    criterion_mse = nn.MSELoss()
    criterion_lpips = lpips.LPIPS(net='alex').to(DEVICE)

    # Hyperparams
    TARGET_SIM = 0.25
    LAMBDA_ADV = 1.0
    LAMBDA_MSE = 100.0   # 强约束MSE保证原图相似
    LAMBDA_LPIPS = 1.0

    print(f"Start Training (Semantic Guidance + DiffAug)...")

    for epoch in range(EPOCHS):
        ars_model.train()
        
        # Loggers
        acc_sims = {'fn': 0.0, 'arc': 0.0, 'mag': 0.0, 'ada': 0.0}
        acc_weights = {'fn': 0.0, 'arc': 0.0, 'mag': 0.0, 'ada': 0.0}
        acc_psnr = 0.0
        num_batches = len(train_loader)
        
        loop = tqdm(enumerate(train_loader), total=num_batches)
        
        last_clean, last_adv, last_mask = None, None, None

        for batch_idx, (x, _) in loop:
            x = x.to(DEVICE, non_blocking=True)
            
            # --- A. Generate Semantic Mask (导师) ---
            with torch.no_grad():
                mask = face_parser.get_mask(x)

            # --- B. Forward Pass ---
            optimizer.zero_grad()
            
            # 【关键点 1】: 将 Mask 喂入网络 (Input Concatenation)
            eta, x_raw_adv = ars_model(x, mask)
            
            # 【关键点 2】: 应用 Mask 硬约束 (Output Constraint)
            delta = x_raw_adv - x
            delta_masked = delta * mask
            
            # 得到最终干净的对抗样本
            x_bar = torch.clamp(x + delta_masked, -1.0, 1.0)
            
            # --- C. Robustness Simulation (DiffAug) ---
            # 为了让生成的噪声不仅不可见，还要抗造，我们在计算攻击Loss前先"折磨"它一下
            # x_input_for_loss 可能是模糊的、压缩过的或原图
            x_input_for_loss = diff_aug(x_bar)
            
            if batch_idx == num_batches - 1:
                last_clean, last_adv, last_mask = x.detach(), x_bar.detach(), mask.detach()

            # --- D. Loss Calculation ---
            
            # 1. Protection Loss (攻击效果) -> 用增强后的图算！
            x_in_fn = resize_tensor(x_input_for_loss, 160)
            x_in_112 = resize_tensor(x_input_for_loss, 112)
            
            # Get Features
            with torch.no_grad():
                # 原图特征 (无增强)
                id_orig_fn = facenet(resize_tensor(x, 160))
                id_orig_arc = arcface(resize_tensor(x, 112))
                id_orig_mag = magface(resize_tensor(x, 112)) if magface else None
                id_orig_ada = adaface(resize_tensor(x, 112)) if adaface else None
            
            # 对抗图特征 (有增强)
            id_prot_fn = facenet(x_in_fn)
            id_prot_arc = arcface(x_in_112)
            id_prot_mag = magface(x_in_112) if magface else None
            id_prot_ada = adaface(x_in_112) if adaface else None
            
            # Cosine Similarity
            # 如果模型没加载成功，similarity 默认为 0
            sim_fn = criterion_cos(id_prot_fn, id_orig_fn).mean()
            sim_arc = criterion_cos(id_prot_arc, id_orig_arc).mean()
            sim_mag = criterion_cos(id_prot_mag, id_orig_mag).mean() if magface else torch.tensor(0.0).to(DEVICE)
            sim_ada = criterion_cos(id_prot_ada, id_orig_ada).mean() if adaface else torch.tensor(0.0).to(DEVICE)
            
            beta = torch.tensor([1.0, 1.2, 3.0, 1.5]).to(DEVICE) # 论文中的增强因子 [cite: 543]
            sigma = 0.05 # 温度因子 [cite: 543]
            
            # 计算 Cosine Similarity
            raw_sims = torch.stack([sim_fn, sim_arc, sim_mag, sim_ada])
            
            # MAA 自适应权重计算 [cite: 493]
            with torch.no_grad():
                weights = F.softmax((raw_sims * beta) / sigma, dim=0)
            
            w_fn, w_arc, w_mag, w_ada = weights[0], weights[1], weights[2], weights[3]
            
            # 这里的 TARGET_SIM 统一为 0.25 [cite: 542]
            loss_adv = (w_fn * torch.relu(sim_fn - TARGET_SIM)) + \
                       (w_arc * torch.relu(sim_arc - TARGET_SIM)) + \
                       (w_mag * torch.relu(sim_mag - TARGET_SIM) if magface else 0) + \
                       (w_ada * torch.relu(sim_ada - TARGET_SIM) if adaface else 0)

            # 视觉损失设置 [cite: 544]
            loss_mse = criterion_mse(x_bar, x)
            loss_lpips_val = criterion_lpips(x_bar, x).mean()
            
            loss_adv = (w_fn * torch.relu(sim_fn - TARGET_SIM)) + \
                       (w_arc * torch.relu(sim_arc - TARGET_SIM)) + \
                       (w_mag * torch.relu(sim_mag - TARGET_SIM) if magface else 0) + \
                       (w_ada * torch.relu(sim_ada - TARGET_SIM) if adaface else 0)

            # 2. Visual Loss (视觉质量) -> 用干净的对抗样本算！
            loss_mse = criterion_mse(x_bar, x)
            loss_lpips_val = criterion_lpips(x_bar, x).mean()
            
            # Total Loss
            loss = (LAMBDA_ADV * loss_adv) + (LAMBDA_MSE * loss_mse) + (LAMBDA_LPIPS * loss_lpips_val)
            
            loss.backward()
            optimizer.step()
            
            # --- Logging ---
            acc_sims['fn'] += sim_fn.item()
            acc_sims['arc'] += sim_arc.item()
            acc_weights['fn'] += w_fn.item()
            acc_weights['arc'] += w_arc.item()
            
            acc_sims['mag'] += sim_mag.item() # 无条件累加，即使是0
            if magface: acc_weights['mag'] += w_mag.item()
            
            acc_sims['ada'] += sim_ada.item() # 无条件累加，即使是0
            if adaface: acc_weights['ada'] += w_ada.item()
            
            with torch.no_grad():
                # PSNR calc on clean adv (un-augmented)
                x_dn = (x + 1) / 2
                x_bar_dn = (x_bar + 1) / 2
                batch_psnr = psnr(x_bar_dn, x_dn, data_range=1.0)
            acc_psnr += batch_psnr.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            
            # 【修改】强制显示所有指标，没有加载则显示 0.00
            postfix = {
                'fn': f"{sim_fn:.2f}", 
                'arc': f"{sim_arc:.2f}", 
                'mag': f"{sim_mag:.2f}",
                'ada': f"{sim_ada:.2f}", # 强制显示
                'psnr': f"{batch_psnr:.1f}"
            }
            loop.set_postfix(postfix)
            
        scheduler.step()
        
        # --- Epoch Summary ---
        print(f"\n{'='*20} Epoch {epoch+1} Summary {'='*20}")
        print(f" > FaceNet : {acc_sims['fn']/num_batches:.4f} (W: {acc_weights['fn']/num_batches:.2f})")
        print(f" > ArcFace : {acc_sims['arc']/num_batches:.4f} (W: {acc_weights['arc']/num_batches:.2f})")
        
        # 【修改】强制打印 MagFace 和 AdaFace
        print(f" > MagFace : {acc_sims['mag']/num_batches:.4f} (W: {acc_weights['mag']/num_batches:.2f})")
        print(f" > AdaFace : {acc_sims['ada']/num_batches:.4f} (W: {acc_weights['ada']/num_batches:.2f})")
        
        print(f" > PSNR    : {acc_psnr/num_batches:.2f} dB")
        print(f"{'='*56}\n")
        
        save_path = os.path.join(CHECKPOINT_DIR, f"ars_stage1_epoch_{epoch+1}.pth")
        torch.save(ars_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        if (epoch+1) % 5 == 0 and last_clean is not None:
            save_sample_images(last_clean, last_adv, last_mask, epoch, img_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)