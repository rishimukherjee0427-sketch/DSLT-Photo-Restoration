"""
experiments/ablation_gated_vs_ungated.py
Ablation experiment comparing:
  - Config A: Ungated Baseline (fixed scale gates = 1.0)
  - Config B: Proposed Damage-Conditioned Adaptive Gating

Ensures 100% fair comparison:
  - Sets identical seed before EACH config's initialization
  - Identical data order, epochs, loss function, and learning rate
  - Logs per-epoch PSNR, SSIM, and Loss to results/ablation_results.json
"""

import sys
import os
import ssl

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.damage_estimator import DamageEstimator
from models.losses import CombinedLoss
from utils.dataloader import OnlineDegradationDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_psnr_batch(output, target):
    mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean()


def compute_ssim_batch(output, target, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = output.size(1)
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=output.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(output, window, padding=pad, groups=channels)
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=channels)
    sigma1_sq = torch.nn.functional.conv2d(output ** 2, window, padding=pad, groups=channels) - mu1 ** 2
    sigma2_sq = torch.nn.functional.conv2d(target ** 2, window, padding=pad, groups=channels) - mu2 ** 2
    sigma12 = torch.nn.functional.conv2d(output * target, window, padding=pad, groups=channels) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def train_single_config(config_name, use_gating, epochs, clean_dir, device, seed=42):
    # CRITICAL: Reset seed before EACH config to guarantee identical initialization and data order
    set_seed(seed)
    
    print(f"\n{'='*70}")
    print(f"STARTING ABLATION: {config_name}")
    print(f"Gating Mode: {'Adaptive Gated (Config B)' if use_gating else 'Fixed Ungated 1.0 (Config A)'}")
    print(f"{'='*70}")
    
    # Dataset & Dataloader
    dataset = OnlineDegradationDataset(clean_dir, crop_size=256, augment=True, severity='mild')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Model Setup
    damage_estimator = DamageEstimator(in_channels=3, base_ch=32).to(device)
    encoder = Encoder().to(device)
    dslt = MultiScaleDSLT().to(device)
    decoder = Decoder().to(device)
    criterion = CombinedLoss(device=device, mode='psnr')
    
    all_params = list(damage_estimator.parameters()) + list(encoder.parameters()) + list(dslt.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.99))
    
    history = []
    
    for epoch in range(1, epochs + 1):
        damage_estimator.train(); encoder.train(); dslt.train(); decoder.train()
        
        train_loss_acc = 0.0
        train_psnr_acc = 0.0
        train_ssim_acc = 0.0
        n_batches = 0
        
        for batch_idx, (degraded, clean, gt_severity) in enumerate(loader):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            gt_severity = gt_severity.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Predict damage severity map and scale gates
            pred_severity, scale_gates = damage_estimator(degraded)
            
            # Encoder -> Gated DSLT -> Decoder
            bottleneck, skips = encoder(degraded)
            bn_t, skips_t = dslt(bottleneck, skips, scale_gates=scale_gates, use_gating=use_gating)
            correction = decoder(bn_t, skips_t)
            output = torch.clamp(degraded + correction, 0.0, 1.0)
            
            # Loss calculation
            total_loss, l_char, l_perc, l_ssim, l_adv, l_dmg = criterion(
                output, clean, pred_severity=pred_severity, target_severity=gt_severity
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                psnr_val = compute_psnr_batch(output, clean).item()
                ssim_val = compute_ssim_batch(output, clean).item()
                
            train_loss_acc += total_loss.detach().item()
            train_psnr_acc += psnr_val
            train_ssim_acc += ssim_val
            n_batches += 1
            
            if n_batches >= 20:  # 20 batches per epoch for fast ablation check
                break
                
        avg_loss = train_loss_acc / max(1, n_batches)
        avg_psnr = train_psnr_acc / max(1, n_batches)
        avg_ssim = train_ssim_acc / max(1, n_batches)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim
        })
        
    return history


def run_ablation():
    device = get_device()
    print(f"Device: {device}")
    
    clean_dir = "data/train_clean"
    if not os.path.exists(clean_dir):
        print(f"Warning: {clean_dir} does not exist. Creating fallback directory...")
        os.makedirs(clean_dir, exist_ok=True)
        # Create a few dummy clean images if needed
        for i in range(10):
            dummy = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(clean_dir, f"dummy_{i:02d}.jpg"), dummy)
            
    epochs = 5
    
    # 1. Config A: Fixed Ungated Baseline (use_gating=False)
    history_a = train_single_config(
        config_name="Config A (Ungated Baseline)",
        use_gating=False,
        epochs=epochs,
        clean_dir=clean_dir,
        device=device,
        seed=42
    )
    
    # 2. Config B: Proposed Damage-Conditioned Adaptive Gating (use_gating=True)
    history_b = train_single_config(
        config_name="Config B (Adaptive Damage-Gated)",
        use_gating=True,
        epochs=epochs,
        clean_dir=clean_dir,
        device=device,
        seed=42
    )
    
    # Save comparison to results/
    os.makedirs("results", exist_ok=True)
    out_path = "results/ablation_results.json"
    results_payload = {
        "config_a_ungated": history_a,
        "config_b_adaptive_gated": history_b
    }
    
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f"ABLATION COMPLETE! Comparative metrics written to: {out_path}")
    print("=" * 70)
    print(f"Final Performance Comparison (Epoch {epochs}):")
    print(f"  - Config A (Ungated Baseline):        PSNR = {history_a[-1]['psnr']:.2f} dB, SSIM = {history_a[-1]['ssim']:.4f}")
    print(f"  - Config B (Adaptive Damage Gating):  PSNR = {history_b[-1]['psnr']:.2f} dB, SSIM = {history_b[-1]['ssim']:.4f}")
    diff_psnr = history_b[-1]['psnr'] - history_a[-1]['psnr']
    print(f"  -> Delta PSNR: {diff_psnr:+.2f} dB")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation()
