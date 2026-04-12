"""
DSLT v4 Training — MPS-Optimized for Apple M4 Pro
v4 Upgrades: 3-scale DSLT (e2+e3+bottleneck), deeper translator blocks (4+6+10),
             LR warmup, resume fine-tuning from best checkpoint.
"""

import sys
import os

# ─── Environment Setup (MUST be before torch import) ─────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "..", ".torch_cache")
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import numpy as np
import time
import json
import copy
import cv2

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.losses import CombinedLoss
from utils.dataloader import PairedDataset


# ─── Device Setup ─────────────────────────────────────────

def get_device():
    """Detect best device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        print("✅ Using Apple M4 Pro GPU (MPS backend)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return torch.device("cuda")
    print("⚠️  Falling back to CPU")
    return torch.device("cpu")


# ─── EMA (Exponential Moving Average) ────────────────────

class EMA:
    """Maintains exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        """Apply EMA weights to model (for evaluation)."""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


# ─── Metrics ──────────────────────────────────────────────

def compute_psnr_batch(output, target):
    """Batch PSNR computation — stays on device, no .item() inside loop."""
    mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean()


def compute_ssim_batch(output, target, window_size=11):
    """Fast differentiable SSIM — stays on device."""
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


# ─── Visualization ────────────────────────────────────────

def save_visualization(degraded, output, clean, epoch, save_dir="results/visualizations"):
    """Save side-by-side comparison every N epochs."""
    os.makedirs(save_dir, exist_ok=True)
    n = min(4, degraded.size(0))
    rows = []
    for i in range(n):
        d = degraded[i].cpu().permute(1, 2, 0).numpy()
        o = output[i].cpu().permute(1, 2, 0).numpy()
        c = clean[i].cpu().permute(1, 2, 0).numpy()
        row = np.concatenate([d, o, c], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    grid_bgr = cv2.cvtColor((np.clip(grid, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_dir}/epoch_{epoch:03d}.jpg", grid_bgr)


# ─── Main Training ────────────────────────────────────────

def train():
    device = get_device()

    # ─── Hyperparameters ──────────────────────────────────
    EPOCHS = 120
    BATCH_SIZE = 8 if device.type == "mps" else 4
    LR = 8e-5       # Lower LR for stable fine-tuning from v3 best checkpoint
    WEIGHT_DECAY = 1e-5
    EMA_DECAY = 0.999
    WARMUP_EPOCHS = 5  # Linear LR warmup to avoid overshooting on resume
    RESUME = True   # Fine-tune from best checkpoint!

    # ─── Data ─────────────────────────────────────────────
    dataset = PairedDataset("data/train_clean", "data/train_degraded", augment=True)
    if len(dataset) == 0:
        print("❌ No data. Run utils/degradation.py first.")
        sys.exit(1)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"Data: {train_size} train | {val_size} val | batch={BATCH_SIZE}")

    # ─── Model ────────────────────────────────────────────
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt = MultiScaleDSLT().to(device)

    total_params = sum(p.numel() for m in [encoder, decoder, dslt] for p in m.parameters())
    print(f"Params: {total_params:,}")

    # ─── Loss, Optimizer, Scheduler ───────────────────────
    criterion = CombinedLoss(device)
    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(dslt.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts: T_0=30 epochs per cycle
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=1, eta_min=2e-6
    )

    # Linear LR warmup override for first WARMUP_EPOCHS epochs
    def get_warmup_lr(epoch):
        if epoch < WARMUP_EPOCHS:
            return LR * (epoch + 1) / WARMUP_EPOCHS
        return None  # Use scheduler after warmup

    # ─── EMA ──────────────────────────────────────────────
    class ModelWrapper(nn.Module):
        def __init__(self, enc, dec, d):
            super().__init__()
            self.encoder = enc
            self.decoder = dec
            self.dslt = d
    model_wrap = ModelWrapper(encoder, decoder, dslt)
    ema = EMA(model_wrap, decay=EMA_DECAY)

    # ─── Resume ───────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    start_epoch = 0
    best_val_psnr = 0.0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": []}

    checkpoint_path = "results/best_model.pth"
    if RESUME and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt['encoder'], strict=True)   # Encoder unchanged
        decoder.load_state_dict(ckpt['decoder'], strict=True)   # Decoder unchanged
        # DSLT v4 has new layers — load what matches, new layers init from scratch
        missing, unexpected = dslt.load_state_dict(ckpt['dslt'], strict=False)
        print(f"  ↳ DSLT: {len(missing)} new keys (random init), {len(unexpected)} dropped")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float("inf"))
        best_val_psnr = ckpt.get('val_psnr', 0.0)
        print(f"📂 Resumed from epoch {start_epoch} (PSNR={best_val_psnr:.2f})")

    # ─── Training Loop ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DSLT v4 Fine-Tuning — epochs {start_epoch+1}→{EPOCHS}, {device}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        # ── Train ──
        encoder.train(); decoder.train(); dslt.train()
        train_loss_acc = 0.0
        n_batches = 0

        for degraded, clean in train_loader:
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            bottleneck, skips = encoder(degraded)
            bn_t, skips_t = dslt(bottleneck, skips)
            correction = decoder(bn_t, skips_t)
            output = torch.clamp(degraded + correction, 0, 1)

            total_loss, _, _, _ = criterion(output, clean)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)  # tighter clip for fine-tuning
            optimizer.step()

            train_loss_acc += total_loss.detach()
            n_batches += 1

        # Sync MPS
        if device.type == "mps":
            torch.mps.synchronize()

        avg_train = (train_loss_acc / n_batches).item()

        # LR warmup override
        warmup_lr = get_warmup_lr(epoch - start_epoch)
        if warmup_lr is not None:
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step(epoch - start_epoch)
        ema.update(model_wrap)

        # ── Validate ──
        encoder.eval(); decoder.eval(); dslt.eval()
        val_loss_acc = 0.0
        val_psnr_acc = 0.0
        val_ssim_acc = 0.0
        val_batches = 0
        last_deg = last_out = last_clean = None

        with torch.no_grad():
            for degraded, clean in val_loader:
                degraded = degraded.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)

                bottleneck, skips = encoder(degraded)
                bn_t, skips_t = dslt(bottleneck, skips)
                correction = decoder(bn_t, skips_t)
                output = torch.clamp(degraded + correction, 0, 1)

                total_loss, _, _, _ = criterion(output, clean)
                val_loss_acc += total_loss.detach()
                val_psnr_acc += compute_psnr_batch(output, clean).detach()
                val_ssim_acc += compute_ssim_batch(output, clean).detach()
                val_batches += 1

                last_deg, last_out, last_clean = degraded, output, clean

        if device.type == "mps":
            torch.mps.synchronize()

        avg_val = (val_loss_acc / val_batches).item()
        avg_psnr = (val_psnr_acc / val_batches).item()
        avg_ssim = (val_ssim_acc / val_batches).item()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)
        history["lr"].append(lr)

        print(f"E[{epoch+1:3d}/{EPOCHS}] T:{avg_train:.4f} V:{avg_val:.4f} "
              f"PSNR:{avg_psnr:.2f} SSIM:{avg_ssim:.4f} LR:{lr:.6f} {elapsed:.0f}s")

        # Save best model
        score = avg_psnr + 10 * avg_ssim  # Combined metric
        best_score = best_val_psnr + 10 * (history["val_ssim"][-2] if len(history["val_ssim"]) > 1 else 0)
        if avg_val < best_val_loss or avg_psnr > best_val_psnr:
            best_val_loss = min(best_val_loss, avg_val)
            best_val_psnr = max(best_val_psnr, avg_psnr)
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'dslt': dslt.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'val_psnr': best_val_psnr,
                'val_ssim': avg_ssim,
            }, checkpoint_path)
            # Also save EMA version
            torch.save(ema.state_dict(), "results/ema_weights.pth")
            print(f"  ✅ Saved (PSNR={best_val_psnr:.2f})")

        # Visualization every 5 epochs
        if (epoch + 1) % 5 == 0 and last_deg is not None:
            save_visualization(last_deg, last_out, last_clean, epoch + 1)

    # ─── Save History + Plots ─────────────────────────────
    with open("results/training_history.json", "w") as f:
        json.dump(history, f)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["val_psnr"], color='green')
        axes[1].set_title("Val PSNR"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(history["val_ssim"], color='blue')
        axes[2].set_title("Val SSIM"); axes[2].grid(True, alpha=0.3)
        axes[3].plot(history["lr"], color='orange')
        axes[3].set_title("Learning Rate"); axes[3].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/training_curves.png", dpi=150)
        print("📊 Curves saved")
    except Exception:
        pass

    print(f"\n🚀 Done! Best PSNR={best_val_psnr:.2f}")


if __name__ == "__main__":
    train()