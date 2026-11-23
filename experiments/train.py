"""
DEPRECATED (legacy v5 pipeline): trains/evaluates on ALL of data/train_clean, i.e. no held-out
split (train/test leakage). Do not use for reported results. Use instead:
  training   : experiments/run_ablation.py, experiments/train_baseline.py (data/splits/train.txt)
  evaluation : experiments/eval_all.py (held-out data/splits/test.txt, cached degradations)
"""
"""
DSLT v5 Training — Progressive Multi-Phase Training
Phase 1: 128×128 patches, learn coarse structure
Phase 2: 256×256 patches, refine details
Phase 3: 384×384 patches (if enough memory), fine textures
Features: Gradient accumulation, OneCycleLR, EMA, mixed precision where available.
"""

import sys
import os

# ─── Environment Setup (MUST be before torch import) ─────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "..", ".torch_cache")
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import json
import copy
import cv2

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.losses import CombinedLoss
from utils.dataloader import OnlineDegradationDataset, PairedDataset


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
    def __init__(self, model, decay=0.9999):
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
        # Resize all to same size for visualization
        target_h = 256
        d = cv2.resize(d, (target_h, target_h))
        o = cv2.resize(o, (target_h, target_h))
        c = cv2.resize(c, (target_h, target_h))
        row = np.concatenate([d, o, c], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    grid_bgr = cv2.cvtColor((np.clip(grid, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_dir}/epoch_{epoch:03d}.jpg", grid_bgr)


# ─── Training Phase ──────────────────────────────────────

def run_phase(phase_name, encoder, decoder, dslt, criterion, device,
              train_loader, val_loader, epochs, lr, ema, model_wrap, all_params,
              history, best_state, grad_accum_steps=4):
    """Run a single training phase."""

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))

    # OneCycleLR for efficient convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    print(f"\n{'='*70}")
    print(f"Phase: {phase_name} | {epochs} epochs | LR={lr} | accum={grad_accum_steps}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ──
        encoder.train(); decoder.train(); dslt.train()
        train_loss_acc = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (degraded, clean) in enumerate(train_loader):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            bottleneck, skips = encoder(degraded)
            bn_t, skips_t = dslt(bottleneck, skips)
            correction = decoder(bn_t, skips_t)
            output = torch.clamp(degraded + correction, 0, 1)

            total_loss, _, _, _, _ = criterion(output, clean)
            loss = total_loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            train_loss_acc += total_loss.detach()
            n_batches += 1

        # Sync MPS
        if device.type == "mps":
            torch.mps.synchronize()

        avg_train = (train_loss_acc / n_batches).item()
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

                total_loss, _, _, _, _ = criterion(output, clean)
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
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)
        history["lr"].append(lr_now)

        total_epoch = len(history["train_loss"])
        print(f"E[{total_epoch:3d}] T:{avg_train:.4f} V:{avg_val:.4f} "
              f"PSNR:{avg_psnr:.2f} SSIM:{avg_ssim:.4f} LR:{lr_now:.2e} {elapsed:.0f}s")

        # Save best model
        if avg_psnr > best_state["best_psnr"] or avg_val < best_state["best_loss"]:
            best_state["best_psnr"] = max(best_state["best_psnr"], avg_psnr)
            best_state["best_loss"] = min(best_state["best_loss"], avg_val)
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'dslt': dslt.state_dict(),
                'epoch': total_epoch,
                'val_loss': best_state["best_loss"],
                'val_psnr': best_state["best_psnr"],
                'val_ssim': avg_ssim,
                'version': 'v5',
            }, "results/best_model_v5.pth")
            # Save EMA
            ema_state = {}
            for name, mod in [('encoder', encoder), ('decoder', decoder), ('dslt', dslt)]:
                ema_state[name] = {k: v for k, v in ema.shadow.items() if k.startswith(name + '.')}
            torch.save(ema.shadow, "results/ema_weights_v5.pth")
            print(f"  ✅ Saved (PSNR={best_state['best_psnr']:.2f})")

        # Visualization every 5 epochs
        if total_epoch % 5 == 0 and last_deg is not None:
            save_visualization(last_deg, last_out, last_clean, total_epoch)

    return history, best_state


# ─── Main Training ────────────────────────────────────────

def train():
    device = get_device()

    # ─── Model ────────────────────────────────────────────
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt = MultiScaleDSLT().to(device)

    total_params = sum(p.numel() for m in [encoder, decoder, dslt] for p in m.parameters())
    print(f"Model params: {total_params:,}")

    # ─── Loss (PSNR-maximizing mode) ──────────────────────
    criterion = CombinedLoss(device, mode='psnr')
    print(f"Loss mode: PSNR (w_percep={criterion.w_perceptual}, w_ssim={criterion.w_ssim})")
    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(dslt.parameters())

    # ─── EMA ──────────────────────────────────────────────
    class ModelWrapper(nn.Module):
        def __init__(self, enc, dec, d):
            super().__init__()
            self.encoder = enc
            self.decoder = dec
            self.dslt = d
    model_wrap = ModelWrapper(encoder, decoder, dslt)
    ema = EMA(model_wrap, decay=0.9999)

    # ─── State ────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": []}
    best_state = {"best_psnr": 0.0, "best_loss": float("inf")}

    # ─── Check for resume ─────────────────────────────────
    if os.path.exists("results/best_model_v5.pth"):
        ckpt = torch.load("results/best_model_v5.pth", map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        dslt.load_state_dict(ckpt['dslt'])
        best_state["best_psnr"] = ckpt.get('val_psnr', 0.0)
        best_state["best_loss"] = ckpt.get('val_loss', float("inf"))
        print(f"📂 Resumed v5 checkpoint (PSNR={best_state['best_psnr']:.2f})")

    # ─── Data source ──────────────────────────────────────
    clean_dir = "data/train_clean"
    if not os.path.exists(clean_dir) or len(os.listdir(clean_dir)) == 0:
        print("❌ No clean images found. Run utils/degradation.py first or place images in data/train_clean/")
        sys.exit(1)

    # ==========================================================
    # PHASE 1: 128×128 patches — Learn coarse structure
    # ==========================================================
    print("\n" + "▓"*70)
    print("▓  PHASE 1: 128×128 — Coarse Structure Learning")
    print("▓"*70)

    dataset_p1 = OnlineDegradationDataset(clean_dir, crop_size=128, augment=True, severity='mild')
    train_size = int(0.9 * len(dataset_p1))
    val_size = len(dataset_p1) - train_size
    train_set, val_set = random_split(dataset_p1, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    batch_size_p1 = 16 if device.type == "mps" else 8
    train_loader = DataLoader(train_set, batch_size=batch_size_p1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size_p1, shuffle=False, num_workers=0)
    print(f"Data: {train_size} train | {val_size} val | batch={batch_size_p1}")

    history, best_state = run_phase(
        "128×128 Coarse", encoder, decoder, dslt, criterion, device,
        train_loader, val_loader, epochs=100, lr=3e-4, ema=ema,
        model_wrap=model_wrap, all_params=all_params, history=history,
        best_state=best_state, grad_accum_steps=2
    )

    # ==========================================================
    # PHASE 2: 256×256 patches — Refine details
    # ==========================================================
    print("\n" + "▓"*70)
    print("▓  PHASE 2: 256×256 — Detail Refinement")
    print("▓"*70)

    dataset_p2 = OnlineDegradationDataset(clean_dir, crop_size=256, augment=True, severity='mild')
    train_set2, val_set2 = random_split(dataset_p2, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))

    batch_size_p2 = 8 if device.type == "mps" else 4
    train_loader2 = DataLoader(train_set2, batch_size=batch_size_p2, shuffle=True, num_workers=0)
    val_loader2 = DataLoader(val_set2, batch_size=batch_size_p2, shuffle=False, num_workers=0)
    print(f"Data: {train_size} train | {val_size} val | batch={batch_size_p2}")

    history, best_state = run_phase(
        "256×256 Detail", encoder, decoder, dslt, criterion, device,
        train_loader2, val_loader2, epochs=150, lr=1.5e-4, ema=ema,
        model_wrap=model_wrap, all_params=all_params, history=history,
        best_state=best_state, grad_accum_steps=4
    )

    # ==========================================================
    # PHASE 3: 384×384 patches — Fine textures + sharpness
    # ==========================================================
    print("\n" + "▓"*70)
    print("▓  PHASE 3: 384×384 — Fine Texture Sharpening")
    print("▓"*70)

    dataset_p3 = OnlineDegradationDataset(clean_dir, crop_size=384, augment=True, severity='mild')
    train_set3, val_set3 = random_split(dataset_p3, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))

    batch_size_p3 = 4 if device.type == "mps" else 2
    train_loader3 = DataLoader(train_set3, batch_size=batch_size_p3, shuffle=True, num_workers=0)
    val_loader3 = DataLoader(val_set3, batch_size=batch_size_p3, shuffle=False, num_workers=0)
    print(f"Data: {train_size} train | {val_size} val | batch={batch_size_p3}")

    history, best_state = run_phase(
        "384×384 Sharpen", encoder, decoder, dslt, criterion, device,
        train_loader3, val_loader3, epochs=100, lr=5e-5, ema=ema,
        model_wrap=model_wrap, all_params=all_params, history=history,
        best_state=best_state, grad_accum_steps=8
    )

    # ─── Save History + Plots ─────────────────────────────
    with open("results/training_history_v5.json", "w") as f:
        json.dump(history, f)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Losses
        axes[0, 0].plot(history["train_loss"], label="Train", alpha=0.8)
        axes[0, 0].plot(history["val_loss"], label="Val", alpha=0.8)
        axes[0, 0].set_title("Loss (Charbonnier + Multi-Loss)")
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel("Epoch")

        # PSNR
        axes[0, 1].plot(history["val_psnr"], color='green')
        axes[0, 1].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Target: 30 dB')
        axes[0, 1].axhline(y=23.69, color='orange', linestyle='--', alpha=0.5, label='v4 baseline: 23.69 dB')
        axes[0, 1].set_title("Val PSNR (dB)")
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel("Epoch")

        # SSIM
        axes[1, 0].plot(history["val_ssim"], color='blue')
        axes[1, 0].axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target: 0.85')
        axes[1, 0].set_title("Val SSIM")
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlabel("Epoch")

        # LR
        axes[1, 1].plot(history["lr"], color='orange')
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel("Step")

        # Phase boundaries
        phase1_end = 100
        phase2_end = 100 + 150
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=phase2_end, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.savefig("results/training_curves_v5.png", dpi=150)
        print("📊 Training curves saved")
    except Exception as e:
        print(f"⚠️  Could not save plots: {e}")

    print(f"\n🚀 Done! Best PSNR={best_state['best_psnr']:.2f} dB")
    print(f"   Model: results/best_model_v5.pth")
    print(f"   EMA:   results/ema_weights_v5.pth")


if __name__ == "__main__":
    train()