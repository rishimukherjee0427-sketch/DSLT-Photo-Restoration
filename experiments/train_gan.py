"""
DSLT v5 GAN Fine-Tuning — Phase 4: Adversarial Sharpening
Loads the PSNR-optimized checkpoint and fine-tunes with a PatchGAN discriminator.
The discriminator is OPTIONAL and can be toggled off at any time.

Usage:
    python experiments/train_gan.py                    # Default: 50 epochs, discriminator ON
    python experiments/train_gan.py --epochs 30        # Custom epoch count
    python experiments/train_gan.py --no-gan           # Run without GAN (sanity check)
    python experiments/train_gan.py --multi-scale      # Use 2-scale discriminator
"""

import sys
import os
import argparse

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
import numpy as np
import time
import json
import gc
import cv2

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.discriminator import get_discriminator
from models.losses import CombinedLoss
from utils.dataloader import OnlineDegradationDataset


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


# ─── Metrics ──────────────────────────────────────────────

def compute_psnr_batch(output, target):
    """Batch PSNR computation — stays on device."""
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

def save_visualization(degraded, output, clean, epoch, save_dir="results/visualizations_gan"):
    """Save side-by-side comparison."""
    os.makedirs(save_dir, exist_ok=True)
    n = min(4, degraded.size(0))
    rows = []
    for i in range(n):
        d = degraded[i].cpu().permute(1, 2, 0).numpy()
        o = output[i].cpu().permute(1, 2, 0).numpy()
        c = clean[i].cpu().permute(1, 2, 0).numpy()
        target_h = 256
        d = cv2.resize(d, (target_h, target_h))
        o = cv2.resize(o, (target_h, target_h))
        c = cv2.resize(c, (target_h, target_h))
        row = np.concatenate([d, o, c], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    grid_bgr = cv2.cvtColor((np.clip(grid, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_dir}/gan_epoch_{epoch:03d}.jpg", grid_bgr)


# ─── GAN Fine-Tuning ─────────────────────────────────────

def train_gan(args):
    device = get_device()

    # ─── Load Pre-Trained Generator ───────────────────────
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt = MultiScaleDSLT().to(device)

    # Find best checkpoint
    ckpt_path = None
    for p in ["results/best_model_v5.pth", "results/best_model_v5_gan.pth", "results/best_model.pth"]:
        if os.path.exists(p):
            ckpt_path = p
            break

    base_psnr = 0.0
    base_epoch = 0

    if ckpt_path is None:
        print("⚠️  No pre-trained model found — training GAN from scratch (random init)")
        print("   For better results, run experiments/train.py first!")
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            encoder.load_state_dict(ckpt['encoder'])
            decoder.load_state_dict(ckpt['decoder'])
            dslt.load_state_dict(ckpt['dslt'])
            base_psnr = ckpt.get('val_psnr', 0.0)
            base_epoch = ckpt.get('epoch', '?')
            print(f"📂 Loaded generator from {ckpt_path}")
            print(f"   Base PSNR: {base_psnr:.2f} dB | Epoch: {base_epoch}")
        except RuntimeError as e:
            print(f"⚠️  Checkpoint {ckpt_path} incompatible with current v5 architecture")
            print(f"   Error: {str(e)[:150]}")
            print(f"   Training from scratch (random init) instead!")
            ckpt_path = None

    gen_params = sum(p.numel() for m in [encoder, decoder, dslt] for p in m.parameters())
    print(f"   Generator params: {gen_params:,}")

    # ─── Discriminator ────────────────────────────────────
    use_disc = not args.no_gan
    discriminator = None

    if use_disc:
        discriminator = get_discriminator(
            multi_scale=args.multi_scale,
            device=device
        )

    # ─── Loss ─────────────────────────────────────────────
    criterion = CombinedLoss(
        device,
        use_discriminator=use_disc,
        discriminator=discriminator,
        mode='gan' if use_disc else 'psnr'
    )

    # ─── Optimizers (TTUR: discriminator LR > generator LR) ──
    gen_params_list = (
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(dslt.parameters())
    )

    # Low LR for generator — we're fine-tuning, not training from scratch
    optimizer_G = torch.optim.AdamW(
        gen_params_list, lr=args.lr_g,
        weight_decay=1e-4, betas=(0.9, 0.99)
    )

    optimizer_D = None
    scheduler_D = None
    if use_disc and discriminator is not None:
        # Higher LR for discriminator (TTUR: Two Time-scale Update Rule)
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=args.lr_d,
            betas=(0.0, 0.9)  # β1=0 is standard for discriminator
        )

    # ─── Data ─────────────────────────────────────────────
    clean_dir = "data/train_clean"
    if not os.path.exists(clean_dir) or len(os.listdir(clean_dir)) == 0:
        print("❌ No clean images found. Place images in data/train_clean/")
        sys.exit(1)

    dataset = OnlineDegradationDataset(clean_dir, crop_size=args.crop_size, augment=True, severity='mild')

    # Limit dataset size for MPS memory constraints
    total_images = len(dataset)
    if args.max_images > 0 and args.max_images < total_images:
        indices = list(range(args.max_images))
        dataset = torch.utils.data.Subset(dataset, indices)
        total_images = len(dataset)
        print(f"   Using {total_images}/{len(indices)} images (--max-images)")

    train_size = int(0.9 * total_images)
    val_size = total_images - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"📊 Data: {train_size} train | {val_size} val | batch={batch_size} | crop={args.crop_size}")

    # ─── Schedulers ───────────────────────────────────────
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, T_max=args.epochs, eta_min=1e-7
    )
    if optimizer_D is not None:
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_D, T_max=args.epochs, eta_min=1e-7
        )

    # ─── State ────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/visualizations_gan", exist_ok=True)

    history = {
        "train_loss_g": [], "train_loss_d": [],
        "val_loss": [], "val_psnr": [], "val_ssim": [],
        "d_real_acc": [], "d_fake_acc": [],
        "lr_g": [], "lr_d": [],
    }
    best_psnr = 0.0
    freeze_encoder_epochs = args.freeze_encoder

    print(f"\n{'='*70}")
    print(f"GAN Fine-Tuning | {args.epochs} epochs | GAN={'ON' if use_disc else 'OFF'}")
    print(f"Generator LR: {args.lr_g} | Discriminator LR: {args.lr_d}")
    if freeze_encoder_epochs > 0:
        print(f"Encoder frozen for first {freeze_encoder_epochs} epochs")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Freeze/Unfreeze encoder ──
        if epoch <= freeze_encoder_epochs:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
        elif epoch == freeze_encoder_epochs + 1:
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad = True
            print("  🔓 Encoder unfrozen")

        # ── Train ──
        if epoch > freeze_encoder_epochs:
            encoder.train()
        decoder.train()
        dslt.train()
        if discriminator is not None:
            discriminator.train()

        g_loss_acc = 0.0
        d_loss_acc = 0.0
        d_real_acc_acc = 0.0
        d_fake_acc_acc = 0.0
        n_batches = 0

        for batch_idx, (degraded, clean) in enumerate(train_loader):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # ── Step 1: Generator Forward (for D update) ──
            with torch.no_grad():
                bottleneck, skips = encoder(degraded)
                bn_t, skips_t = dslt(bottleneck, skips)
                correction = decoder(bn_t, skips_t)
                fake_for_d = torch.clamp(degraded + correction, 0, 1).detach()

            # ── Step 2: Update Discriminator ──
            d_loss_val = 0.0
            d_r_acc = 0.0
            d_f_acc = 0.0

            if use_disc and discriminator is not None and optimizer_D is not None:
                optimizer_D.zero_grad(set_to_none=True)
                d_loss, d_r_acc, d_f_acc = criterion.discriminator_loss(clean, fake_for_d)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
                d_loss_val = d_loss.item()
                del d_loss

            del fake_for_d, bottleneck, skips, bn_t, skips_t, correction

            # ── Step 3: Update Generator ──
            optimizer_G.zero_grad(set_to_none=True)

            bottleneck, skips = encoder(degraded)
            bn_t, skips_t = dslt(bottleneck, skips)
            correction = decoder(bn_t, skips_t)
            output = torch.clamp(degraded + correction, 0, 1)

            total_loss, loss_char, loss_p, loss_ssim, loss_adv = criterion(output, clean)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params_list, max_norm=1.0)
            optimizer_G.step()

            g_loss_acc += total_loss.item()
            d_loss_acc += d_loss_val
            d_real_acc_acc += d_r_acc
            d_fake_acc_acc += d_f_acc
            n_batches += 1

            del total_loss, loss_char, loss_p, loss_ssim, loss_adv
            del bottleneck, skips, bn_t, skips_t, correction, output
            del degraded, clean

            # Periodic MPS memory cleanup
            if device.type == "mps" and (batch_idx + 1) % 50 == 0:
                torch.mps.synchronize()
                torch.mps.empty_cache()

        # Sync + clear MPS cache between epochs
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        gc.collect()

        scheduler_G.step()
        if scheduler_D is not None:
            scheduler_D.step()

        avg_g_loss = g_loss_acc / max(n_batches, 1)
        avg_d_loss = d_loss_acc / max(n_batches, 1)
        avg_d_real = d_real_acc_acc / max(n_batches, 1)
        avg_d_fake = d_fake_acc_acc / max(n_batches, 1)

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

                # Validate without GAN loss (pure pixel metrics)
                loss_char = torch.mean(torch.sqrt((output - clean) ** 2 + 1e-6))
                val_loss_acc += loss_char.item()
                val_psnr_acc += compute_psnr_batch(output, clean).item()
                val_ssim_acc += compute_ssim_batch(output, clean).item()
                val_batches += 1

                last_deg, last_out, last_clean = degraded, output, clean

        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        gc.collect()

        avg_val = val_loss_acc / max(val_batches, 1)
        avg_psnr = val_psnr_acc / max(val_batches, 1)
        avg_ssim = val_ssim_acc / max(val_batches, 1)
        lr_g_now = optimizer_G.param_groups[0]['lr']
        lr_d_now = optimizer_D.param_groups[0]['lr'] if optimizer_D else 0.0
        elapsed = time.time() - t0

        # Record history
        history["train_loss_g"].append(avg_g_loss)
        history["train_loss_d"].append(avg_d_loss)
        history["val_loss"].append(avg_val)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)
        history["d_real_acc"].append(avg_d_real)
        history["d_fake_acc"].append(avg_d_fake)
        history["lr_g"].append(lr_g_now)
        history["lr_d"].append(lr_d_now)

        # Print progress
        gan_info = ""
        if use_disc:
            gan_info = f" D:{avg_d_loss:.4f} Racc:{avg_d_real:.2f} Facc:{avg_d_fake:.2f}"
        print(f"E[{epoch:3d}] G:{avg_g_loss:.4f}{gan_info} "
              f"PSNR:{avg_psnr:.2f} SSIM:{avg_ssim:.4f} "
              f"LRg:{lr_g_now:.2e} {elapsed:.0f}s", flush=True)

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_dict = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'dslt': dslt.state_dict(),
                'epoch': epoch,
                'val_psnr': best_psnr,
                'val_ssim': avg_ssim,
                'version': 'v5-gan',
                'base_checkpoint': ckpt_path,
                'use_discriminator': use_disc,
            }
            if discriminator is not None:
                save_dict['discriminator'] = discriminator.state_dict()
            torch.save(save_dict, "results/best_model_v5_gan.pth")
            print(f"  ✅ Saved best GAN model (PSNR={best_psnr:.2f})")

        # Visualization every 5 epochs
        if epoch % 5 == 0 and last_deg is not None:
            save_visualization(last_deg, last_out, last_clean, epoch)

    # ─── Save History + Plots ─────────────────────────────
    with open("results/training_history_v5_gan.json", "w") as f:
        json.dump(history, f)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Generator Loss
        axes[0, 0].plot(history["train_loss_g"], label="Generator", alpha=0.8, color='blue')
        axes[0, 0].set_title("Generator Loss")
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel("Epoch")

        # Discriminator Loss
        axes[0, 1].plot(history["train_loss_d"], label="Discriminator", alpha=0.8, color='red')
        axes[0, 1].set_title("Discriminator Loss")
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel("Epoch")

        # PSNR
        axes[0, 2].plot(history["val_psnr"], color='green')
        axes[0, 2].axhline(y=base_psnr, color='orange', linestyle='--', alpha=0.5,
                           label=f'PSNR-only baseline: {base_psnr:.2f} dB')
        axes[0, 2].set_title("Val PSNR (dB)")
        axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlabel("Epoch")

        # SSIM
        axes[1, 0].plot(history["val_ssim"], color='blue')
        axes[1, 0].set_title("Val SSIM")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlabel("Epoch")

        # Discriminator Accuracy
        axes[1, 1].plot(history["d_real_acc"], label="Real Acc", alpha=0.8, color='green')
        axes[1, 1].plot(history["d_fake_acc"], label="Fake Acc", alpha=0.8, color='red')
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Equilibrium')
        axes[1, 1].set_title("Discriminator Accuracy")
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylim(0, 1)

        # Learning Rates
        axes[1, 2].plot(history["lr_g"], label="Generator LR", color='blue')
        axes[1, 2].plot(history["lr_d"], label="Discriminator LR", color='red')
        axes[1, 2].set_title("Learning Rate Schedule")
        axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xlabel("Epoch")

        plt.suptitle("DSLT v5 — GAN Fine-Tuning", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/training_curves_v5_gan.png", dpi=150)
        print("📊 GAN training curves saved")
    except Exception as e:
        print(f"⚠️  Could not save plots: {e}")

    print(f"\n🚀 GAN Fine-Tuning Complete!")
    print(f"   Best PSNR: {best_psnr:.2f} dB (base was {base_psnr:.2f} dB)")
    print(f"   Model: results/best_model_v5_gan.pth")
    print(f"   History: results/training_history_v5_gan.json")
    print(f"   Curves: results/training_curves_v5_gan.png")


# ─── CLI ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DSLT v5 GAN Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of GAN fine-tuning epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--crop-size", type=int, default=128,
                        help="Training crop size (default: 128, use 256 for CUDA)")
    parser.add_argument("--max-images", type=int, default=500,
                        help="Max training images per epoch (default: 500, 0=all)")
    parser.add_argument("--lr-g", type=float, default=1e-5,
                        help="Generator learning rate (default: 1e-5)")
    parser.add_argument("--lr-d", type=float, default=4e-5,
                        help="Discriminator learning rate (default: 4e-5)")
    parser.add_argument("--no-gan", action="store_true",
                        help="Disable discriminator (sanity check mode)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Use 2-scale discriminator (~1.6M params)")
    parser.add_argument("--freeze-encoder", type=int, default=5,
                        help="Freeze encoder for N epochs (default: 5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_gan(args)
