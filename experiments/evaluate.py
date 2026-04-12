"""
DSLT v5 Evaluation — Ablation Study with Self-Ensemble TTA.
"""

import sys
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "..", ".torch_cache")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
import cv2
import json
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.inference import restore_image
from utils.dataloader import PairedDataset


def evaluate():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # Try v5 checkpoint first, fall back to v4
    ckpt_path = "results/best_model_v5.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "results/best_model.pth"
    if not os.path.exists(ckpt_path):
        print("❌ No model. Train first."); sys.exit(1)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt_model = MultiScaleDSLT().to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    dslt_model.load_state_dict(ckpt['dslt'])
    encoder.eval(); decoder.eval(); dslt_model.eval()
    print(f"✅ Loaded: {ckpt_path} (v{ckpt.get('version', '4')})")

    dataset = PairedDataset("data/train_clean", "data/train_degraded", augment=False)
    N = min(100, len(dataset))
    os.makedirs("results/comparisons", exist_ok=True)

    # Try to load LPIPS (optional)
    try:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        has_lpips = True
    except ImportError:
        print("⚠️  lpips not installed, skipping LPIPS metric")
        has_lpips = False

    metrics = {
        "unet_baseline": {"psnr": [], "ssim": [], "lpips": []},
        "dslt_single": {"psnr": [], "ssim": [], "lpips": []},
        "dslt_multiscale": {"psnr": [], "ssim": [], "lpips": []},
        "dslt_multiscale_tta": {"psnr": [], "ssim": [], "lpips": []},
    }

    print(f"Evaluating {N} images on {device}...")

    with torch.no_grad():
        for i in range(N):
            degraded, clean = dataset[i]
            deg_batch = degraded.unsqueeze(0).to(device)
            clean_np = clean.permute(1, 2, 0).numpy()
            degraded_np = degraded.permute(1, 2, 0).numpy()

            b, skips = encoder(deg_batch)

            # (A) U-Net baseline (no DSLT)
            corr_a = decoder(b, skips)
            out_a = torch.clamp(deg_batch + corr_a, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            # (B) Single-scale DSLT (bottleneck only)
            bn_t, skips_t = dslt_model(b, skips)
            corr_b = decoder(bn_t, skips)
            out_b = torch.clamp(deg_batch + corr_b, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            # (C) Full multi-scale DSLT
            corr_c = decoder(bn_t, skips_t)
            out_c = torch.clamp(deg_batch + corr_c, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            # (D) Multi-scale DSLT + Self-Ensemble TTA (only for first 20 — slow)
            if i < 20:
                out_d_tensor = restore_image(
                    encoder, decoder, dslt_model,
                    degraded.unsqueeze(0), device,
                    use_tta=True, use_tiles=False
                )
                out_d = out_d_tensor.squeeze(0).permute(1, 2, 0).numpy()
            else:
                out_d = out_c  # Skip TTA for speed

            for name, out_np in [
                ("unet_baseline", out_a),
                ("dslt_single", out_b),
                ("dslt_multiscale", out_c),
                ("dslt_multiscale_tta", out_d if i < 20 else None),
            ]:
                if out_np is None:
                    continue
                metrics[name]["psnr"].append(psnr(clean_np, np.clip(out_np, 0, 1), data_range=1.0))
                metrics[name]["ssim"].append(ssim(clean_np, np.clip(out_np, 0, 1), data_range=1.0, channel_axis=2))

                if has_lpips:
                    out_t = torch.from_numpy(out_np).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
                    clean_t = torch.from_numpy(clean_np).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
                    lpips_val = loss_fn_vgg(out_t, clean_t).item()
                    metrics[name]["lpips"].append(lpips_val)

            if i < 10:
                row = np.concatenate([degraded_np, out_a, out_c, clean_np], axis=1)
                cv2.imwrite(f"results/comparisons/compare_{i:02d}.jpg",
                           cv2.cvtColor((np.clip(row, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{N}] processed")

    results = {}
    labels = {
        "unet_baseline": "U-Net Baseline",
        "dslt_single": "Single-Scale DSLT",
        "dslt_multiscale": "Multi-Scale DSLT",
        "dslt_multiscale_tta": "Multi-Scale + TTA",
    }
    print(f"\n{'='*75}")
    header = f"{'Model':<25} {'PSNR ↑':>8} {'SSIM ↑':>8}"
    if has_lpips:
        header += f" {'LPIPS ↓':>8}"
    print(header)
    print(f"{'-'*65}")
    for name, vals in metrics.items():
        if len(vals["psnr"]) == 0:
            continue
        r = {
            "psnr_mean": float(np.mean(vals["psnr"])),
            "ssim_mean": float(np.mean(vals["ssim"])),
        }
        if has_lpips and len(vals["lpips"]) > 0:
            r["lpips_mean"] = float(np.mean(vals["lpips"]))
        results[name] = r
        line = f"{labels[name]:<25} {r['psnr_mean']:>8.2f} {r['ssim_mean']:>8.4f}"
        if "lpips_mean" in r:
            line += f" {r['lpips_mean']:>8.4f}"
        print(line)
    print(f"{'='*75}")

    with open("results/metrics_v5.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 results/metrics_v5.json | 🖼️  results/comparisons/")


if __name__ == "__main__":
    evaluate()