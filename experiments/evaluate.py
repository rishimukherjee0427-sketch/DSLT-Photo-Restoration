"""
DSLT v3 Evaluation — Ablation Study with MPS support.
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
import lpips

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from utils.dataloader import PairedDataset


def evaluate():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    if not os.path.exists("results/best_model.pth"):
        print("❌ No model. Train first."); sys.exit(1)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt_model = MultiScaleDSLT().to(device)

    ckpt = torch.load("results/best_model.pth", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    dslt_model.load_state_dict(ckpt['dslt'])
    encoder.eval(); decoder.eval(); dslt_model.eval()

    dataset = PairedDataset("data/train_clean", "data/train_degraded", augment=False)
    N = min(100, len(dataset))
    os.makedirs("results/comparisons", exist_ok=True)

    # LPIPS perceptual metric
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    metrics = {
        "unet_baseline": {"psnr": [], "ssim": [], "lpips": []},
        "dslt_bottleneck_only": {"psnr": [], "ssim": [], "lpips": []},
        "dslt_multiscale": {"psnr": [], "ssim": [], "lpips": []},
    }

    print(f"Evaluating {N} images on {device}...")

    with torch.no_grad():
        for i in range(N):
            degraded, clean = dataset[i]
            deg_batch = degraded.unsqueeze(0).to(device)
            clean_np = clean.permute(1, 2, 0).numpy()
            degraded_np = degraded.permute(1, 2, 0).numpy()

            b, skips = encoder(deg_batch)

            # (A) U-Net baseline
            corr_a = decoder(b, skips)
            out_a = torch.clamp(deg_batch + corr_a, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            # (B) Bottleneck-only DSLT
            bn_t, skips_t = dslt_model(b, skips)
            corr_b = decoder(bn_t, skips)
            out_b = torch.clamp(deg_batch + corr_b, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            # (C) Full multi-scale DSLT
            corr_c = decoder(bn_t, skips_t)
            out_c = torch.clamp(deg_batch + corr_c, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            for name, out_np in [("unet_baseline", out_a), ("dslt_bottleneck_only", out_b), ("dslt_multiscale", out_c)]:
                metrics[name]["psnr"].append(psnr(clean_np, out_np, data_range=1.0))
                metrics[name]["ssim"].append(ssim(clean_np, out_np, data_range=1.0, channel_axis=2))
                
                # Compute LPIPS (needs NHWC -> NCHW tensor bounded [-1, 1])
                out_t = torch.from_numpy(out_np).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
                clean_t = torch.from_numpy(clean_np).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
                lpips_val = loss_fn_vgg(out_t, clean_t).item()
                metrics[name]["lpips"].append(lpips_val)

            if i < 10:
                row = np.concatenate([degraded_np, out_a, out_c, clean_np], axis=1)
                cv2.imwrite(f"results/comparisons/compare_{i:02d}.jpg",
                           cv2.cvtColor((row * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    results = {}
    labels = {"unet_baseline": "U-Net Baseline", "dslt_bottleneck_only": "Single DSLT",
              "dslt_multiscale": "Multi-Scale DSLT"}
    print(f"\n{'='*75}")
    print(f"{'Model':<25} {'PSNR ↑':>8} {'SSIM ↑':>8} {'LPIPS ↓':>8}")
    print(f"{'-'*60}")
    for name, vals in metrics.items():
        r = {
            "psnr_mean": float(np.mean(vals["psnr"])), 
            "ssim_mean": float(np.mean(vals["ssim"])),
            "lpips_mean": float(np.mean(vals["lpips"]))
        }
        results[name] = r
        print(f"{labels[name]:<25} {r['psnr_mean']:>8.2f} {r['ssim_mean']:>8.4f} {r['lpips_mean']:>8.4f}")
    print(f"{'='*75}")

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 results/metrics.json | 🖼️  results/comparisons/")


if __name__ == "__main__":
    evaluate()
import torch
import lpips
import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms

# ========================= CONFIG =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose backbone: 'alex' is faster and often better for forward scores; 'vgg' is more traditional
loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device).eval()

# Transform: Important - LPIPS expects images in [-1, 1] range
transform = transforms.Compose([
    transforms.ToTensor(),                    # [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1,1]
])

# Paths - CHANGE THESE to your folders
restored_dir = "results/restored"      # folder with your model outputs
gt_dir       = "data/clean_raw/BnW vs Colored/Colored"   # or wherever your ground-truth colored images are

# Make sure filenames match between restored and GT
# =========================================================

def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

def compute_metrics(restored_path, gt_path):
    img_rest = load_image(restored_path)
    img_gt   = load_image(gt_path)
    
    # LPIPS (lower is better)
    lpips_val = loss_fn(img_rest, img_gt).item()
    
    # Convert to numpy [0,255] for PSNR/SSIM
    rest_np = (img_rest.squeeze(0).cpu().numpy().transpose(1,2,0) * 0.5 + 0.5) * 255.0
    gt_np   = (img_gt.squeeze(0).cpu().numpy().transpose(1,2,0) * 0.5 + 0.5) * 255.0
    
    rest_np = np.clip(rest_np, 0, 255).astype(np.uint8)
    gt_np   = np.clip(gt_np, 0, 255).astype(np.uint8)
    
    # PSNR
    psnr_val = psnr(gt_np, rest_np, data_range=255)
    
    # SSIM (multichannel for color)
    ssim_val = ssim(gt_np, rest_np, channel_axis=2, data_range=255)
    
    return psnr_val, ssim_val, lpips_val

# ========================= MAIN =========================
if __name__ == "__main__":
    restored_files = sorted([f for f in os.listdir(restored_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    
    print(f"Evaluating {len(restored_files)} images...\n")
    
    for fname in restored_files:
        restored_path = os.path.join(restored_dir, fname)
        gt_path       = os.path.join(gt_dir, fname)   # assuming same filename
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {fname}")
            continue
            
        p, s, l = compute_metrics(restored_path, gt_path)
        
        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)
        
        print(f"{fname:30} | PSNR: {p:6.2f} | SSIM: {s:.4f} | LPIPS: {l:.4f}")
    
    # Final averages
    print("\n" + "="*60)
    print(f"Average PSNR : {np.mean(psnr_list):.2f}")
    print(f"Average SSIM : {np.mean(ssim_list):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_list):.4f}   ← Lower is better")
    print("="*60)