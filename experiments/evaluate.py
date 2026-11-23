"""
DSLT v5 Evaluation — Ablation Study & Model Comparison
=======================================================
Evaluates and compares multiple model variants side-by-side:
  (A) Baseline v5 (PSNR-optimized)
  (B) Baseline v5 + EMA weights
  (C) GAN-sharpened v5
  (D) U-Net only (no DSLT — architectural ablation)
  (E) Single-scale DSLT (bottleneck only — architectural ablation)
  (F) Full pipeline + Self-Ensemble TTA

Supports CLI-driven metric selection and checkpoint paths.

Usage:
  python experiments/evaluate.py                                          # Full ablation
  python experiments/evaluate.py --model results/best_model.pth           # Single model
  python experiments/evaluate.py --metrics psnr ssim lpips                 # Select metrics
  python experiments/evaluate.py --num-images 50 --no-tta                  # Quick run
  python experiments/evaluate.py --export-csv results/ablation.csv         # CSV export
"""

import sys
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "..", ".torch_cache")
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import torch
import numpy as np
import cv2
import json
import time
import gc
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.inference import restore_image
from utils.dataloader import OnlineDegradationDataset


# ═══════════════════════════════════════════════════════════
#  Device
# ═══════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        print("✅ Device: Apple MPS GPU")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("✅ Device: CUDA GPU")
        return torch.device("cuda")
    print("⚠️  Device: CPU")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════
#  Model Loading Helpers
# ═══════════════════════════════════════════════════════════

def load_model(ckpt_path, device):
    """Load encoder/decoder/dslt from a checkpoint. Returns (encoder, decoder, dslt, meta)."""
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt = MultiScaleDSLT().to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    dslt.load_state_dict(ckpt['dslt'])
    encoder.eval(); decoder.eval(); dslt.eval()

    meta = {
        'version': ckpt.get('version', '?'),
        'epoch': ckpt.get('epoch', '?'),
        'val_psnr': ckpt.get('val_psnr', 0.0),
        'val_ssim': ckpt.get('val_ssim', 0.0),
    }
    return encoder, decoder, dslt, meta


def load_ema_model(base_ckpt_path, ema_path, device):
    """
    Load EMA weights into a model. EMA weights are stored as a flat
    state_dict with keys like 'encoder.xxx', 'decoder.xxx', 'dslt.xxx'.
    """
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    dslt = MultiScaleDSLT().to(device)

    # Load base checkpoint first (for meta info)
    base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=True)

    # Load EMA shadow weights
    ema_shadow = torch.load(ema_path, map_location=device, weights_only=True)

    # EMA shadow stores keys as 'encoder.enc1.conv.0.weight' etc.
    enc_state, dec_state, dslt_state = {}, {}, {}
    for key, val in ema_shadow.items():
        if key.startswith('encoder.'):
            enc_state[key[len('encoder.'):]] = val
        elif key.startswith('decoder.'):
            dec_state[key[len('decoder.'):]] = val
        elif key.startswith('dslt.'):
            dslt_state[key[len('dslt.'):]] = val

    if enc_state:
        encoder.load_state_dict(enc_state)
    else:
        # Fallback: EMA file might just be base checkpoint format
        encoder.load_state_dict(base_ckpt['encoder'])

    if dec_state:
        decoder.load_state_dict(dec_state)
    else:
        decoder.load_state_dict(base_ckpt['decoder'])

    if dslt_state:
        dslt.load_state_dict(dslt_state)
    else:
        dslt.load_state_dict(base_ckpt['dslt'])

    encoder.eval(); decoder.eval(); dslt.eval()

    meta = {
        'version': base_ckpt.get('version', '?') + '+EMA',
        'epoch': base_ckpt.get('epoch', '?'),
        'val_psnr': base_ckpt.get('val_psnr', 0.0),
    }
    return encoder, decoder, dslt, meta


# ═══════════════════════════════════════════════════════════
#  Inference Modes
# ═══════════════════════════════════════════════════════════

def infer_full(encoder, decoder, dslt, deg_batch, device):
    """Full multi-scale DSLT inference."""
    b, skips = encoder(deg_batch)
    bn_t, skips_t = dslt(b, skips)
    correction = decoder(bn_t, skips_t)
    return torch.clamp(deg_batch + correction, 0, 1)


def infer_unet_only(encoder, decoder, dslt, deg_batch, device):
    """U-Net baseline: no DSLT translation (architectural ablation)."""
    b, skips = encoder(deg_batch)
    correction = decoder(b, skips)
    return torch.clamp(deg_batch + correction, 0, 1)


def infer_single_dslt(encoder, decoder, dslt, deg_batch, device):
    """Single-scale DSLT: translate bottleneck only, keep original skips."""
    b, skips = encoder(deg_batch)
    bn_t, _ = dslt(b, skips)
    correction = decoder(bn_t, skips)
    return torch.clamp(deg_batch + correction, 0, 1)


def infer_tta(encoder, decoder, dslt, deg_batch, device):
    """Full DSLT + Self-Ensemble TTA (8× augmentation average)."""
    out = restore_image(
        encoder, decoder, dslt,
        deg_batch.cpu(), device,
        use_tta=True, use_tiles=False
    )
    return out.to(device)


# ═══════════════════════════════════════════════════════════
#  Metric Computation
# ═══════════════════════════════════════════════════════════

def compute_metrics(output_np, clean_np, metrics_list, lpips_fn=None, device=None):
    """Compute requested metrics for a single image pair."""
    results = {}
    output_np = np.clip(output_np, 0, 1)

    if 'psnr' in metrics_list:
        results['psnr'] = float(calc_psnr(clean_np, output_np, data_range=1.0))

    if 'ssim' in metrics_list:
        results['ssim'] = float(calc_ssim(clean_np, output_np, data_range=1.0, channel_axis=2))

    if 'lpips' in metrics_list and lpips_fn is not None and device is not None:
        out_t = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
        cln_t = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
        results['lpips'] = float(lpips_fn(out_t, cln_t).item())

    return results


# ═══════════════════════════════════════════════════════════
#  Main Ablation Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate(args):
    device = get_device()
    metrics_list = [m.lower() for m in args.metrics]

    # ── LPIPS setup ───────────────────────────────────────
    lpips_fn = None
    if 'lpips' in metrics_list:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='vgg').to(device)
            print("✅ LPIPS (VGG) loaded")
        except ImportError:
            print("⚠️  lpips not installed — skipping LPIPS metric")
            print("   Install with: pip install lpips")
            metrics_list = [m for m in metrics_list if m != 'lpips']

    # ── Discover checkpoints ─────────────────────────────
    # Model variants to evaluate: (label, ckpt_path, inference_fn, description)
    variants = OrderedDict()

    # Auto-discover available checkpoints
    base_paths = [
        ("results/best_model_v5.pth", "Baseline v5"),
        ("results/best_model.pth", "Baseline (legacy)"),
    ]
    gan_path = "results/best_model_v5_gan.pth"
    ema_paths = [
        "results/ema_weights_v5.pth",
        "results/ema_weights.pth",
    ]

    # If user specified a single model, just evaluate that
    if args.model:
        if not os.path.exists(args.model):
            print(f"❌ Checkpoint not found: {args.model}")
            sys.exit(1)
        single_path = args.model
    else:
        single_path = None

    # ── Build variant list ────────────────────────────────
    base_ckpt = None  # The primary PSNR-optimized checkpoint

    if single_path:
        # Single-model mode: evaluate one checkpoint in all architectural modes
        base_ckpt = single_path
        variants["full_dslt"] = {
            "label": "Full Multi-Scale DSLT",
            "ckpt": single_path,
            "infer": infer_full,
            "desc": "Full pipeline from specified checkpoint",
        }
        if not args.no_architectural:
            variants["unet_only"] = {
                "label": "U-Net Only (no DSLT)",
                "ckpt": single_path,
                "infer": infer_unet_only,
                "desc": "Ablation: decoder without latent translation",
            }
            variants["single_dslt"] = {
                "label": "Single-Scale DSLT",
                "ckpt": single_path,
                "infer": infer_single_dslt,
                "desc": "Ablation: bottleneck-only translation",
            }
        if not args.no_tta:
            variants["full_dslt_tta"] = {
                "label": "Full DSLT + TTA",
                "ckpt": single_path,
                "infer": infer_tta,
                "desc": "Self-ensemble (8× augmentation average)",
            }
    else:
        # Full ablation: compare all available checkpoints
        for bp, label in base_paths:
            if os.path.exists(bp):
                base_ckpt = bp
                # (A) Full pipeline — baseline
                variants["baseline_full"] = {
                    "label": f"{label} (Full DSLT)",
                    "ckpt": bp,
                    "infer": infer_full,
                    "desc": "PSNR-optimized, full multi-scale pipeline",
                }
                if not args.no_architectural:
                    # (D) U-Net only
                    variants["baseline_unet"] = {
                        "label": f"{label} (U-Net Only)",
                        "ckpt": bp,
                        "infer": infer_unet_only,
                        "desc": "Ablation: no latent translation",
                    }
                    # (E) Single-scale DSLT
                    variants["baseline_single"] = {
                        "label": f"{label} (Single DSLT)",
                        "ckpt": bp,
                        "infer": infer_single_dslt,
                        "desc": "Ablation: bottleneck-only translation",
                    }
                break  # Use first found base checkpoint

        # (B) EMA variant
        if base_ckpt:
            for ep in ema_paths:
                if os.path.exists(ep):
                    variants["ema_full"] = {
                        "label": "v5 + EMA Weights",
                        "ckpt": base_ckpt,
                        "ema_path": ep,
                        "infer": infer_full,
                        "desc": "Exponential Moving Average smoothed weights",
                    }
                    break

        # (C) GAN-sharpened
        if os.path.exists(gan_path):
            variants["gan_full"] = {
                "label": "v5 + GAN Sharpening",
                "ckpt": gan_path,
                "infer": infer_full,
                "desc": "Adversarial fine-tuned for texture/sharpness",
            }

        # (F) TTA on best model
        if base_ckpt and not args.no_tta:
            variants["baseline_tta"] = {
                "label": "Baseline + TTA (×8)",
                "ckpt": base_ckpt,
                "infer": infer_tta,
                "desc": "Self-ensemble: +0.3-0.5dB free PSNR boost",
            }

    if not variants:
        print("❌ No checkpoints found! Train a model first:")
        print("   python experiments/train.py")
        sys.exit(1)

    # ── Print plan ────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  DSLT v5 — Ablation Evaluation")
    print(f"{'═'*72}")
    print(f"  Metrics:   {', '.join(m.upper() for m in metrics_list)}")
    print(f"  Variants:  {len(variants)}")
    for key, v in variants.items():
        print(f"    • {v['label']:<30s}  [{v['ckpt']}]")
    print(f"  Images:    {args.num_images}")
    print(f"  Crop Size: {args.crop_size}")
    print(f"{'═'*72}\n")

    # ── Load validation data ─────────────────────────────
    clean_dir = "data/train_clean"
    if not os.path.exists(clean_dir) or len(os.listdir(clean_dir)) == 0:
        print("❌ No clean images in data/train_clean/")
        sys.exit(1)

    # Use OnlineDegradationDataset with a fixed seed for reproducible ablation
    torch.manual_seed(123)
    np.random.seed(123)

    dataset = OnlineDegradationDataset(clean_dir, crop_size=args.crop_size, augment=False)
    N = min(args.num_images, len(dataset))
    print(f"📊 Evaluating on {N} images (crop={args.crop_size}×{args.crop_size})\n")

    # ── Pre-load all validation pairs (ensures same data for all variants) ──
    print("Loading validation pairs...", end=" ", flush=True)
    val_pairs = []
    for i in range(N):
        degraded, clean = dataset[i]
        val_pairs.append((degraded, clean))
    print(f"Done ({N} pairs)\n")

    # ── Evaluate each variant ─────────────────────────────
    all_results = OrderedDict()
    all_outputs = {}  # For visual comparison
    os.makedirs("results/comparisons", exist_ok=True)

    for var_key, var_info in variants.items():
        label = var_info['label']
        print(f"▸ Evaluating: {label}")
        t0 = time.time()

        # Load model for this variant
        try:
            if 'ema_path' in var_info:
                enc, dec, dslt_m, meta = load_ema_model(
                    var_info['ckpt'], var_info['ema_path'], device
                )
            else:
                enc, dec, dslt_m, meta = load_model(var_info['ckpt'], device)
        except Exception as e:
            print(f"  ⚠️  Failed to load: {e}")
            continue

        infer_fn = var_info['infer']
        per_image_metrics = {m: [] for m in metrics_list}
        var_outputs = []

        with torch.no_grad():
            for i, (degraded, clean) in enumerate(val_pairs):
                deg_batch = degraded.unsqueeze(0).to(device)
                clean_np = clean.permute(1, 2, 0).numpy()

                # Run inference
                output = infer_fn(enc, dec, dslt_m, deg_batch, device)
                output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()

                # Compute metrics
                m = compute_metrics(output_np, clean_np, metrics_list, lpips_fn, device)
                for metric_name, val in m.items():
                    per_image_metrics[metric_name].append(val)

                # Store first few outputs for visual comparison
                if i < 10:
                    var_outputs.append(output_np)

                if (i + 1) % 25 == 0:
                    psnr_so_far = np.mean(per_image_metrics.get('psnr', [0]))
                    print(f"    [{i+1:3d}/{N}] PSNR={psnr_so_far:.2f}", flush=True)

                # Memory cleanup
                del deg_batch, output
                if device.type == "mps" and (i + 1) % 50 == 0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()

        elapsed = time.time() - t0

        # Aggregate results
        result = {}
        for metric_name in metrics_list:
            vals = per_image_metrics[metric_name]
            if vals:
                result[f"{metric_name}_mean"] = float(np.mean(vals))
                result[f"{metric_name}_std"] = float(np.std(vals))
                result[f"{metric_name}_median"] = float(np.median(vals))
                result[f"{metric_name}_min"] = float(np.min(vals))
                result[f"{metric_name}_max"] = float(np.max(vals))
        result["time_seconds"] = elapsed
        result["time_per_image"] = elapsed / max(N, 1)
        result["description"] = var_info['desc']
        result["checkpoint"] = var_info['ckpt']

        all_results[var_key] = result
        all_outputs[var_key] = var_outputs

        # Print summary for this variant
        summary_parts = [f"{label:<30s}"]
        if 'psnr_mean' in result:
            summary_parts.append(f"PSNR={result['psnr_mean']:.2f}±{result.get('psnr_std', 0):.2f}")
        if 'ssim_mean' in result:
            summary_parts.append(f"SSIM={result['ssim_mean']:.4f}")
        if 'lpips_mean' in result:
            summary_parts.append(f"LPIPS={result['lpips_mean']:.4f}")
        summary_parts.append(f"({elapsed:.1f}s)")
        print(f"  ✅ {'  '.join(summary_parts)}")

        # Free model memory
        del enc, dec, dslt_m
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        gc.collect()

    # ══════════════════════════════════════════════════════
    #  Results Summary
    # ══════════════════════════════════════════════════════

    print(f"\n{'═'*80}")
    print(f"  ABLATION RESULTS")
    print(f"{'═'*80}")

    # Build header
    header = f"{'Model':<32s}"
    for m in metrics_list:
        arrow = '↑' if m in ('psnr', 'ssim') else '↓'
        header += f" {m.upper()+' '+arrow:>10s}"
    header += f" {'Time':>8s}"
    print(header)
    print(f"{'─'*80}")

    # Determine best values for highlighting
    best_vals = {}
    for m in metrics_list:
        key = f"{m}_mean"
        vals = [r[key] for r in all_results.values() if key in r]
        if vals:
            if m in ('psnr', 'ssim'):
                best_vals[m] = max(vals)
            else:  # lpips — lower is better
                best_vals[m] = min(vals)

    for var_key, result in all_results.items():
        label = variants[var_key]['label']
        line = f"  {label:<30s}"
        for m in metrics_list:
            key = f"{m}_mean"
            if key in result:
                val = result[key]
                is_best = abs(val - best_vals.get(m, val)) < 1e-6
                marker = " ★" if is_best else "  "
                line += f" {val:>8.4f}{marker}"
            else:
                line += f" {'N/A':>10s}"
        line += f" {result.get('time_seconds', 0):>7.1f}s"
        print(line)

    print(f"{'═'*80}")
    print(f"  ★ = best in column")

    # ── Improvement deltas ────────────────────────────────
    if 'psnr' in metrics_list and len(all_results) > 1:
        print(f"\n{'─'*80}")
        print(f"  PSNR Deltas (vs first variant):")
        base_key = list(all_results.keys())[0]
        base_psnr = all_results[base_key].get('psnr_mean', 0)
        for var_key, result in all_results.items():
            if var_key == base_key:
                continue
            delta = result.get('psnr_mean', 0) - base_psnr
            sign = '+' if delta >= 0 else ''
            label = variants[var_key]['label']
            print(f"    {label:<30s}  {sign}{delta:.3f} dB")
        print(f"{'─'*80}")

    # ── Save visual comparisons ──────────────────────────
    print("\n📸 Saving visual comparisons...")
    for i in range(min(10, N)):
        imgs = []
        # Degraded input
        degraded_np = val_pairs[i][0].permute(1, 2, 0).numpy()
        imgs.append(degraded_np)

        # Each variant's output
        for var_key in all_results.keys():
            if i < len(all_outputs.get(var_key, [])):
                imgs.append(all_outputs[var_key][i])

        # Ground truth
        clean_np = val_pairs[i][1].permute(1, 2, 0).numpy()
        imgs.append(clean_np)

        row = np.concatenate(imgs, axis=1)
        row_bgr = cv2.cvtColor(
            (np.clip(row, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(f"results/comparisons/ablation_{i:02d}.jpg", row_bgr)

    # ── Save JSON results ────────────────────────────────
    output_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_list,
        "num_images": N,
        "crop_size": args.crop_size,
        "variants": {},
    }
    for var_key, result in all_results.items():
        output_json["variants"][var_key] = {
            "label": variants[var_key]['label'],
            **result,
        }

    json_path = "results/ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"📄 {json_path}")

    # ── CSV export ────────────────────────────────────────
    if args.export_csv:
        csv_lines = []
        csv_header = ["variant", "label"] + \
                     [f"{m}_mean" for m in metrics_list] + \
                     [f"{m}_std" for m in metrics_list] + \
                     ["time_seconds"]
        csv_lines.append(",".join(csv_header))

        for var_key, result in all_results.items():
            row = [var_key, f"\"{variants[var_key]['label']}\""]
            for m in metrics_list:
                row.append(f"{result.get(f'{m}_mean', 'N/A')}")
            for m in metrics_list:
                row.append(f"{result.get(f'{m}_std', 'N/A')}")
            row.append(f"{result.get('time_seconds', 0):.2f}")
            csv_lines.append(",".join(row))

        with open(args.export_csv, "w") as f:
            f.write("\n".join(csv_lines) + "\n")
        print(f"📑 {args.export_csv}")

    # ── Generate comparison plot ─────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_metrics = len(metrics_list)
        fig, axes = plt.subplots(1, n_metrics + 1, figsize=(6 * (n_metrics + 1), 6))
        if n_metrics + 1 == 1:
            axes = [axes]

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4', '#795548']
        labels = [variants[k]['label'] for k in all_results.keys()]
        short_labels = []
        for lb in labels:
            # Shorten for plot readability
            lb = lb.replace("Baseline (legacy)", "Legacy")
            lb = lb.replace("Baseline v5 ", "")
            lb = lb.replace("v5 + ", "+")
            short_labels.append(lb)

        # Bar chart for each metric
        for idx, m in enumerate(metrics_list):
            ax = axes[idx]
            vals = [all_results[k].get(f"{m}_mean", 0) for k in all_results.keys()]
            stds = [all_results[k].get(f"{m}_std", 0) for k in all_results.keys()]
            bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)],
                          yerr=stds, capsize=4, alpha=0.85, edgecolor='white', linewidth=1.5)

            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)
            arrow = '↑ higher better' if m in ('psnr', 'ssim') else '↓ lower better'
            ax.set_title(f"{m.upper()} ({arrow})", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='y')
            ax.set_axisbelow(True)

        # Timing comparison
        ax = axes[-1]
        times = [all_results[k].get('time_per_image', 0) for k in all_results.keys()]
        bars = ax.bar(range(len(times)), times, color=colors[:len(times)],
                      alpha=0.85, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                    f'{val:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)
        ax.set_title("Time / Image (s) ↓", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_axisbelow(True)

        plt.suptitle("DSLT v5 — Ablation Study", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        plot_path = "results/ablation_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 {plot_path}")

    except Exception as e:
        print(f"⚠️  Could not save plots: {e}")

    # ── Final Summary ─────────────────────────────────────
    print(f"\n🚀 Ablation complete!")
    print(f"   📄 Results:      results/ablation_results.json")
    print(f"   📊 Plot:         results/ablation_comparison.png")
    print(f"   🖼️  Comparisons:  results/comparisons/ablation_*.jpg")
    if args.export_csv:
        print(f"   📑 CSV:          {args.export_csv}")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="DSLT v5 — Ablation Evaluation & Model Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/evaluate.py                                    # Full ablation
  python experiments/evaluate.py --model results/best_model.pth     # Single checkpoint
  python experiments/evaluate.py --metrics psnr ssim lpips          # All metrics
  python experiments/evaluate.py --num-images 20 --no-tta           # Quick test
  python experiments/evaluate.py --export-csv results/ablation.csv  # CSV export
        """
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a single checkpoint to evaluate (default: auto-discover all)")
    parser.add_argument("--metrics", nargs='+', default=["psnr", "ssim", "lpips"],
                        choices=["psnr", "ssim", "lpips"],
                        help="Metrics to compute (default: psnr ssim lpips)")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of validation images (default: 100)")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Validation crop size (default: 256)")
    parser.add_argument("--no-tta", action="store_true",
                        help="Skip TTA variant (much faster)")
    parser.add_argument("--no-architectural", action="store_true",
                        help="Skip U-Net/single-DSLT ablation variants")
    parser.add_argument("--export-csv", type=str, default=None,
                        help="Export results to CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)