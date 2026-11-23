# DSLT v6 — Multi-Scale Deep Space Latent Translation with Adaptive Gating

Advanced vintage photo restoration using a deep residual U-Net with RRDB blocks, 4-scale DSLT with dilated convolutions, **damage-conditioned adaptive gating**, and progressive multi-phase training.

## v5 Improvements (over v4)

| Change | v4 | v5 |
|---|---|---|
| **Encoder** | 3-level (32→64→128→256) | 4-level (64→128→256→512) + RRDB bottleneck |
| **Downsampling** | MaxPool2d | Strided convolutions |
| **Upsampling** | ConvTranspose2d | PixelShuffle (no checkerboard) |
| **Normalization** | BatchNorm | InstanceNorm |
| **DSLT** | 3-scale (e2+e3+bn) | 4-scale (e2+e3+e4+bn) with dilated conv |
| **Pixel Loss** | L1 | Charbonnier |
| **Perceptual Loss** | VGG single-layer | VGG multi-scale (4 layers) |
| **Frequency Loss** | ❌ | ✅ FFT magnitude + phase |
| **Edge Loss** | ❌ | ✅ Laplacian |
| **Training** | Single phase, 256×256 | Progressive 3-phase (128→256→384) |
| **Data** | Pre-generated pairs | Online degradation (infinite variety) |
| **Inference** | Single pass | Self-ensemble TTA (+0.3-0.5 dB) |
| **Parameters** | 12.7M | ~30M |

## v6 Improvements (over v5)

| Change | v5 | v6 |
|---|---|---|
| **Gating** | Uniform translation across all scales | Damage-conditioned adaptive gating per scale |
| **Damage Estimator** | ❌ | ✅ Lightweight CNN predicting 4-channel severity map |
| **Scale Gates** | Fixed (implicitly 1.0) | Per-scale base gates (e2, e3, e4, bottleneck) from a "rich" gate head, refined per channel and per pixel (FiLM-style; see Architecture) |
| **Translation** | `T(e)` | `e + G ⊙ (T(e) − e)`, with `G` a per-channel × per-pixel gate tensor |
| **Severity Map** | ❌ | ✅ GT: [blur, noise, scratch, fade] per pixel |
| **Aux Loss** | ❌ | ✅ L1 severity map loss (λ=0.15), plus optional gate supervision / sparsity / bimodality losses |
| **Ablation** | ❌ | ✅ Gated vs. ungated comparison (`experiments/run_ablation.py`) |
| **Parameters** | ~30M | 46.06M total (see breakdown below) |

> The original v6 scalar soft gate `g·T(e) + (1−g)·e` was found to be non-functional (see *Why the original gating did nothing*). It survives only as `MultiScaleDSLT(gate_mode='scalar')` for reproducing the legacy behaviour.

## Architecture

- **Encoder** (8.95M): 4-level with strided conv + CBAM attention, RRDB bottleneck
- **Damage Estimator** (1.56M, `models/damage_estimator.py`): lightweight encoder-decoder CNN predicting a (B,4,H,W) severity map [blur, noise, scratch, fade]. Its default `gate_head='rich'` predicts (B,4) base gates from avg+max-pooled e3 and e4 features plus avg+max of the detached predicted severity map, with the final bias initialised to 1.0 (gates start ≈ 0.73).
- **Multi-Scale DSLT** (21.98M, `models/latent_translator.py`, default `gate_mode='film'`): dilated residual translators `T` at e2, e3, e4 and the bottleneck. Per scale, the output is `e + G ⊙ (T(e) − e)`, where `G = σ(logit(g_scale) + Δ_channel + Δ_spatial)`:
  - `g_scale` is the base gate from the damage estimator.
  - `Δ_channel` is a squeeze-excitation-style per-channel offset computed from the pooled features and all 4 gates.
  - `Δ_spatial` is a per-pixel offset computed from the predicted severity map, resized to that scale.
  - Both offsets are zero-initialised (0.09M parameters in `gate_mods`), so at init `G` equals the scalar gate. The e1 skip is not gated. `use_gating=False` gives the ungated baseline (`G ≡ 1`).
- **Decoder** (13.57M): PixelShuffle upsampling + skip connections. The output is `clamp(x + correction, 0, 1)`.
- **Parameters:** total 46.06M (encoder + decoder + DSLT = 44.41M without the damage estimator and gate modulators).
- **Loss** (`CombinedLoss(mode='psnr')`): Charbonnier + 0.01×VGG16 multi-scale perceptual + 0.25×(1−SSIM) + 0.02×LAB + 0.05×FFT + 0.03×Edge + 0.15×damage-severity L1, plus optional `w_gate_sup·L_sup + w_gate_sparsity·L_sparse + w_gate_bimodal·L_bimodal`.
  - Since v7 (`vgg_normalize=True`, the default), VGG inputs are ImageNet mean/std-normalised. Models trained before this fix (`results/ablation/`, `results/baselines/`, all v5 checkpoints) used un-normalised VGG inputs.

## Features

- 🖼️ Upload & restore vintage/degraded/B&W photos
- 🔄 Before/after comparison slider
- 🎨 7 post-processing filters: Vivid, Warm, Cool, B&W, Sharpen, Colorize
- ⬇️ HD download
- 🚀 MPS-accelerated (Apple Silicon) training & inference
- 🔮 Self-ensemble TTA for maximum quality
- 🎯 Damage-conditioned adaptive gating (per-image, per-scale)
- 📊 Built-in ablation: gated vs. ungated comparison

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Generate clean image dataset (or place images in data/train_clean/)
python utils/degradation.py

# Train v6 (progressive 3-phase: ~8-12 hours on M4 Pro)
python experiments/train.py

# Evaluate
python experiments/evaluate.py
```

## Benchmark: Fair, Held-Out Comparison

All models below are scored on the **same 250 held-out test images**, with **identical cached degraded inputs** at three damage levels and one shared implementation of each metric. The metrics are PSNR (dB, per image, capped at 50 dB), SSIM (skimage) and LPIPS (AlexNet). ↑ means higher is better and ↓ means lower is better.

| Model | Params (M) | Mild PSNR↑ | Mild SSIM↑ | Mild LPIPS↓ | Medium PSNR↑ | Medium SSIM↑ | Medium LPIPS↓ | Severe PSNR↑ | Severe SSIM↑ | Severe LPIPS↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Degraded input (no restoration) | 0 | 25.60 | 0.7847 | 0.2482 | 21.99 | 0.7074 | 0.3866 | 15.25 | 0.5111 | 0.6844 |
| Bicubic (down/up ×2) | 0 | 24.81 | 0.7826 | 0.2802 | 21.69 | 0.7154 | 0.4041 | 15.32 | 0.5168 | 0.6795 |
| Restormer† | 3.75 | 26.17 | **0.8045** | **0.2214** | 23.79 | 0.7460 | **0.3259** | 18.01 | 0.5746 | **0.5751** |
| PromptIR† | 7.22 | 26.16 | **0.8076** | 0.2235 | 23.85 | **0.7493** | 0.3279 | 18.02 | 0.5755 | 0.5772 |
| DSLT, ungated (Config A) | 44.41 | 26.74 | 0.8021 | 0.2261 | 23.93 | 0.7430 | 0.3419 | 18.43 | 0.5898 | 0.5882 |
| DSLT, adaptive gating (Config B) | 46.06 | **26.81** | 0.8021 | 0.2251 | 23.98 | 0.7442 | 0.3417 | 18.42 | 0.5884 | 0.5874 |
| DSLT, gating + severity sup. + sparsity (Config C) | 46.06 | 26.78 | 0.8019 | 0.2258 | **23.99** | 0.7439 | 0.3417 | **18.45** | **0.5909** | 0.5865 |

† Reduced-size ("tiny") from-scratch re-implementations of Restormer (CVPR'22) and PromptIR (NeurIPS'23), not the official checkpoints. Restormer, PromptIR and DSLT Configs A–C use the same training protocol.

**Training protocol (identical for Restormer, PromptIR and DSLT A–C):**
- Trained only on `data/splits/train.txt` (4,279 images), with mixed mild/medium/severe online degradation.
- Batch 4 at 128 px crops, for **800 optimizer steps**.
- AdamW, lr 2e-4, cosine schedule with 50-step warmup, gradient clip 1.0.
- EMA 0.99, with the best epoch chosen by val PSNR.
- Same `CombinedLoss(mode='psnr')`.
- All compute ran on one Apple M4 GPU and took about 27 minutes in total. This is a short, equal-budget comparison, so absolute numbers are well below those of fully trained models.

**Contaminated legacy checkpoint:** the old `results/best_model_v5.pth` scores 29.90 / 0.8794 / 0.1295 on mild test images. It was trained on *all* of `data/train_clean`, including every val and test image, so it is excluded from the comparison. The earlier "30.16 dB" figure was measured on training images for the same reason.

### Key findings
1. **Gating ablation:** on the test set, adaptive gating (B) gives +0.07 dB (mild) / +0.05 dB (medium) / −0.01 dB (severe) over ungated (A). Gate supervision (C) gives +0.04 / +0.06 / +0.02 dB. These gains are within run-to-run noise at this budget. On the val subset during training B was +0.4 dB over A, but this did not carry over to test.
2. **DSLT vs. baselines:** every DSLT config beats both baselines on PSNR at all three levels: +0.6 dB on mild and +0.4 dB on severe. The baselines have lower (better) LPIPS, and SSIM is roughly tied. DSLT also has 6–12× more parameters.
3. **Severe damage remains hard:** every model reaches only about 18–18.5 dB, around 3 dB above the input.

### Why the original gating did nothing (diagnosis)
Diagnosis was run with `experiments/diagnose_gates.py`; outputs are in `results/gate_diagnosis/`.
- **No training signal:** the gate head received **zero gradient** from the severity loss and only a tiny one from the pixel loss. The gates collapsed to per-scale constants: their spread across images was 0.001–0.003, starting from a sigmoid init of about 0.5.
- **No per-image conditioning:** shuffling gates across images changed PSNR by **0.000 dB**. The gate acted only as a global scale, which is redundant with the translator's learnable residual scale.
- **Uninformative gate input:** the gate input (avg-pooled e4) could not predict damage severity (ridge-probe R² < 0, worse than predicting the mean).

**Fix** (in `models/damage_estimator.py`, `models/latent_translator.py` and `models/losses.py`; old checkpoints still load):
- **Richer gate head:** pooled e3 and e4 features plus the predicted severity map.
- **FiLM-style gating:** per-channel and per-pixel gates replace the single per-image scalar.
- **Optional gate losses:** severity supervision, sparsity and bimodality (`w_gate_sup`, `w_gate_sparsity`, `w_gate_bimodal`).

**Effect:** gates now track damage. Their spread across images is 0.13–0.20, and their correlation with ground-truth severity is 0.64–0.69, up from about 0.2. Config C reaches gate–target correlations of 0.66–0.78 on val. Gate logs are in `results/ablation/{A,B,C}_history.json` and `results/ablation/gate_stats.png`, and curves are in `results/ablation/ablation_curves.png`.

### Data split and severity levels
- **Deduplicated split:** `data/splits/` holds train/val/test = 4279/250/250. 221 near-duplicates were excluded, using flip-aware pHash + dHash clustering, so near-duplicate images cannot leak across splits.
- **Severity levels:** `mild`, `medium`, `severe` and `mixed` are defined in `utils/dataloader.py`. The full parameter table is in that file's header. `mild` and the legacy `heavy` level produce bit-identical output to the previous code.

### Reproduce
```bash
python experiments/make_benchmark.py --splits --cache            # cached degraded val/test inputs
for c in A B C; do python experiments/run_ablation.py --config $c --epochs 8 --steps_per_epoch 100 --warmup 50 --ema_decay 0.99 --val_n 24; done
for m in restormer promptir; do python experiments/train_baseline.py --model $m --size tiny --iters 800 --crop 128 --severity mixed --file_list data/splits/train.txt --warmup 50 --ema --ema_decay 0.99; done
python experiments/eval_all.py --split test --model identity --model bicubic --model restormer --model promptir \
       --model dslt_A_ungated --model dslt_B_gated --model dslt_C_gated_sup   # writes results/benchmark_table.{md,csv}
python experiments/plot_ablation.py
```

## GAN Fine-Tuning (Optional — Phase 4)

Fine-tune the PSNR-optimized model with a PatchGAN discriminator for sharper outputs.
The discriminator is **training-only** — inference uses the generator exclusively.

```bash
# Default: 50 epochs, discriminator ON
python experiments/train_gan.py

# Custom settings
python experiments/train_gan.py --epochs 30 --lr-g 1e-5 --lr-d 4e-5

# Use multi-scale discriminator (~1.6M params vs ~800K)
python experiments/train_gan.py --multi-scale

# Disable GAN (sanity check — fine-tune without adversarial loss)
python experiments/train_gan.py --no-gan
```

> **Note**: GAN training trades PSNR (~0.5-2 dB drop) for significantly sharper,
> more perceptually realistic outputs (lower LPIPS). The app auto-detects and
> prefers GAN checkpoints when available.

## Run

```bash
python app.py
# Open http://127.0.0.1:8080
```

## Project Structure

```
├── app.py                    # Flask backend (v6-aware, auto-detects GAN model)
├── models/
│   ├── autoencoder.py        # Deep U-Net (RRDB + PixelShuffle + CBAM) + DamageEstimator integration
│   ├── latent_translator.py  # 4-scale DSLT with dilated conv + adaptive soft gating
│   ├── damage_estimator.py   # Lightweight CNN: 4-ch severity map + 4 scale gates
│   ├── discriminator.py      # PatchGAN discriminator (training-only, ~800K params)
│   ├── losses.py             # Charbonnier + FFT + MultiPerceptual + SSIM + LAB + Edge + GAN + DamageSeverityL1
│   ├── filters.py            # Post-processing filters
│   └── inference.py          # Self-ensemble TTA + tile-based inference
├── experiments/
│   ├── train.py              # Progressive 3-phase training (PSNR-focused)
│   ├── train_gan.py          # Phase 4 GAN fine-tuning (sharpness-focused)
│   ├── evaluate.py           # Ablation study (Baseline/Single/Multi/TTA/GAN)
│   └── ablation_gated_vs_ungated.py  # Gated vs ungated comparison (Config A/B)
├── utils/
│   ├── dataloader.py         # Online degradation + random crop + augmentation + severity map
│   └── degradation.py        # Synthetic degradation pipeline + 4-ch severity map generation
├── templates/index.html      # Frontend
├── static/css/style.css      # Styling
└── results/                  # Models, metrics, visualizations, ablation_results.json
```
