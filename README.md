# DSLT v5 — Multi-Scale Deep Space Latent Translation

Advanced vintage photo restoration using a deep residual U-Net with RRDB blocks, 4-scale DSLT with dilated convolutions, and progressive multi-phase training.

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

## Architecture

- **Encoder**: 4-level (3→64→128→256→512) with strided conv + CBAM attention
- **Bottleneck**: 4× RRDB (Residual-in-Residual Dense Blocks) + CBAM
- **Multi-Scale DSLT**: Latent translation at e2 (4 blocks) + e3 (6 blocks) + e4 (6 blocks) + bottleneck (8 blocks)
- **Decoder**: PixelShuffle upsampling + skip connections + learnable residual scaling
- **Loss**: Charbonnier + 0.05×MultiPerceptual + 0.15×SSIM + 0.05×LAB + 0.10×FFT + 0.05×Edge

## Features

- 🖼️ Upload & restore vintage/degraded/B&W photos
- 🔄 Before/after comparison slider
- 🎨 7 post-processing filters: Vivid, Warm, Cool, B&W, Sharpen, Colorize
- ⬇️ HD download
- 🚀 MPS-accelerated (Apple Silicon) training & inference
- 🔮 Self-ensemble TTA for maximum quality

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

# Train v5 (progressive 3-phase: ~8-12 hours on M4 Pro)
python experiments/train.py

# Evaluate
python experiments/evaluate.py
```

## Run

```bash
python app.py
# Open http://127.0.0.1:8080
```

## Project Structure

```
├── app.py                    # Flask backend (v5-aware)
├── models/
│   ├── autoencoder.py        # Deep U-Net (RRDB + PixelShuffle + CBAM)
│   ├── latent_translator.py  # 4-scale DSLT with dilated conv
│   ├── losses.py             # Charbonnier + FFT + MultiPerceptual + SSIM + LAB + Edge
│   ├── filters.py            # Post-processing filters
│   └── inference.py          # Self-ensemble TTA + tile-based inference
├── experiments/
│   ├── train.py              # Progressive 3-phase training
│   └── evaluate.py           # Ablation study (Baseline/Single/Multi/TTA)
├── utils/
│   ├── dataloader.py         # Online degradation + random crop + augmentation
│   └── degradation.py        # Synthetic degradation pipeline
├── templates/index.html      # Frontend
├── static/css/style.css      # Styling
└── results/                  # Models, metrics, visualizations
```
