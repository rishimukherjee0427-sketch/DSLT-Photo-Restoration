# DSLT v3 — Multi-Scale Deep Space Latent Translation

Advanced vintage photo restoration using a multi-scale U-Net architecture with Deep Space Latent Translation (DSLT), CBAM attention, and residual learning.

## Architecture

- **Encoder**: 3-level U-Net (32→64→128→256) with CBAM attention
- **Multi-Scale DSLT**: Latent translation at bottleneck (8 blocks) + encoder level 3 (4 blocks)
- **Decoder**: Skip connections + tanh-bounded residual corrections
- **Loss**: L1 + 0.1×Perceptual (VGG16) + 0.05×SSIM
- **Parameters**: 12.7M

## Features

- 🖼️ Upload & restore vintage/degraded/B&W photos
- 🔄 Before/after comparison slider
- 🎨 7 post-processing filters: Vivid, Warm, Cool, B&W, Sharpen, Colorize
- ⬇️ HD download
- 🚀 MPS-accelerated (Apple Silicon) training & inference

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Generate training pairs
python utils/degradation.py

# Train (auto-detects MPS/CUDA/CPU)
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
├── app.py                    # Flask backend
├── models/
│   ├── autoencoder.py        # U-Net encoder + decoder (CBAM)
│   ├── latent_translator.py  # Multi-scale DSLT
│   ├── losses.py             # L1 + Perceptual + SSIM
│   └── filters.py            # Post-processing filters
├── experiments/
│   ├── train.py              # MPS-optimized training
│   └── evaluate.py           # PSNR/SSIM ablation
├── utils/
│   ├── dataloader.py         # Dataset + augmentation
│   └── degradation.py        # Synthetic degradation pipeline
├── templates/index.html      # Frontend
├── static/css/style.css      # Styling
└── results/                  # Models, metrics, visualizations
```
