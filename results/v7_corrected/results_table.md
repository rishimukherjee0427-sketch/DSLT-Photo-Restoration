**Test split (250 images), mean ± SD over 2 seeds (123, 2026); corrected VGG normalisation; 800 training steps.** PSNR (dB) ↑ / SSIM ↑ / LPIPS ↓

| Model | Params (M) | Mild PSNR | Mild SSIM | Mild LPIPS | Medium PSNR | Medium SSIM | Medium LPIPS | Severe PSNR | Severe SSIM | Severe LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A (ungated) | 44.41 | 26.71 ± 0.14 | 0.8026 ± 0.0003 | 0.2220 ± 0.0020 | 24.11 ± 0.01 | 0.7465 ± 0.0006 | 0.3355 ± 0.0005 | 18.85 ± 0.23 | 0.5976 ± 0.0033 | 0.5749 ± 0.0044 |
| DSLT-B (adaptive gating) | 46.06 | 26.59 ± 0.28 | 0.8017 ± 0.0002 | 0.2233 ± 0.0025 | 23.97 ± 0.14 | 0.7445 ± 0.0005 | 0.3379 ± 0.0020 | 18.82 ± 0.01 | 0.5978 ± 0.0006 | 0.5772 ± 0.0014 |
| DSLT-C (gating + severity sup. + sparsity) | 46.06 | 26.69 ± 0.18 | 0.8020 ± 0.0008 | 0.2231 ± 0.0020 | 24.08 ± 0.08 | 0.7458 ± 0.0006 | 0.3371 ± 0.0010 | 18.92 ± 0.14 | 0.5997 ± 0.0029 | 0.5767 ± 0.0017 |
