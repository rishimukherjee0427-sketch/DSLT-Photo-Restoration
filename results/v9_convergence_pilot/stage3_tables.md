## Test results (250 images, seed 42, 20k steps at 256², best-val checkpoint)

| Model | Mild PSNR | Mild SSIM | Mild LPIPS | Medium PSNR | Medium SSIM | Medium LPIPS | Severe PSNR | Severe SSIM | Severe LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A full (44.4M) | 29.74 | 0.8771 | 0.1407 | 27.51 | 0.8254 | 0.2271 | 22.82 | 0.6897 | 0.4264 |
| Encoder-decoder, no translators (22.5M) | 29.79 | 0.8789 | 0.1374 | 27.43 | 0.8255 | 0.2265 | 22.78 | 0.6861 | 0.4352 |

## Encoder-decoder − full: image-level paired tests (n=250; ONE training seed; Holm over 9; pre-registered margins)

| Severity | Metric | Δ mean | 90% CI | 95% CI | d_z | % images better | p | p Holm | Verdict |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| mild | PSNR | +0.050 | [-0.047, +0.150] | [-0.065, +0.170] | +0.05 | 55% | 0.39 | 1 | **non-inferior (not equivalent)** |
| mild | SSIM | +0.0018 | [+0.0006, +0.0031] | [+0.0004, +0.0033] | +0.16 | 62% | 0.00036 | 0.0022 ✓ | **non-inferior (not equivalent)** |
| mild | LPIPS | -0.0033 | [-0.0044, -0.0022] | [-0.0046, -0.0020] | -0.31 | 69% | 1.7e-08 | 1.4e-07 ✓ | **equivalent** |
| medium | PSNR | -0.080 | [-0.175, +0.016] | [-0.193, +0.034] | -0.09 | 47% | 0.16 | 0.79 | **inconclusive** |
| medium | SSIM | +0.0001 | [-0.0011, +0.0013] | [-0.0013, +0.0015] | +0.01 | 47% | 0.41 | 1 | **equivalent** |
| medium | LPIPS | -0.0007 | [-0.0019, +0.0006] | [-0.0022, +0.0008] | -0.05 | 55% | 0.27 | 1 | **equivalent** |
| severe | PSNR | -0.039 | [-0.131, +0.048] | [-0.148, +0.065] | -0.05 | 47% | 0.52 | 1 | **inconclusive** |
| severe | SSIM | -0.0036 | [-0.0052, -0.0021] | [-0.0054, -0.0019] | -0.25 | 35% | 3.5e-07 | 2.4e-06 ✓ | **worse beyond margin** |
| severe | LPIPS | +0.0088 | [+0.0070, +0.0105] | [+0.0067, +0.0108] | +0.52 | 31% | 4.1e-14 | 3.7e-13 ✓ | **worse beyond margin** |

## Training

| Model | Best epoch (of 20) | Best val PSNR (mean of 3 sev.) | Final-epoch val PSNR | Train s/step (median epoch) | Wall-clock train+val (min) | Peak train MPS GB |
|---|---:|---:|---:|---:|---:|---:|
| DSLT-A full (44.4M) | 20 | 26.533 | 26.533 | 0.841 | 296 | 7.19 |
| Encoder-decoder, no translators (22.5M) | 20 | 26.572 | 26.572 | 0.615 | 215 | 5.77 |

## Inference cost (MPS FP32, median of 30 after 10 warm-up, one model per process)

| Model | Params (M) | GMACs @128² | GMACs @256² | Latency ms @256² b1 | Throughput img/s @256² b4 |
|---|---:|---:|---:|---:|---:|
| DSLT-A full (44.4M) | 44.41 | 19.1 | 76.3 | 57.2 | 24.8 |
| Encoder-decoder, no translators (22.5M) | 22.52 | 11.6 | 46.4 | 37.7 | 38.9 |
