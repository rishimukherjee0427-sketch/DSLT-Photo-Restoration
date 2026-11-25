## Test results (250 images; mean ± SD over seeds 42, 123, 2026; 20k steps at 256², best-val checkpoint)

| Model | Mild PSNR | Mild SSIM | Mild LPIPS | Medium PSNR | Medium SSIM | Medium LPIPS | Severe PSNR | Severe SSIM | Severe LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A full (44.4M) | 28.87 ± 1.47 | 0.8537 ± 0.0412 | 0.1670 ± 0.0424 | 26.53 ± 1.61 | 0.8003 ± 0.0428 | 0.2620 ± 0.0554 | 21.54 ± 2.15 | 0.6593 ± 0.0488 | 0.4767 ± 0.0759 |
| Encoder-decoder (22.5M) | 29.65 ± 0.14 | 0.8775 ± 0.0013 | 0.1446 ± 0.0064 | 27.38 ± 0.11 | 0.8238 ± 0.0016 | 0.2342 ± 0.0067 | 22.71 ± 0.10 | 0.6843 ± 0.0026 | 0.4425 ± 0.0066 |

Per seed (PSNR / SSIM / LPIPS):

| Model | Seed | Mild | Medium | Severe |
|---|---:|---|---|---|
| DSLT-A full (44.4M) | 42 | 29.74 / 0.8771 / 0.1407 | 27.51 / 0.8254 / 0.2271 | 22.82 / 0.6897 / 0.4264 |
| DSLT-A full (44.4M) | 123 | 29.71 / 0.8779 / 0.1443 | 27.41 / 0.8247 / 0.2331 | 22.74 / 0.6852 / 0.4397 |
| DSLT-A full (44.4M) | 2026 | 27.17 / 0.8061 / 0.2159 | 24.67 / 0.7509 / 0.3258 | 19.06 / 0.6030 / 0.5640 |
| Encoder-decoder (22.5M) | 42 | 29.79 / 0.8789 / 0.1374 | 27.43 / 0.8255 / 0.2265 | 22.78 / 0.6861 / 0.4352 |
| Encoder-decoder (22.5M) | 123 | 29.64 / 0.8771 / 0.1464 | 27.46 / 0.8223 / 0.2375 | 22.76 / 0.6814 / 0.4477 |
| Encoder-decoder (22.5M) | 2026 | 29.52 / 0.8764 / 0.1499 | 27.26 / 0.8237 / 0.2387 | 22.60 / 0.6854 / 0.4448 |

## Encoder-decoder − full, three seeds (pre-registered Stage-2 procedure; Holm over 9)

| Severity | Metric | Δ mean | per-seed Δ (42, 123, 2026) | seed SD | 95% CI (t, seeds) | 90% CI (2-level) | 95% CI (2-level) | d_z | % img Tfull better | p Holm | Verdict |
|---|---|---:|---|---:|---|---|---|---:|---:|---:|---|
| mild | PSNR | +0.778 | +0.050, -0.062, +2.347 | 1.3597 | [-2.599, +4.156] | [-0.059, +1.737] | [-0.084, +2.209] | +0.72 | 17% | 7.5e-25 ✓ | **non-inferior (not equivalent)** |
| mild | SSIM | +0.0238 | +0.0018, -0.0007, +0.0703 | 0.0403 | [-0.0763, +0.1239] | [-0.0003, +0.0538] | [-0.0007, +0.0661] | +0.65 | 22% | 1.8e-25 ✓ | **non-inferior (not equivalent)** |
| mild | LPIPS | -0.0224 | -0.0033, +0.0021, -0.0661 | 0.0379 | [-0.1166, +0.0717] | [-0.0520, +0.0009] | [-0.0609, +0.0018] | -0.52 | 27% | 5.1e-20 ✓ | **non-inferior (not equivalent)** |
| medium | PSNR | +0.852 | -0.080, +0.042, +2.594 | 1.5097 | [-2.898, +4.602] | [-0.074, +1.874] | [-0.096, +2.485] | +0.90 | 12% | 2.4e-31 ✓ | **non-inferior (not equivalent)** |
| medium | SSIM | +0.0235 | +0.0001, -0.0023, +0.0727 | 0.0427 | [-0.0825, +0.1295] | [-0.0020, +0.0544] | [-0.0024, +0.0691] | +0.70 | 12% | 1.8e-32 ✓ | **non-inferior (not equivalent)** |
| medium | LPIPS | -0.0278 | -0.0007, +0.0044, -0.0871 | 0.0514 | [-0.1556, +0.1000] | [-0.0631, +0.0033] | [-0.0830, +0.0042] | -0.90 | 13% | 1.5e-34 ✓ | **non-inferior (not equivalent)** |
| severe | PSNR | +1.172 | -0.039, +0.016, +3.539 | 2.0502 | [-3.921, +6.265] | [-0.057, +2.517] | [-0.079, +3.441] | +1.18 | 10% | 1.6e-36 ✓ | **non-inferior (not equivalent)** |
| severe | SSIM | +0.0250 | -0.0036, -0.0038, +0.0824 | 0.0497 | [-0.0985, +0.1485] | [-0.0043, +0.0577] | [-0.0046, +0.0796] | +1.10 | 4% | 8.6e-41 ✓ | **inconclusive** |
| severe | LPIPS | -0.0342 | +0.0088, +0.0080, -0.1192 | 0.0737 | [-0.2172, +0.1489] | [-0.0809, +0.0092] | [-0.1160, +0.0096] | -1.40 | 3% | 4.9e-41 ✓ | **inconclusive** |

## Per-seed image-level tests (each conditional on one training run; Holm over 9 within seed)

| Severity | Metric | Seed | Δ | 95% CI (img) | d_z | % img Tfull better | p Holm | Verdict (img-level) |
|---|---|---:|---:|---|---:|---:|---:|---|
| mild | PSNR | 42 | +0.050 | [-0.065, +0.170] | +0.05 | 45% | 1 | non-inferior (not equivalent) |
| mild | PSNR | 123 | -0.062 | [-0.164, +0.038] | -0.08 | 60% | 0.12 | inconclusive |
| mild | PSNR | 2026 | +2.347 | [+2.009, +2.685] | +0.85 | 13% | 2.3e-30 | non-inferior (not equivalent) |
| mild | SSIM | 42 | +0.0018 | [+0.0004, +0.0033] | +0.16 | 38% | 0.0022 | non-inferior (not equivalent) |
| mild | SSIM | 123 | -0.0007 | [-0.0018, +0.0003] | -0.09 | 58% | 0.11 | equivalent |
| mild | SSIM | 2026 | +0.0703 | [+0.0576, +0.0838] | +0.67 | 22% | 2.9e-27 | non-inferior (not equivalent) |
| mild | LPIPS | 42 | -0.0033 | [-0.0046, -0.0020] | -0.31 | 31% | 1.4e-07 | equivalent |
| mild | LPIPS | 123 | +0.0021 | [+0.0007, +0.0033] | +0.20 | 59% | 9.6e-05 | equivalent |
| mild | LPIPS | 2026 | -0.0661 | [-0.0824, -0.0507] | -0.53 | 32% | 6.3e-18 | non-inferior (not equivalent) |
| medium | PSNR | 42 | -0.080 | [-0.193, +0.034] | -0.09 | 53% | 0.79 | inconclusive |
| medium | PSNR | 123 | +0.042 | [-0.064, +0.156] | +0.05 | 54% | 1 | non-inferior (not equivalent) |
| medium | PSNR | 2026 | +2.594 | [+2.299, +2.892] | +1.09 | 8% | 8.9e-36 | non-inferior (not equivalent) |
| medium | SSIM | 42 | +0.0001 | [-0.0013, +0.0015] | +0.01 | 53% | 1 | equivalent |
| medium | SSIM | 123 | -0.0023 | [-0.0039, -0.0010] | -0.21 | 66% | 5.7e-05 | inconclusive |
| medium | SSIM | 2026 | +0.0727 | [+0.0612, +0.0851] | +0.75 | 10% | 2.1e-34 | non-inferior (not equivalent) |
| medium | LPIPS | 42 | -0.0007 | [-0.0022, +0.0008] | -0.05 | 45% | 1 | equivalent |
| medium | LPIPS | 123 | +0.0044 | [+0.0030, +0.0058] | +0.39 | 68% | 8.2e-10 | inconclusive |
| medium | LPIPS | 2026 | -0.0871 | [-0.0986, -0.0763] | -0.96 | 12% | 6.5e-35 | non-inferior (not equivalent) |
| severe | PSNR | 42 | -0.039 | [-0.148, +0.065] | -0.05 | 53% | 1 | inconclusive |
| severe | PSNR | 123 | +0.016 | [-0.082, +0.115] | +0.02 | 50% | 1 | equivalent |
| severe | PSNR | 2026 | +3.539 | [+3.235, +3.859] | +1.42 | 5% | 1.2e-40 | non-inferior (not equivalent) |
| severe | SSIM | 42 | -0.0036 | [-0.0054, -0.0019] | -0.25 | 65% | 2.4e-06 | worse beyond margin |
| severe | SSIM | 123 | -0.0038 | [-0.0052, -0.0024] | -0.33 | 73% | 5.1e-11 | worse beyond margin |
| severe | SSIM | 2026 | +0.0824 | [+0.0749, +0.0907] | +1.30 | 1% | 8.5e-42 | non-inferior (not equivalent) |
| severe | LPIPS | 42 | +0.0088 | [+0.0067, +0.0108] | +0.52 | 69% | 3.7e-13 | worse beyond margin |
| severe | LPIPS | 123 | +0.0080 | [+0.0060, +0.0100] | +0.50 | 70% | 8.6e-13 | worse beyond margin |
| severe | LPIPS | 2026 | -0.1192 | [-0.1281, -0.1112] | -1.78 | 0% | 8.3e-42 | non-inferior (not equivalent) |

## Training (validation-only selection)

| Model | Seed | Best epoch | Best val PSNR (mean 3 sev.) | Final val mild / medium / severe | Train s/step (median) | Wall-clock (min) | Peak train GB |
|---|---:|---:|---:|---|---:|---:|---:|
| DSLT-A full (44.4M) | 42 | 20 | 26.533 | 29.71 / 27.28 / 22.60 | 0.841 | 296 | 7.19 |
| DSLT-A full (44.4M) | 123 | 20 | 26.500 | 29.74 / 27.24 / 22.52 | 0.826 | 290 | 7.20 |
| DSLT-A full (44.4M) | 2026 | 1 | 23.608 | 26.16 / 23.82 / 18.47 | 0.828 | 291 | 7.20 |
| Encoder-decoder (22.5M) | 42 | 20 | 26.572 | 29.89 / 27.30 / 22.53 | 0.615 | 215 | 5.77 |
| Encoder-decoder (22.5M) | 123 | 20 | 26.450 | 29.70 / 27.17 / 22.48 | 0.612 | 214 | 5.76 |
| Encoder-decoder (22.5M) | 2026 | 20 | 26.275 | 29.46 / 27.02 / 22.35 | 0.622 | 219 | 5.77 |

## Learning-curve crossover (validation, Tfull − Tnone)

| Seed | Severe PSNR: Tfull ahead at step 1k? | Severe PSNR: sustained Tfull lead from step | Final Δ severe PSNR | Severe SSIM: sustained lead from | Final Δ severe SSIM | Final Δ mild PSNR | Final Δ medium PSNR |
|---:|---|---|---:|---|---:|---:|---:|
| 42 | yes | 9000 | +0.075 | 9000 | +0.0030 | -0.175 | -0.019 |
| 123 | yes | 9000 | +0.036 | 1000 | +0.0051 | +0.046 | +0.067 |
| 2026 | yes | never | -3.874 | never | -0.0975 | -3.293 | -3.195 |

## Efficiency (inference: existing Stage-3 profiling.json; training: per-run manifests, mean over seeds)

| Model | Params (M) | GMACs @256² | Latency ms @256² b1 | Throughput img/s @256² b4 | Train s/step | Train wall-clock (min) | Peak train GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A full (44.4M) | 44.41 | 76.3 | 57.2 | 24.8 | 0.832 | 293 | 7.19 |
| Encoder-decoder (22.5M) | 22.52 | 46.4 | 37.7 | 38.9 | 0.616 | 216 | 5.76 |
