## Test results (250 images; mean ± SD over seeds 42, 123, 2026; independently trained, 800 steps)

| Model | Mild PSNR | Mild SSIM | Mild LPIPS | Medium PSNR | Medium SSIM | Medium LPIPS | Severe PSNR | Severe SSIM | Severe LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A full (all translators) | 26.75 ± 0.20 | 0.8023 ± 0.0010 | 0.2207 ± 0.0023 | 24.17 ± 0.23 | 0.7461 ± 0.0023 | 0.3317 ± 0.0041 | 18.81 ± 0.30 | 0.5966 ± 0.0064 | 0.5755 ± 0.0065 |
| DSLT-e2 (e2 translator only) | 26.74 ± 0.12 | 0.8023 ± 0.0006 | 0.2216 ± 0.0034 | 24.08 ± 0.13 | 0.7446 ± 0.0015 | 0.3352 ± 0.0056 | 18.74 ± 0.12 | 0.5954 ± 0.0032 | 0.5798 ± 0.0061 |
| DSLT-EncoderDecoder (no translators) | 26.81 ± 0.17 | 0.8030 ± 0.0012 | 0.2189 ± 0.0044 | 24.25 ± 0.22 | 0.7466 ± 0.0026 | 0.3293 ± 0.0077 | 18.94 ± 0.36 | 0.5974 ± 0.0067 | 0.5747 ± 0.0108 |

Per seed (PSNR / SSIM / LPIPS):

| Model | Seed | Mild | Medium | Severe |
|---|---:|---|---|---|
| DSLT-A full (all translators) | 42 | 26.65 / 0.8014 / 0.2230 | 23.94 / 0.7436 / 0.3364 | 18.47 / 0.5895 / 0.5821 |
| DSLT-A full (all translators) | 123 | 26.63 / 0.8022 / 0.2208 | 24.17 / 0.7465 / 0.3290 | 19.04 / 0.6019 / 0.5692 |
| DSLT-A full (all translators) | 2026 | 26.98 / 0.8033 / 0.2184 | 24.41 / 0.7481 / 0.3297 | 18.91 / 0.5984 / 0.5753 |
| DSLT-e2 (e2 translator only) | 42 | 26.78 / 0.8026 / 0.2208 | 24.12 / 0.7456 / 0.3339 | 18.62 / 0.5917 / 0.5835 |
| DSLT-e2 (e2 translator only) | 123 | 26.61 / 0.8015 / 0.2253 | 23.93 / 0.7429 / 0.3413 | 18.76 / 0.5975 / 0.5832 |
| DSLT-e2 (e2 translator only) | 2026 | 26.85 / 0.8027 / 0.2188 | 24.18 / 0.7452 / 0.3304 | 18.85 / 0.5969 / 0.5728 |
| DSLT-EncoderDecoder (no translators) | 42 | 26.80 / 0.8020 / 0.2239 | 24.04 / 0.7445 / 0.3381 | 18.52 / 0.5897 / 0.5868 |
| DSLT-EncoderDecoder (no translators) | 123 | 26.64 / 0.8025 / 0.2168 | 24.23 / 0.7459 / 0.3241 | 19.13 / 0.6016 / 0.5660 |
| DSLT-EncoderDecoder (no translators) | 2026 | 26.98 / 0.8043 / 0.2161 | 24.47 / 0.7495 / 0.3256 | 19.17 / 0.6010 / 0.5714 |

## Pre-registered equivalence + difference tests (variant − full; margins PSNR ±0.10 dB, SSIM ±0.0020, LPIPS ±0.0050)

| Comparison | Severity | Metric | Δ mean | 90% CI (TOST) | 95% CI | d_z | per-seed Δ | p | p Holm (18) | Verdict |
|---|---|---|---:|---|---|---:|---|---:|---:|---|
| DSLT-e2 (e2 translator only) | mild | PSNR | -0.009 | [-0.134, +0.116] | [-0.156, +0.142] | -0.01 | +0.123, -0.020, -0.131 | 0.82 | 0.82 | **inconclusive** |
| DSLT-e2 (e2 translator only) | mild | SSIM | -0.0000 | [-0.0011, +0.0010] | [-0.0013, +0.0013] | -0.01 | +0.0012, -0.0006, -0.0007 | 0.16 | 0.63 | **equivalent** |
| DSLT-e2 (e2 translator only) | mild | LPIPS | +0.0009 | [-0.0017, +0.0039] | [-0.0021, +0.0045] | +0.13 | -0.0021, +0.0045, +0.0004 | 0.34 | 0.69 | **equivalent** |
| DSLT-e2 (e2 translator only) | medium | PSNR | -0.094 | [-0.278, +0.101] | [-0.298, +0.166] | -0.19 | +0.184, -0.243, -0.224 | 0.0058 | 0.047 ✓ | **inconclusive** |
| DSLT-e2 (e2 translator only) | medium | SSIM | -0.0015 | [-0.0039, +0.0010] | [-0.0042, +0.0018] | -0.23 | +0.0021, -0.0037, -0.0029 | 0.0039 | 0.043 ✓ | **inconclusive** |
| DSLT-e2 (e2 translator only) | medium | LPIPS | +0.0035 | [-0.0019, +0.0099] | [-0.0025, +0.0118] | +0.41 | -0.0025, +0.0123, +0.0007 | 1.6e-07 | 2.3e-06 ✓ | **inconclusive** |
| DSLT-e2 (e2 translator only) | severe | PSNR | -0.066 | [-0.243, +0.104] | [-0.278, +0.135] | -0.20 | +0.146, -0.286, -0.058 | 0.0023 | 0.03 ✓ | **inconclusive** |
| DSLT-e2 (e2 translator only) | severe | SSIM | -0.0012 | [-0.0040, +0.0015] | [-0.0044, +0.0020] | -0.19 | +0.0022, -0.0045, -0.0015 | 0.017 | 0.12 | **inconclusive** |
| DSLT-e2 (e2 translator only) | severe | LPIPS | +0.0043 | [-0.0017, +0.0108] | [-0.0023, +0.0134] | +0.58 | +0.0014, +0.0140, -0.0025 | 1.1e-15 | 2e-14 ✓ | **inconclusive** |
| DSLT-EncoderDecoder (no translators) | mild | PSNR | +0.057 | [-0.053, +0.156] | [-0.078, +0.175] | +0.09 | +0.147, +0.017, +0.006 | 0.21 | 0.63 | **non-inferior (not equivalent)** |
| DSLT-EncoderDecoder (no translators) | mild | SSIM | +0.0006 | [-0.0002, +0.0015] | [-0.0004, +0.0017] | +0.09 | +0.0006, +0.0003, +0.0010 | 0.0025 | 0.03 ✓ | **equivalent** |
| DSLT-EncoderDecoder (no translators) | mild | LPIPS | -0.0018 | [-0.0039, +0.0003] | [-0.0042, +0.0007] | -0.31 | +0.0010, -0.0040, -0.0023 | 1.9e-07 | 2.6e-06 ✓ | **equivalent** |
| DSLT-EncoderDecoder (no translators) | medium | PSNR | +0.075 | [-0.008, +0.153] | [-0.028, +0.167] | +0.12 | +0.103, +0.052, +0.069 | 0.0041 | 0.043 ✓ | **non-inferior (not equivalent)** |
| DSLT-EncoderDecoder (no translators) | medium | SSIM | +0.0006 | [-0.0007, +0.0017] | [-0.0010, +0.0019] | +0.08 | +0.0009, -0.0007, +0.0014 | 0.063 | 0.31 | **equivalent** |
| DSLT-EncoderDecoder (no translators) | medium | LPIPS | -0.0024 | [-0.0052, +0.0003] | [-0.0055, +0.0015] | -0.38 | +0.0018, -0.0049, -0.0041 | 5.4e-10 | 9.3e-09 ✓ | **non-inferior (not equivalent)** |
| DSLT-EncoderDecoder (no translators) | severe | PSNR | +0.137 | [+0.045, +0.241] | [+0.033, +0.266] | +0.38 | +0.055, +0.088, +0.269 | 5e-09 | 8e-08 ✓ | **non-inferior (not equivalent)** |
| DSLT-EncoderDecoder (no translators) | severe | SSIM | +0.0008 | [-0.0006, +0.0023] | [-0.0008, +0.0026] | +0.13 | +0.0002, -0.0004, +0.0027 | 0.0048 | 0.043 ✓ | **non-inferior (not equivalent)** |
| DSLT-EncoderDecoder (no translators) | severe | LPIPS | -0.0008 | [-0.0041, +0.0029] | [-0.0043, +0.0044] | -0.13 | +0.0047, -0.0031, -0.0039 | 0.03 | 0.18 | **equivalent** |

## Cost (inference: MPS FP32, median of 30 after 10 warm-up, one model per process; training: from run manifests)

| Model | Params (M) | GMACs @128² | GMACs @256² | Latency ms @128² b1 | Latency ms @256² b1 | Throughput img/s @256² b4 | Peak MPS GB @256² b4 | Train min (mean) | Peak train MPS GB (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A full (all translators) | 44.41 | 19.1 | 76.3 | 22.1 | 58.5 | 24.5 | 2.29 | 11.3 | 3.04 |
| DSLT-e2 (e2 translator only) | 23.02 | 13.6 | 54.5 | 16.6 | 45.7 | 32.1 | 2.29 | 36.6 | 2.60 |
| DSLT-EncoderDecoder (no translators) | 22.52 | 11.6 | 46.4 | 15.2 | 37.8 | 38.8 | 2.29 | 21.0 | 2.65 |
