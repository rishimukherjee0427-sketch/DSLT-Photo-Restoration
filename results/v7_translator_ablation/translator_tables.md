## Translator contribution (inference-time interventions on DSLT-A; test split; mean ± SD over seeds 42, 123, 2026)

| Variant | Mild PSNR | Mild SSIM | Mild LPIPS | Medium PSNR | Medium SSIM | Medium LPIPS | Severe PSNR | Severe SSIM | Severe LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A (original) | 26.72 ± 0.10 | 0.8026 ± 0.0002 | 0.2225 ± 0.0016 | 24.07 ± 0.07 | 0.7459 ± 0.0011 | 0.3367 ± 0.0022 | 18.72 ± 0.28 | 0.5953 ± 0.0046 | 0.5781 ± 0.0063 |
| All translators off (T=I) | 26.63 ± 0.11 | 0.8031 ± 0.0004 | 0.2218 ± 0.0015 | 23.99 ± 0.14 | 0.7462 ± 0.0007 | 0.3356 ± 0.0028 | 18.62 ± 0.24 | 0.5927 ± 0.0034 | 0.5795 ± 0.0064 |
| e2 translator off | 26.65 ± 0.06 | 0.8031 ± 0.0002 | 0.2216 ± 0.0015 | 24.01 ± 0.12 | 0.7462 ± 0.0008 | 0.3353 ± 0.0029 | 18.65 ± 0.25 | 0.5935 ± 0.0030 | 0.5791 ± 0.0067 |
| e3 translator off | 26.73 ± 0.09 | 0.8027 ± 0.0002 | 0.2225 ± 0.0015 | 24.07 ± 0.08 | 0.7460 ± 0.0011 | 0.3367 ± 0.0021 | 18.71 ± 0.27 | 0.5950 ± 0.0045 | 0.5783 ± 0.0063 |
| e4 translator off | 26.72 ± 0.10 | 0.8026 ± 0.0002 | 0.2225 ± 0.0016 | 24.07 ± 0.07 | 0.7459 ± 0.0011 | 0.3367 ± 0.0022 | 18.72 ± 0.27 | 0.5951 ± 0.0046 | 0.5781 ± 0.0063 |
| bottleneck translator off | 26.73 ± 0.11 | 0.8026 ± 0.0002 | 0.2226 ± 0.0016 | 24.07 ± 0.05 | 0.7459 ± 0.0011 | 0.3370 ± 0.0020 | 18.71 ± 0.28 | 0.5950 ± 0.0049 | 0.5783 ± 0.0060 |
| Residual scale s=0 (all) | 26.63 ± 0.11 | 0.8031 ± 0.0004 | 0.2218 ± 0.0015 | 23.99 ± 0.14 | 0.7462 ± 0.0007 | 0.3356 ± 0.0028 | 18.62 ± 0.24 | 0.5927 ± 0.0034 | 0.5795 ± 0.0064 |
| Learned content off, rescaling kept | 26.68 ± 0.10 | 0.8024 ± 0.0007 | 0.2225 ± 0.0008 | 24.05 ± 0.09 | 0.7455 ± 0.0007 | 0.3364 ± 0.0024 | 18.65 ± 0.23 | 0.5926 ± 0.0030 | 0.5791 ± 0.0062 |

### Paired differences vs. original DSLT-A (variant − original; Holm over 54 tests)

| Variant | Severity | Metric | Δ mean | 95% CI (seed×image bootstrap) | d_z | per-seed Δ | p | p Holm | Sig. | Seeds agree | CI excl. 0 | Direction |
|---|---|---|---:|---|---:|---|---:|---:|:---:|:---:|:---:|---|
| All translators off (T=I) | mild | PSNR | -0.098 | [-0.260, +0.043] | -0.16 | -0.244, +0.025, -0.074 | 0.13 | 1 | – | – | – | worse |
| All translators off (T=I) | mild | SSIM | +0.0005 | [-0.0001, +0.0012] | +0.15 | +0.0009, +0.0006, +0.0000 | 0.0012 | 0.042 | ✓ | ✓ | – | better |
| All translators off (T=I) | mild | LPIPS | -0.0007 | [-0.0022, +0.0005] | -0.17 | +0.0002, -0.0023, +0.0001 | 0.064 | 1 | – | – | – | better |
| All translators off (T=I) | medium | PSNR | -0.078 | [-0.185, +0.007] | -0.16 | -0.151, -0.013, -0.070 | 0.15 | 1 | – | ✓ | – | worse |
| All translators off (T=I) | medium | SSIM | +0.0002 | [-0.0006, +0.0011] | +0.05 | +0.0008, +0.0000, -0.0002 | 0.018 | 0.5 | – | – | – | better |
| All translators off (T=I) | medium | LPIPS | -0.0011 | [-0.0025, -0.0000] | -0.26 | -0.0005, -0.0026, -0.0003 | 0.00014 | 0.0057 | ✓ | ✓ | ✓ | better |
| All translators off (T=I) | severe | PSNR | -0.096 | [-0.142, -0.048] | -0.39 | -0.068, -0.137, -0.081 | 1.3e-09 | 6.2e-08 | ✓ | ✓ | ✓ | worse |
| All translators off (T=I) | severe | SSIM | -0.0026 | [-0.0046, -0.0010] | -0.40 | -0.0018, -0.0046, -0.0014 | 4.1e-09 | 2e-07 | ✓ | ✓ | ✓ | worse |
| All translators off (T=I) | severe | LPIPS | +0.0014 | [+0.0006, +0.0023] | +0.23 | +0.0013, +0.0012, +0.0019 | 0.00035 | 0.013 | ✓ | ✓ | ✓ | worse |
| e2 translator off | mild | PSNR | -0.075 | [-0.186, +0.035] | -0.13 | -0.155, +0.010, -0.079 | 0.39 | 1 | – | – | – | worse |
| e2 translator off | mild | SSIM | +0.0005 | [-0.0000, +0.0010] | +0.14 | +0.0007, +0.0005, +0.0002 | 0.0024 | 0.081 | – | ✓ | – | better |
| e2 translator off | mild | LPIPS | -0.0009 | [-0.0023, +0.0002] | -0.23 | -0.0001, -0.0024, -0.0002 | 0.0026 | 0.084 | – | ✓ | – | better |
| e2 translator off | medium | PSNR | -0.064 | [-0.150, +0.012] | -0.13 | -0.110, -0.014, -0.069 | 0.62 | 1 | – | ✓ | – | worse |
| e2 translator off | medium | SSIM | +0.0002 | [-0.0005, +0.0009] | +0.05 | +0.0006, -0.0000, +0.0001 | 0.019 | 0.51 | – | – | – | better |
| e2 translator off | medium | LPIPS | -0.0015 | [-0.0029, -0.0003] | -0.35 | -0.0007, -0.0029, -0.0008 | 1e-07 | 4.5e-06 | ✓ | ✓ | ✓ | better |
| e2 translator off | severe | PSNR | -0.071 | [-0.115, -0.027] | -0.31 | -0.055, -0.111, -0.047 | 3.5e-06 | 0.00016 | ✓ | ✓ | ✓ | worse |
| e2 translator off | severe | SSIM | -0.0018 | [-0.0040, -0.0000] | -0.29 | -0.0006, -0.0041, -0.0006 | 2.3e-05 | 0.00097 | ✓ | ✓ | ✓ | worse |
| e2 translator off | severe | LPIPS | +0.0011 | [+0.0002, +0.0020] | +0.17 | +0.0014, +0.0006, +0.0012 | 0.0095 | 0.27 | – | ✓ | ✓ | worse |
| e3 translator off | mild | PSNR | +0.008 | [-0.003, +0.018] | +0.19 | +0.005, +0.016, +0.005 | 0.00015 | 0.0059 | ✓ | ✓ | – | better |
| e3 translator off | mild | SSIM | +0.0000 | [-0.0000, +0.0002] | +0.09 | +0.0001, -0.0000, +0.0000 | 0.00044 | 0.016 | ✓ | – | – | better |
| e3 translator off | mild | LPIPS | -0.0000 | [-0.0002, +0.0001] | -0.09 | -0.0001, +0.0000, +0.0000 | 0.22 | 1 | – | – | – | better |
| e3 translator off | medium | PSNR | +0.001 | [-0.011, +0.009] | +0.03 | -0.009, +0.005, +0.006 | 0.32 | 1 | – | – | – | better |
| e3 translator off | medium | SSIM | +0.0000 | [-0.0001, +0.0001] | +0.02 | +0.0001, -0.0000, +0.0000 | 0.12 | 1 | – | – | – | better |
| e3 translator off | medium | LPIPS | +0.0000 | [-0.0001, +0.0001] | +0.05 | -0.0001, +0.0001, +0.0001 | 0.67 | 1 | – | – | – | worse |
| e3 translator off | severe | PSNR | -0.010 | [-0.018, -0.001] | -0.42 | -0.013, -0.017, +0.000 | 6e-10 | 3e-08 | ✓ | – | ✓ | worse |
| e3 translator off | severe | SSIM | -0.0003 | [-0.0004, -0.0002] | -0.45 | -0.0003, -0.0004, -0.0002 | 1.1e-10 | 5.5e-09 | ✓ | ✓ | ✓ | worse |
| e3 translator off | severe | LPIPS | +0.0002 | [+0.0001, +0.0003] | +0.44 | +0.0002, +0.0003, +0.0001 | 4.4e-11 | 2.3e-09 | ✓ | ✓ | ✓ | worse |
| e4 translator off | mild | PSNR | -0.000 | [-0.004, +0.003] | -0.02 | -0.003, +0.003, -0.000 | 0.7 | 1 | – | – | – | worse |
| e4 translator off | mild | SSIM | +0.0000 | [-0.0000, +0.0000] | +0.01 | +0.0000, -0.0000, +0.0000 | 0.89 | 1 | – | – | – | better |
| e4 translator off | mild | LPIPS | +0.0000 | [-0.0000, +0.0000] | +0.07 | +0.0000, +0.0000, -0.0000 | 0.39 | 1 | – | – | – | worse |
| e4 translator off | medium | PSNR | +0.002 | [-0.000, +0.004] | +0.15 | +0.002, +0.000, +0.003 | 0.0066 | 0.21 | – | ✓ | – | better |
| e4 translator off | medium | SSIM | -0.0000 | [-0.0000, +0.0000] | -0.15 | -0.0000, -0.0000, -0.0000 | 0.032 | 0.8 | – | ✓ | – | worse |
| e4 translator off | medium | LPIPS | +0.0000 | [-0.0000, +0.0001] | +0.11 | +0.0000, +0.0001, -0.0000 | 0.12 | 1 | – | – | – | worse |
| e4 translator off | severe | PSNR | -0.003 | [-0.006, +0.000] | -0.18 | -0.001, -0.005, -0.004 | 0.056 | 1 | – | ✓ | – | worse |
| e4 translator off | severe | SSIM | -0.0002 | [-0.0002, -0.0001] | -0.49 | -0.0001, -0.0002, -0.0002 | 4.1e-12 | 2.2e-10 | ✓ | ✓ | ✓ | worse |
| e4 translator off | severe | LPIPS | +0.0000 | [-0.0000, +0.0001] | +0.17 | +0.0000, +0.0001, -0.0000 | 0.019 | 0.51 | – | – | – | worse |
| bottleneck translator off | mild | PSNR | +0.003 | [-0.010, +0.017] | +0.06 | +0.018, -0.005, -0.003 | 0.84 | 1 | – | – | – | better |
| bottleneck translator off | mild | SSIM | -0.0000 | [-0.0001, +0.0001] | -0.01 | +0.0001, +0.0000, -0.0001 | 0.51 | 1 | – | – | – | worse |
| bottleneck translator off | mild | LPIPS | +0.0001 | [+0.0001, +0.0002] | +0.33 | +0.0001, +0.0001, +0.0002 | 0.00015 | 0.0059 | ✓ | ✓ | ✓ | worse |
| bottleneck translator off | medium | PSNR | +0.002 | [-0.014, +0.020] | +0.03 | +0.021, -0.005, -0.012 | 0.86 | 1 | – | – | – | better |
| bottleneck translator off | medium | SSIM | -0.0000 | [-0.0002, +0.0001] | -0.04 | +0.0001, +0.0000, -0.0002 | 0.81 | 1 | – | – | – | worse |
| bottleneck translator off | medium | LPIPS | +0.0002 | [+0.0001, +0.0004] | +0.41 | +0.0001, +0.0002, +0.0004 | 2e-08 | 9.5e-07 | ✓ | ✓ | ✓ | worse |
| bottleneck translator off | severe | PSNR | -0.010 | [-0.025, +0.004] | -0.19 | -0.000, -0.005, -0.025 | 0.0074 | 0.22 | – | ✓ | – | worse |
| bottleneck translator off | severe | SSIM | -0.0003 | [-0.0007, +0.0000] | -0.31 | -0.0007, +0.0000, -0.0003 | 2.4e-05 | 0.001 | ✓ | – | – | worse |
| bottleneck translator off | severe | LPIPS | +0.0003 | [-0.0002, +0.0007] | +0.28 | -0.0003, +0.0004, +0.0007 | 0.00011 | 0.0046 | ✓ | – | – | worse |
| Learned content off, rescaling kept | mild | PSNR | -0.042 | [-0.098, +0.006] | -0.15 | -0.058, -0.041, -0.027 | 0.035 | 0.84 | – | ✓ | – | worse |
| Learned content off, rescaling kept | mild | SSIM | -0.0002 | [-0.0010, +0.0006] | -0.09 | +0.0006, -0.0003, -0.0010 | 0.39 | 1 | – | – | – | worse |
| Learned content off, rescaling kept | mild | LPIPS | -0.0000 | [-0.0012, +0.0012] | -0.00 | -0.0000, -0.0012, +0.0013 | 0.79 | 1 | – | – | – | better |
| Learned content off, rescaling kept | medium | PSNR | -0.026 | [-0.079, +0.021] | -0.10 | -0.044, -0.035, +0.001 | 0.21 | 1 | – | – | – | worse |
| Learned content off, rescaling kept | medium | SSIM | -0.0004 | [-0.0012, +0.0005] | -0.12 | +0.0005, -0.0007, -0.0010 | 0.27 | 1 | – | – | – | worse |
| Learned content off, rescaling kept | medium | LPIPS | -0.0004 | [-0.0016, +0.0008] | -0.11 | -0.0003, -0.0017, +0.0009 | 0.07 | 1 | – | – | – | better |
| Learned content off, rescaling kept | severe | PSNR | -0.065 | [-0.121, -0.015] | -0.35 | -0.023, -0.122, -0.049 | 4.2e-08 | 1.9e-06 | ✓ | ✓ | ✓ | worse |
| Learned content off, rescaling kept | severe | SSIM | -0.0027 | [-0.0049, -0.0011] | -0.55 | -0.0014, -0.0050, -0.0017 | 2.3e-15 | 1.2e-13 | ✓ | ✓ | ✓ | worse |
| Learned content off, rescaling kept | severe | LPIPS | +0.0010 | [+0.0003, +0.0017] | +0.21 | +0.0007, +0.0010, +0.0013 | 0.00073 | 0.026 | ✓ | ✓ | ✓ | worse |

### Diagnostics

| Seed | learned s (e2, e3, e4, bn) | mean ‖T(z)−z‖/‖z‖ (e2, e3, e4, bn) | max \|full − canonical\| | max \|s=0 − off_all\| |
|---:|---|---|---:|---:|
| 42 | [0.211, 0.197, 0.187, 0.192] | [0.387, 0.355, 0.372, 0.847] | 0 | 0 |
| 123 | [0.206, 0.186, 0.181, 0.19] | [0.387, 0.439, 0.325, 0.976] | 0 | 0 |
| 2026 | [0.21, 0.191, 0.19, 0.197] | [0.284, 0.308, 0.443, 0.907] | 0 | 0 |
