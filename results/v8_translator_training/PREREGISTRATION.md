# Stage 2 Pre-registration: Translator Ablation (written before any Stage 2 training or results)

**Configurations** (all ungated, no gating modules built, `experiments/run_ablation.py`):

| ID | Architecture |
|---|---|
| `Tfull` | All 4 translators (e2, e3, e4, bn) |
| `Te2` | e2 translator only |
| `Tnone` | No translators (encoder–decoder) |

**Training settings** (identical for all three):
- Seeds 42, 123, 2026.
- 800 steps (8 × 100), batch 4, crop 128, AdamW 2e-4, cosine schedule, warmup 50, grad clip 1.0, EMA 0.99.
- Mixed severity, FP32 on MPS, `vgg_normalize=True`, train split only.
- Best epoch selected on val (24 images × 3 severities, mean PSNR).
- Each configuration is tested once on the 250-image test split.

## Primary questions

1. Is `Te2` practically equivalent to `Tfull`?
2. Is `Tnone` practically equivalent to `Tfull`?

Each question is asked separately per severity (mild, medium, severe) and per metric.

## Equivalence margins (fixed now)

| Metric | Margin δ | Rationale |
|---|---|---|
| PSNR | ±0.10 dB | The practical-relevance threshold used throughout this project (Stage 0 go criterion) |
| SSIM | ±0.0020 | Roughly the largest seed-level SD of A observed so far (0.0046 at severe, 0.0011 at medium), halved toward the smaller severities. Below typically reportable SSIM differences |
| LPIPS | ±0.0050 | About 2× the largest seed SD observed for A (0.0063 at severe, 0.0022 at medium), scaled down. Below typically reportable LPIPS differences |

## Decision rules

- **Equivalence (two one-sided tests at α = 0.05):** the 90% two-level (seed × image) bootstrap CI of the mean paired difference (variant − `Tfull`) lies entirely within (−δ, +δ). Only then is a variant called *practically equivalent* for that severity and metric.
- **Non-inferiority:** the lower bound of that 90% CI (for LPIPS, the upper bound, since lower is better) does not cross −δ (for LPIPS, +δ).
- **Difference tests:** two-sided Wilcoxon signed-rank on per-image seed-averaged differences (n = 250), Holm-corrected over all difference tests (2 comparisons × 3 severities × 3 metrics = 18). Reported with 95% CIs, Cohen's d_z and per-seed Δ.
- **Wording rules:**
  - A non-significant difference is **not** evidence of equivalence.
  - If a CI crosses a margin, the result is reported as *inconclusive*, not *equivalent*.
  - With 3 seeds, seed-level generalisation is **not** claimed. Equivalence statements are conditional on this budget and protocol.

## Secondary measures (descriptive only, no inference)

Parameters, MACs at 128² and 256², latency (MPS, batch 1 at 128² and 256²), batch-4 throughput, peak MPS memory, and training time.

## Visual comparison

Fixed images chosen by the Stage 1 rule: the same test indices as `results/v7_translator_ablation/visual_comparison_seed42.png`, seed 42.
