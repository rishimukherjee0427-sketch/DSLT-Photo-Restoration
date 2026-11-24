# Stage 0 Decision Report: Does Damage Conditioning Help DSLT?

**Scope:** Stage 0 only. No baselines were retrained, no new architectures built, no convergence pilot run, and the README was not touched.

**Sources:** all numbers come from `results/v7_stage0/stage0_tables.md`, `stats.json` and `profiling_table.md`, which `experiments/stage0.py` generates. Interpretation is marked as such.

## 0. Verification

| Check | Result |
|---|---|
| Starting state | Commit `e612ad1`, clean tree. Historical checkpoints and results untouched. All new outputs are in `results/v7_stage0/` |
| Checkpoint integrity (12 checkpoints: A/B/C/O × seeds 42, 123, 2026) | All pass (`checkpoint_integrity.json`): SHA256 equals the training-manifest hash, 0 non-finite weights, metadata as expected (config, seed, `vgg_normalize=True`, 8×100 steps, crop 128) |
| GT severity maps vs. cached test inputs | Max \|Δ\| = 0.5/255 at every severity (PNG rounding) |
| Prediction alignment | All 12 evaluations checked 750/750 inputs against the expected cached test image, plus severity order for the oracle (recorded in each `*_eval_manifest.json`) |
| Training cost | 6 runs in 26.1 min (4.1–5.2 min each). Evaluation: 12 jobs |

## 1. Computational profile

Apple M4 Pro via MPS, FP32 inference. Latency is the median of 30 runs after 10 warm-up runs.

| Model | Params (M) | GMACs @128² | GMACs @256² | Latency @128² (ms) | Latency @256² (ms) | Throughput @256², batch 4 (img/s) |
|---|---:|---:|---:|---:|---:|---:|
| DSLT-A | 44.41 | 19.1 | 76.3 | 21.8 | 57.5 | 24.9 |
| DSLT-B / C | 46.06 | 20.6 | 82.5 | 24.3 | 68.1 | 20.9 |
| DSLT-O | 46.06 | 20.9 | 83.5 | 25.3 | 73.1 | 19.9 |
| Restormer-tiny | 3.75 | 5.1 | 20.6 | 18.4 | 51.1 | 13.5 |
| PromptIR-tiny | 7.22 | 6.4 | 25.7 | 37.3 | 71.1 | 12.1 |

- **Gating overhead:** gating costs +3.7% parameters, +8% MACs and +18% latency at 256² compared with A.
- **DSLT vs. baselines:** DSLT has 3.7× (vs. PromptIR-tiny) to 4.1× (vs. Restormer-tiny) more MACs, but higher batch-4 throughput on MPS. That is a hardware and kernel effect, not a sign of efficiency.
- **Measurement caveats:** MACs count convolutions and matmuls only. The peak-memory column (`profiling_table.md`) is process-wide driver allocation and is not reliable per model.

## 2. Oracle conditioning (O vs. A vs. C; 3 seeds; Holm over 27 tests)

**DSLT-O** is a diagnostic, not a deployable model. It is DSLT-C with the ground-truth severity map given to the gate head and spatial gates, in training, validation and test. It measures the most that damage information can add to this backbone under this protocol.

Test PSNR, mean ± SD over 3 seeds:

| Model | Mild | Medium | Severe |
|---|---:|---:|---:|
| A (ungated) | 26.72 ± 0.10 | 24.07 ± 0.07 | 18.72 ± 0.28 |
| C (learned conditioning, deployable) | 26.72 ± 0.14 | 24.07 ± 0.06 | 18.80 ± 0.24 |
| O (oracle conditioning, diagnostic) | 26.73 ± 0.15 | 24.09 ± 0.09 | 18.81 ± 0.31 |

PSNR Δ (dB), shown as mean [95% seed×image bootstrap CI] {per-seed Δ for seeds 42, 123, 2026}, with Holm p:

| Severity | O − A | O − C | C − A |
|---|---|---|---|
| Mild | +0.004 [−0.070, +0.077] {+0.046, −0.057, +0.025}, p=1 | +0.010 [−0.046, +0.062], p=1 | −0.005 [−0.073, +0.054], p=1 |
| Medium | +0.022 [−0.061, +0.104] {+0.028, −0.048, +0.086}, p=1 | +0.028 [−0.035, +0.091], p=0.72 | −0.006 [−0.106, +0.077], p=1 |
| Severe | **+0.093 [+0.036, +0.147] {+0.038, +0.115, +0.126}, p=1.4e-10** | +0.017 [−0.054, +0.101], p=1 | **+0.076 [+0.006, +0.139] {+0.087, +0.011, +0.132}, p=6.9e-06** |

Other metrics:
- **O − A SSIM:** severe +0.0020 [+0.0004, +0.0034] (Holm-significant). Mild and medium are slightly worse: −0.0005 and −0.0008, both Holm-significant.
- **O − A LPIPS:** worse at mild and medium (+0.0008, Holm-significant), not significant at severe.
- **O − C:** no test reaches significance after Holm correction (smallest p_Holm = 0.44).

## 3. Counterfactual gate intervention (B and C checkpoints with G forced to 1; 3 seeds; Holm over 18 tests)

This is an intervention on already-trained weights. It shows whether the trained network depends on its gates. It is **not** an independently trained ungated model.

| | Mild | Medium | Severe |
|---|---:|---:|---:|
| Δ PSNR, B − B(G=1) | +0.011 [+0.003, +0.023], p=1 | +0.007 [−0.001, +0.022], p=1 | −0.007 [−0.015, −0.001], p=0.10 |
| Δ PSNR, C − C(G=1) | +0.041 [−0.025, +0.136], p=1 | +0.023 [−0.038, +0.083], p=1 | −0.004 [−0.030, +0.021], p=0.30 |
| Mean \|Δoutput\| on the [0,1] scale, B, range over seeds | 0.0006–0.0015 | 0.0007–0.0022 | 0.0012–0.0043 |
| Mean \|Δoutput\|, C, range over seeds | 0.0031–0.0099 | 0.0034–0.0092 | 0.0020–0.0036 |

The only Holm-significant effects are tiny ones in B: LPIPS at mild and medium, about +0.0002 worse with gates, and SSIM at severe, −0.0004 with gates.

**Effective gates** (test-image mean for scales e2 / e3 / e4 / bottleneck; measured):
- **B (unsupervised): gates collapsed, and inconsistently across seeds.**
  - e2 ≈ 1.00 and **e3 and e4 ≈ 0.002–0.019** (almost shut off) for all seeds and severities.
  - The bottleneck gate is 0.83 for seed 42, but 0.17 and 0.16–0.23 for seeds 123 and 2026.
  - Gates barely change between mild and severe.
- **C (supervised):** gates follow severity, rising from 0.28–0.44 on mild to 0.79–0.85 on severe, with SD 0.09–0.23 across images.

## 4. Third seed: DSLT-A/B/C over 3 seeds (42, 123, 2026; Holm over 27 tests)

| Comparison | Mild Δ PSNR | Medium Δ PSNR | Severe Δ PSNR | LPIPS (all severities) |
|---|---|---|---|---|
| B − A | −0.053, p=1 | **−0.110 [−0.249, +0.001], p=4.9e-05** | −0.035, p=0.17 | worse: +0.0013 / +0.0024 / +0.0031. Holm-significant at medium and severe; seed-robust at all three |
| C − A | −0.005, p=1 | −0.006, p=1 | **+0.076 [+0.006, +0.139], p=5.5e-06** | worse: +0.0011 / +0.0013 / +0.0014. Holm-significant at all three; seed-robust at mild and medium |
| C − B | +0.048, p=0.86 | **+0.104 [+0.045, +0.175], p=1.9e-06** | **+0.111 [+0.018, +0.202], p=5.7e-11** | better at medium (Holm-significant, seed-robust) and severe (Holm-significant) |

- **Seed-level spread:** across the 3 seeds, the per-config SD of the test mean is up to 0.28 dB at severe (A) and 0.24 dB at mild (B). That is about 3× the largest gating effect.

## 5. Scientific interpretation

Interpretation, grounded in the measurements above.

1. **Damage information is close to worthless for this backbone at this budget.**
   - Even *perfect* severity information (O) adds **+0.004 / +0.022 / +0.093 dB** over the ungated model.
   - Only the severe gain is consistent: its CI excludes 0, all seeds are positive, and it is Holm-significant. It is still below the pre-set 0.1 dB practical threshold, and one seed gives only +0.038 dB.
   - The pre-registered go criterion for Direction B (oracle ≥ 0.1 dB over A, robustly across 3 seeds) is **not met**.
2. **The learned estimator is not the bottleneck.** Learned conditioning (C) is statistically indistinguishable from the oracle (O − C: no Holm-significant test). C's small severe-damage gain (+0.076 dB) reaches about 80% of the oracle's (+0.093 dB). Better severity estimation therefore cannot rescue the approach.
3. **The trained networks don't depend on their gates.**
   - Forcing G = 1 changes PSNR by at most 0.04 dB, and no PSNR change is Holm-significant.
   - In B, the e3 and e4 gates are about 0, so those translators are effectively off, but switching them fully on barely changes the output (mean \|Δ\| ≤ 0.0043).
   - Implication (hypothesis): the translators' residual corrections `T(e) − e` contribute little at e3 and e4. A gate that scales a near-negligible term can't matter. This also questions how much the "latent translation" modules contribute to DSLT-A, which needs a direct test (§7).
4. **Gating has a consistent perceptual cost.** Every gated variant, including the oracle, has slightly *worse* LPIPS than A, by +0.0006 to +0.0031. B is also worse than A in SSIM at mild and medium.
5. **Severity supervision mainly repairs unsupervised gating.** C − B is about +0.10 to +0.11 dB at medium and severe, robust across seeds. But C only gets back to A's level, except at severe, where it gains less than 0.1 dB.
6. **Two kinds of evidence:**
   - Many differences are Holm-significant at the image level, meaning they are consistent across the 250 test images for these trained models.
   - At the seed level the evidence is weaker: with 3 seeds, the SD of seed means (up to 0.28 dB) is larger than every effect, and 3 seeds can't establish generalisation over training runs.
   - Practically meaningful, seed-robust improvements of at least 0.1 dB: **none** over A.

## 6. Recommendation

- **Stop Direction B (improving adaptive gating).** Even the oracle upper bound falls below the practical threshold. On this evidence, further gating designs can at best approach about +0.09 dB on severe damage at this budget.
- **Stop Direction C (a different conditioning mechanism) as the primary line.** The oracle bounds the value of the *information*, whatever the mechanism. Severity information itself is almost redundant here.
  - Caveat: the oracle conditions through the existing FiLM-gate pathway. Conditioning *inside* the translator blocks (SFT-style, unbounded) was not tested.
  - The ≥ 0.1 dB ceiling argument is strong but not airtight. A cheap, separately approved test would be needed before reopening this line.
- **Continue Direction A, as a controlled study.**
  - Main architecture: ungated DSLT-A.
  - Reportable negative findings (on a well-trained backbone, explicit damage conditioning does not help):
    1. Unsupervised gates collapse.
    2. Supervised gates track damage but the network doesn't rely on them.
    3. Oracle conditioning gives less than 0.1 dB.
    4. Gating costs +8% MACs, +18% latency and slightly worse LPIPS.
  - Main caveat: everything above is at 800 steps and 128 px. The conclusion could change with convergence, so the convergence pilot is the first thing that must be confirmed.

## 7. Remaining experiments needed for publication (none started)

| # | Experiment | Why | Compute estimate (measured 0.248 s/step at 128 px, 0.841 s/step at 256 px, DSLT-C, FP32) |
|---|---|---|---|
| 1 | **Translator ablation** on DSLT-A: T = identity at inference on the existing checkpoints (counterfactual), plus a trained "no-translator" U-Net | Tests whether "latent translation", DSLT's namesake, contributes anything (§5.3) | ~5 min inference. A trained variant is ~4 min per seed at this budget |
| 2 | **Convergence pilot:** A at 256 px, 20k steps, 1 seed, full 250-image validation every 1k steps | Establishes the schedule. Tests whether conclusions hold beyond 800 steps | ~4.7 h, plus about 20 min of validation |
| 3 | **Re-test the gating conclusion at convergence:** A vs. C vs. O at the chosen schedule, 3 seeds | Conclusions §5.1–5.3 at a realistic budget | ~9 × 4.7 h ≈ 42 h at 20k steps and 256 px (about 12.5 h at 128 px) |
| 4 | **Fair baselines:** Restormer, PromptIR and NAFNet (parameter- or MAC-matched), plus DSLT-Small, under the corrected protocol, 3 seeds | Valid comparative claims | ~12 runs; baselines at 256 px with grad checkpointing were ~4.6–6.3 s/step, so ~25–35 h each at 20k steps. Total ~100+ h at 256 px; ~25–40 h at 128 px |
| 5 | **Real-world evaluation** on 87 `data/raw` photos: qualitative grid (pre-set selection rule), NIQE/BRISQUE as supplementary, small blind preference study | Real-world generalisation | ~1 h, plus participants |
| 6 | Rename the method (clash with Wan et al.'s "Deep Latent Space Translation" [unverified]); verify the literature; confirm Scopus status of target journals | Novelty and positioning | Writing only |

**Suggested next stage for approval:** #1 (5 minutes, decisive for the paper's framing), then #2 (~5 h). Decide #3 and #4 from the pilot's learning curve.

## 8. Limitations of Stage 0

- **Budget:** 800 steps at 128 px is not converged. Best epochs were 7–8 of 8.
- **Seeds:** with 3 seeds, seed-level variance is poorly estimated.
- **Validation:** best epochs were chosen on only 24 validation images per severity.
- **Data:** degradations are synthetic and in-distribution, on a single 256×256 dataset.
- **Oracle scope:** the oracle tests only the existing FiLM-gate injection pathway.
- **Profiling:** MACs exclude element-wise operations, and peak memory is not separable per model.
- **Scope note:** the A/B/C seeds 123 and 2026 results are reused from `results/v7_corrected`. They were produced by the same pipeline, and their checkpoints were re-verified here.
