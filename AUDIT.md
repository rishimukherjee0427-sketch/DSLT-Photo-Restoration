# DSLT Project Audit

**Date:** 23 Nov 2026 · **Scope:** gating mechanism, evaluation methodology, baselines, reported results, reproducibility · **Repo state:** commit `c142984`

## 1. Summary

| Area | Status before audit | Status after audit |
|---|---|---|
| Damage-conditioned gating | ❌ Non-functional: gates collapsed to constants, 0.000 dB effect | ✅ Gates now track damage, ⚠️ but there is no significant PSNR gain |
| Train/test separation | ❌ All metrics measured on training images | ✅ Deduplicated train/val/test split |
| Severity coverage | ⚠️ Only `mild` / `heavy`, evaluated on mild only | ✅ mild / medium / severe, all models evaluated at all three |
| Baselines | ⚠️ Only U-Net and older DSLT variants, under unknown conditions | ✅ Restormer and PromptIR, same protocol |
| Metrics | ⚠️ PSNR/SSIM only, implementations differ across scripts | ✅ One shared PSNR/SSIM/LPIPS implementation |
| Headline claim "30.16 dB" | ❌ Not valid: measured on training data | ✅ Replaced by a held-out test table |
| Training budget | — | ⚠️ Short, equal budget (800 steps/model), which must be disclosed |

## 2. Critical findings

### C1. Reported metrics were measured on training data (severity: critical, status: fixed)
- `experiments/train.py`, `evaluate.py`, `train_gan.py` and the original ablation all trained and evaluated on `data/train_clean`.
- The "30.16 dB / 0.876 SSIM" result (`results/metrics_v5.json`) is therefore a training-set number.
- `results/best_model_v5.pth` saw every image in the new val/test splits. On test it scores 29.90 dB (mild), but this is **contaminated** and must not be reported as generalisation.
- **Fix:** `data/splits/{train,val,test}.txt` (4279 / 250 / 250, seed 42). All new models were trained on `train.txt` only.

### C2. Near-duplicate leakage in the dataset (severity: critical, status: fixed)
- `train_clean` (5000 images) contains 368 groups of exact md5 duplicates. It is the source images resized and randomly flipped.
- Flip-aware pHash + dHash clustering found 621 multi-image clusters covering 2025 images. A naive random split would have leaked.
- **Fix:** val and test take one image per cluster, and the 221 other cluster members are excluded from all splits.

### C3. Adaptive gating had no effect (severity: critical, status: diagnosed and fixed)
Evidence is in `results/gate_diagnosis/` (300 iters, 96-image probe):
- **No gradient:** the gate head received **0.0 gradient** from the severity loss and 0.0045 from the pixel loss.
- **Collapsed gates:** gate spread across images was 0.001–0.003, so the gates were effectively per-scale constants.
- **No per-image conditioning:** PSNR with predicted gates 22.403 vs. constant mean gate 22.403 vs. gates shuffled across images 22.404.
- **Uninformative input:** the gate input (avg-pooled e4) could not predict severity (ridge R² = −0.15).

**Fix:**
- **Richer gate head:** e3 and e4 avg+max pooling plus the predicted severity map, bias-initialised.
- **FiLM-style gating:** per-channel and spatial gates.
- **Optional gate losses:** severity supervision, sparsity and bimodality.

**Effect:**
- Gate spread across images is now 0.13–0.20.
- Correlation with ground-truth severity rose to 0.64–0.69, from 0.18–0.26.
- Config C reaches 0.66–0.78 gate–target correlation on val.

**Residual issue:** the gating gives **no significant PSNR improvement** on test (+0.07 / +0.05 / −0.01 dB for B vs. A). See §4.

## 3. Major findings

| ID | Finding | Status |
|---|---|---|
| M1 | No LPIPS anywhere in the v5 results; `results/ablation.csv` has blank LPIPS for v5 rows | Fixed: LPIPS (AlexNet) for all models |
| M2 | PSNR formula `10·log10(1/(mse+1e-8))` gives up to 80 dB on pristine images, inflating means. 8/250 mild val images receive no degradation at all | Fixed in the benchmark: per-image cap at 50 dB (the uncapped mild identity mean would read 27.3 dB vs. 25.7) |
| M3 | Old comparison table mixes models evaluated under unknown, different conditions (U-Net 15.78, v4 23.69, etc.) | Fixed: those rows are superseded by `results/benchmark_table.md`. The old rows should not be cited |
| M4 | Unknown severity strings silently fell back to `heavy` | Fixed: now raises `ValueError` |
| M5 | Original ablation: 5 epochs × 20 batches, training-batch metrics, one severity | Fixed: equal-budget A/B/C at 3 severities, best epoch selected on val, final numbers on test |
| M6 | `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` in scripts disables the MPS memory cap. Running jobs in parallel caused heavy swapping on a 24 GB Mac | Fixed in new scripts (0.5 / 0.4 via `setdefault`). The old `train.py` still sets 0.0 |

## 4. Limitations to disclose in the paper

1. **Short training budget:** 800 optimizer steps per model, batch 4, 128 px crops, about 4–8 min each on an Apple M4 Pro GPU. Rankings are indicative, and absolute numbers are far below converged values.
2. **Baselines are reduced-size re-implementations:** Restormer-tiny (3.75 M) and PromptIR-tiny (7.22 M) were written from scratch and are not the official checkpoints or configurations. DSLT is 44–46 M, so the comparison is not parameter-matched.
3. **Single seed:** there are no error bars. Gating differences of 0.01–0.07 dB are within plausible run-to-run noise. Make no claim that gating improves PSNR.
4. **Mixed metric picture:** DSLT leads on PSNR (+0.4–0.6 dB vs. baselines) but trails on LPIPS (0.225 vs. 0.221 mild; 0.587 vs. 0.575 severe). SSIM is roughly tied.
5. **Synthetic degradations only:** the degradations are synthetic, on 256×256 face images. There is no evaluation on real damaged photos with ground truth.
6. **Hand-set gate target:** the gate-supervision target mapping (`a=0.2, b=0.8`, `GATE_MIX`) is heuristic and untuned.
7. **Val/test gap:** on val, B beat A by 0.4 dB, but this did not carry over to test.

## 5. Final results (test split, 250 images)

PSNR (dB) ↑ / SSIM ↑ / LPIPS ↓

| Model | Params | Mild | Medium | Severe |
|---|---:|---|---|---|
| Degraded input | 0 | 25.60 / 0.7847 / 0.2482 | 21.99 / 0.7074 / 0.3866 | 15.25 / 0.5111 / 0.6844 |
| Bicubic | 0 | 24.81 / 0.7826 / 0.2802 | 21.69 / 0.7154 / 0.4041 | 15.32 / 0.5168 / 0.6795 |
| Restormer-tiny | 3.75 M | 26.17 / 0.8045 / 0.2214 | 23.79 / 0.7460 / 0.3259 | 18.01 / 0.5746 / 0.5751 |
| PromptIR-tiny | 7.22 M | 26.16 / 0.8076 / 0.2235 | 23.85 / 0.7493 / 0.3279 | 18.02 / 0.5755 / 0.5772 |
| DSLT A (ungated) | 44.41 M | 26.74 / 0.8021 / 0.2261 | 23.93 / 0.7430 / 0.3419 | 18.43 / 0.5898 / 0.5882 |
| DSLT B (adaptive gating) | 46.06 M | 26.81 / 0.8021 / 0.2251 | 23.98 / 0.7442 / 0.3417 | 18.42 / 0.5884 / 0.5874 |
| DSLT C (gating + supervision + sparsity) | 46.06 M | 26.78 / 0.8019 / 0.2258 | 23.99 / 0.7439 / 0.3417 | 18.45 / 0.5909 / 0.5865 |
| *v5 legacy (contaminated, excluded)* | 44.50 M | *29.90 / 0.8794 / 0.1295* | *26.84 / 0.8158 / 0.2315* | *18.52 / 0.5945 / 0.5203* |

Source: `results/benchmark_results.json`, `results/benchmark_table.{md,csv}`.

## 6. Remaining issues (not fixed)

| ID | Issue | Location |
|---|---|---|
| R1 | Project `.venv` is broken: the `python` symlink is missing and `pyvenv.cfg` points to the old path `~/Documents/Minor Project ` | `.venv/` |
| R2 | `run_server.sh` `cd`s into the old path, so the web app launcher fails | `run_server.sh:2` |
| R3 | `cv2` is used but not imported in the dummy-data fallback | `experiments/ablation_gated_vs_ungated.py` (superseded by `run_ablation.py`) |
| R4 | Legacy `train.py` / `evaluate.py` still use all of `train_clean` and watermark 0.0 | `experiments/train.py`, `experiments/evaluate.py` |
| R5 | README v6 table still describes the old scalar gate (`g·T(e)+(1-g)·e`) and old parameter counts (~31.2 M) | `README.md` (v6 section) |
| R6 | Model checkpoints (`*.pth`, 169–180 MB) exceed GitHub's limit and exist only locally | `results/`, `.gitignore` |
| R7 | Project directory name ends with a trailing space (`Minor Project `), which breaks absolute paths in tools | filesystem |

## 7. Reproducibility

- **Environment:** Python 3.12, torch 2.15, torchvision 0.29, opencv, scikit-image, lpips (`requirements.txt`). Apple MPS.
- **Fixed inputs:** splits and cached benchmark inputs are deterministic (seed 42; per-image seeds via crc32). `python experiments/make_benchmark.py --splits --cache` regenerates the 196 MB cache.
- **Commands:** full training and evaluation commands are in `README.md` → *Benchmark → Reproduce*. Training logs are in `results/phase2_train.log` and evaluation logs in `results/phase2_eval.log`.
