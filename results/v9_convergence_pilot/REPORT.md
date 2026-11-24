# Stage 3 Report: Convergence Pilot, `Tfull` vs `Tnone`

> **This is a one-seed exploratory pilot.** All tests below are image-level and conditional on a single training run (seed 42). They say nothing about variation between training seeds, which Stage 2 showed can reach 0.36 dB. Nothing here supports a confirmatory claim.

**Models compared**
- `Tfull`: DSLT-A with all 4 translators, 44.41M parameters.
- `Tnone`: the same encoder–decoder with no translators, 22.52M parameters.
- Neither model is gated, and no gating modules are built.

**Training protocol** (identical for both models)
- 20 epochs × 1000 steps = 20,000 steps, batch 4, 256×256 crops.
- AdamW 2e-4 with cosine schedule, warmup 500, grad clip 1.0, EMA 0.999.
- Mixed severity, `vgg_normalize=True`, FP32 on MPS, seed 42, train split only.
- After every epoch, validation runs on all 250 val images at each severity. The checkpoint with the best mean val PSNR is kept, and the test set is used once.

**Sources**
- Numbers come from `stage3_tables.md`, `stats.json`, `checkpoint_integrity.json`, `profiling.json` and the `seed42/*_history.json` files, all produced by `experiments/stage3.py` and `experiments/run_ablation.py`.
- Margins are the pre-registered Stage 2 margins (`results/v8_translator_training/PREREGISTRATION.md`): PSNR 0.10 dB, SSIM 0.0020, LPIPS 0.0050.

## 0. Verification

| Check | Result |
|---|---|
| Start state | Committed tree at `8656aff` (pre-flight profile, resume test and runner). Earlier results and checkpoints were left untouched, and all outputs went to `results/v9_convergence_pilot/` |
| Pre-flight (`preflight_profile.json`) | `Tfull` 0.816 s/step, 7.19 GB peak. `Tnone` 0.596 s/step, 5.77 GB peak. Estimated 8.24 h in total, and both fit in memory |
| Recovery test (`recovery_test.json`) | A run stopped and resumed matched an uninterrupted run exactly: max \|Δ\| = 0 for raw weights, EMA weights and optimizer state |
| Sleep prevention | `caffeinate -ims` was active for the whole run, confirmed with `pmset -g assertions` |
| Budget completed | **Both models ran all 20,000 steps** (`last.pth` reports step 20000, epoch 20). There was no interruption, resume or shortening. Run times: `Tfull` 12:41 to 17:38, `Tnone` 17:38 to 21:14 (`train.log`) |
| Checkpoints | 2/2 pass. SHA256 matches the manifest, there are no non-finite weights, and the metadata is correct (config, seed, translators, `gate_mode='none'`, `vgg_normalize`, 20×1000 steps, crop 256) |
| Prediction alignment | 750/750 inputs per model were checked against the expected cached test degradation |
| Checkpoint selection | Validation only. For both models the best epoch was **20**, the last one |

## 1. Test results (250 test images; best-val checkpoint)

PSNR (dB) ↑ / SSIM ↑ / LPIPS ↓

| Model | Params | Mild | Medium | Severe |
|---|---:|---|---|---|
| `Tfull` DSLT-A | 44.41M | 29.74 / 0.8771 / 0.1407 | **27.51** / 0.8254 / 0.2271 | **22.82 / 0.6897 / 0.4264** |
| `Tnone` encoder–decoder | 22.52M | **29.79 / 0.8789 / 0.1374** | 27.43 / **0.8255 / 0.2265** | 22.78 / 0.6861 / 0.4352 |

Compared with Stage 2 (800 steps at 128 px), both models gained about 3–4 dB. Stage 2 reached 26.8 / 24.2 / 18.9 dB on mild / medium / severe.

## 2. Image-level paired tests, `Tnone` − `Tfull`

- n = 250 test images, one training seed.
- 90% bootstrap CI for equivalence and non-inferiority; Wilcoxon signed-rank with Holm correction over 9 tests.

| Severity | Metric | Δ mean | 90% CI | 95% CI | d_z | Images where `Tnone` is better | p (Holm) | Verdict |
|---|---|---:|---|---|---:|---:|---:|---|
| mild | PSNR | +0.050 | [−0.047, +0.150] | [−0.065, +0.170] | +0.05 | 55% | 1 | non-inferior (not equivalent) |
| mild | SSIM | +0.0018 | [+0.0006, +0.0031] | [+0.0004, +0.0033] | +0.16 | 62% | 0.002 ✓ | non-inferior, `Tnone` better |
| mild | LPIPS | −0.0033 | [−0.0044, −0.0022] | [−0.0046, −0.0020] | −0.31 | 69% | 1e-07 ✓ | **equivalent**, `Tnone` slightly better |
| medium | PSNR | −0.080 | [−0.175, +0.016] | [−0.193, +0.034] | −0.09 | 47% | 0.79 | inconclusive |
| medium | SSIM | +0.0001 | [−0.0011, +0.0013] | [−0.0013, +0.0015] | +0.01 | 47% | 1 | **equivalent** |
| medium | LPIPS | −0.0007 | [−0.0019, +0.0006] | [−0.0022, +0.0008] | −0.05 | 55% | 1 | **equivalent** |
| severe | PSNR | −0.039 | [−0.131, +0.048] | [−0.148, +0.065] | −0.05 | 47% | 1 | inconclusive |
| severe | SSIM | −0.0036 | [−0.0052, −0.0021] | [−0.0054, −0.0019] | −0.25 | 35% | 2e-06 ✓ | **worse beyond margin** (narrowly: the CI bound is −0.0021 against a margin of −0.0020) |
| severe | LPIPS | +0.0088 | [+0.0070, +0.0105] | [+0.0067, +0.0108] | +0.52 | 31% | 4e-13 ✓ | **worse beyond margin** |

## 3. Learning curves and convergence

See `learning_curves.png`.

| Model | Best epoch | Best/final val PSNR (mean of 3 severities) | Val gain, epochs 18→20 | Final train loss |
|---|---:|---:|---:|---:|
| `Tfull` | 20 / 20 | 26.533 | +0.030 dB | 0.1906 |
| `Tnone` | 20 / 20 | 26.572 | +0.033 dB | 0.1912 |

Final-epoch validation PSNR, mild / medium / severe:
- `Tfull`: 29.71 / 27.28 / 22.60 dB.
- `Tnone`: 29.89 / 27.30 / 22.53 dB.

**What the curves show**
- **Training loss** curves almost overlap the whole way.
- **Mild:** `Tnone` pulls ahead from about step 5k and finishes about 0.18 dB higher on val.
- **Medium:** the two curves are indistinguishable.
- **Severe:** `Tnone` leads slightly up to about step 9k. The curves **cross at about step 12k**, after which `Tfull` stays ahead and the gap widens slowly to 0.07 dB at step 20k. Val severe SSIM shows the same pattern (0.6831 vs 0.6801). The validation curves therefore agree with the test result on severe images, so the severe result is not a test-set artefact.

**Under- or overfitting**
- **No overfitting.** Val PSNR rises monotonically for both models, the best epoch is the last one, and train and val improve together.
- **Mild under-convergence.** Val was still improving by about 0.015 dB per epoch at the end, while the cosine schedule had already brought the learning rate near zero. Both models have flattened but not fully plateaued.
- A longer schedule could still change the gaps. In particular, the `Tfull` lead on severe was still widening when training stopped.

## 4. Compute

| Measure | `Tfull` | `Tnone` | `Tnone` vs `Tfull` |
|---|---:|---:|---:|
| Parameters | 44.41M | 22.52M | −49% |
| GMACs at 256² | 76.3 | 46.4 | −39% |
| Latency at 256², batch 1 (MPS, median of 30) | 57.2 ms | 37.7 ms | −34% |
| Throughput at 256², batch 4 | 24.8 img/s | 38.9 img/s | +57% |
| Training s/step (median epoch) | 0.841 | 0.615 | −27% |
| Training wall-clock, 20k steps including 20 val passes | 296 min | 215 min | −27% |
| Peak training MPS memory | 7.19 GB | 5.77 GB | −20% |

The training timings are valid here because the Mac stayed awake under `caffeinate`, unlike the Stage 2 runs for seeds 42 and 123.

## 5. Interpretation

1. **The Stage 2 advantage for the encoder–decoder was not retained.**
   - At 800 steps, `Tnone` was better at severe PSNR by +0.137 dB, with a 95% CI excluding 0.
   - At 20k steps, severe PSNR is −0.039 dB, with a 95% CI of [−0.148, +0.065], so the result is inconclusive.
   - The short-budget advantage looks like a faster-early-learning effect: the smaller model learns faster at the start, and the curves show the same thing on medium and severe.
2. **Practical equivalence holds only in part.**
   - Mild: equivalent or better on every metric (SSIM and LPIPS significantly better).
   - Medium: equivalent on SSIM and LPIPS; PSNR is inconclusive (−0.08 dB, with a CI that crosses the margin on the worse side).
   - So "`Tnone` is practically equivalent to `Tfull`" is **not** established at this budget.
3. **On severe damage, the translators now help, mainly in perceptual quality.**
   - Severe LPIPS is 0.0088 better for `Tfull`. That is beyond the 0.005 margin, with d_z = 0.52 and 69% of images favouring `Tfull`.
   - Severe SSIM is 0.0036 better for `Tfull`, just beyond its margin.
   - This is the **first result in Stages 0–3 favouring the translators**. It appears only late in training (curves cross at about step 12k) and only on severe damage.
   - It is also the opposite of the Stage 2 severe result, so the claim "latent translators are redundant" (Stage 2 §5–6) **does not survive a longer budget without qualification**.
4. **The compute advantage of `Tnone` is large and confirmed:** half the parameters, 39% fewer MACs, 34% lower latency, 57% higher throughput, and 27% faster training. The real trade-off is equal or slightly better mild and medium quality, against a small but significant perceptual loss on severe damage.
5. **The one-seed caveat decides how much weight this carries.**
   - The severe LPIPS gap (0.0088) is about 1.4 times the seed-to-seed SD of LPIPS for model A at severe (0.0063, see the pre-registration).
   - A single training run therefore cannot separate a real architectural effect from seed noise. The image-level p-values measure consistency across test images within this one run, not reproducibility across training runs.
   - Stage 2 showed per-seed severe-PSNR differences changing sign.

## 6. What this does and does not support

- **Supported (exploratory):** at 20k steps, removing the translators gives equal or better quality on mild damage, equal quality on medium damage, and somewhat worse perceptual quality on severe damage, at roughly half the cost.
- **Not supported:**
  - That `Tnone` is better than `Tfull` at convergence.
  - That the two models are equivalent at convergence.
  - That translators are generally redundant.
  - Any seed-level generalisation.
- **Unchanged from earlier stages:** gating (Stage 0) was not part of this pilot. Nothing here revisits the finding that damage-conditioned gating does not help.

## 7. Limitations

1. One training seed, so effects are image-level only.
2. 20k steps with a cosine decay to near zero. Both models were still improving slightly, and the severe gap was still widening.
3. Pre-registered margins, set by the authors.
4. Synthetic, in-distribution degradations on one 256×256 face dataset.
5. The pilot was not designed to test gating, other baselines (Restormer, PromptIR, NAFNet), or real degradations.

## 8. Possible next step (not started; awaiting approval)

A 3-seed run of `Tfull` vs `Tnone` at this protocol (about 8.5 h per seed, about 25 h in total) would show whether the severe-damage perceptual advantage of the translators is reproducible. It would also allow seed-level equivalence and non-inferiority under the pre-registered margins. The positioning in the README and manuscript should stay unchanged until that question is settled. As instructed, no further training has been started.

## Files

| File | Contents |
|---|---|
| `train.log` | Training log for both runs |
| `seed42/{Tfull,Tnone}_history.json` | Per-epoch training and validation curves |
| `seed42/*_manifest.json` | Training manifests |
| `seed42/*_best.pth`, `*_last.pth` | Checkpoints (git-ignored) |
| `checkpoint_integrity.json` | Checkpoint integrity results |
| `per_image/*_seed42.json` | Per-image test metrics, with `*_eval_manifest.json` |
| `predictions/*.npz` | Raw test predictions (git-ignored, regenerable) |
| `stats.json`, `stage3_tables.md` | Statistics and the tables in this report |
| `profiling.json` | Latency, throughput and MACs |
| `learning_curves.png` | Learning-curve figure |
| `preflight_profile.json`, `recovery_test.json` | Pre-training checks |
| `stage3.log` | Analysis log |
