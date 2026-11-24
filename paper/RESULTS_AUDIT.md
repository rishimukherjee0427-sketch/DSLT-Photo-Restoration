# Results Audit for `DSLT_research_paper_draft.md`

*Prepared 24 November 2026, from repository state `5d01d02`; updated in QC pass 1, the same day. Nothing here is fabricated: every number below was copied from the named file. No training was run while preparing this audit.*

## 1. Source files

| Key | File | Producer |
|---|---|---|
| S0R | `results/v7_stage0/STAGE0_REPORT.md` | Stage 0 report |
| S0T | `results/v7_stage0/stage0_tables.md` | `experiments/stage0.py` |
| S1R | `results/v7_translator_ablation/REPORT.md` | `experiments/translator_ablation.py` |
| S2R | `results/v8_translator_training/REPORT.md` | Stage 2 report |
| S2T | `results/v8_translator_training/stage2_tables.md` | `experiments/stage2.py` |
| S2P | `results/v8_translator_training/PREREGISTRATION.md` | Committed in `e235240`, before Stage 2 training |
| S2A | `results/v8_translator_training/architecture_verification.json` | `experiments/stage2.py` |
| S3R | `results/v9_convergence_pilot/REPORT.md` | Stage 3 report |
| S3T | `results/v9_convergence_pilot/stage3_tables.md` | `experiments/stage3.py` |
| S3H | `results/v9_convergence_pilot/seed42/{Tfull,Tnone}_history.json` | `experiments/run_ablation.py` |
| S3M | `results/v9_convergence_pilot/seed42/*_manifest.json` | `experiments/run_ablation.py` |
| S3PF | `results/v9_convergence_pilot/preflight_profile.json`, `recovery_test.json` | `experiments/profile_pilot.py` |
| V7 | `results/v7_corrected/REPORT.md` | `experiments/stats_v7.py` |
| AUD | `AUDIT.md` | Project audit |
| RM | `README.md` (benchmark section; degraded-input row) | `results/benchmark_table.md` |
| DL | `utils/dataloader.py` header | Severity definitions |

## 2. Every numerical result used in the manuscript

### 2.1 Dataset and protocol

| Value | Used in | Source |
|---|---|---|
| 5,000 clean images; 368 md5-duplicate groups; 621 pHash/dHash clusters covering 2,025 images | §6 | AUD §C2 |
| Split 4,279 / 250 / 250; 221 excluded; split seed 42 | Abstract, §6 | AUD, RM |
| Degradation parameter table | §7 | DL |
| Degraded-input scores 25.60/0.7847/0.2482, 21.99/0.7074/0.3866, 15.25/0.5111/0.6844 | §7 | RM, AUD §5 (benchmark identity row) |
| 8 of 250 mild validation inputs receive no degradation | §8 | AUD M2 |
| 800-step protocol: 8×100, crop 128, batch 4, AdamW 2e-4, warmup 50, EMA 0.99, val 24 images | §6 | S2P, V7 §1 |
| 20k-step protocol: 20×1000, crop 256, warmup 500, EMA 0.999, val 250 images | §6, §11 | S3R header, `experiments/run_pilot.sh` |
| AdamW β = (0.9, 0.99), weight decay 1e-4, cosine decay to 0.5% of lr | §6 | `experiments/run_ablation.py` (lines 17–18, 288–292, 338–339) |
| Loss weights 0.01 / 0.25 / 0.02 / 0.05 / 0.03 / 0.15 | §6 | README Architecture section |
| No AMP speed-up (1.00–1.01×); BF16 gradient cosine 0.17 | §6 | V7 §1 |
| Resume test: max \|Δ\| = 0 | §6 | S3PF (`recovery_test.json`) |
| Gate supervision w = 0.5, sparsity w = 0.01 | §5 | V7 §1 |

### 2.2 Architecture and costs

| Value | Used in | Source |
|---|---|---|
| Encoder 8.95M, decoder 13.57M, damage estimator 1.56M, gate modules 0.09M, total 46.06M | §5 | README Architecture |
| Inference parameters: Tfull 44.41M, Te2 23.02M, Tnone 22.52M; translators 21.89M | §5, §12 | S2A |
| Per-translator parameters / GMACs: e2 0.50M / 8.2; e3 2.67M / 10.9; e4 8.02M / 8.2; bn 10.69M / 2.7; total 21.88M / 29.9 | §5, §12 | S1R §1 |
| s_k initial value 0.2; trained 0.18–0.21 | §5 | S1R §1 |
| Gating cost: +3.7% params, +8% MACs, +18% latency; B/C 82.5 GMACs, 68.1 ms, 20.9 img/s; O 83.5 GMACs, 73.1 ms, 19.9 img/s; A 57.5 ms | §9, §12 | S0R §1 |
| Te2 54.5 GMACs @ 256² | §12 | S2R §3 |
| Stage 3 cost: params 44.41/22.52 M; GMACs@128² 19.1/11.6; GMACs@256² 76.3/46.4; latency 57.2/37.7 ms; throughput 24.8/38.9 img/s | §12 | S3T |
| Training 0.841/0.615 s/step; 296/215 min; peak 7.19/5.77 GB | §12 | S3T, S3M |
| Relative figures −49%, −39%, −34%, +57%, −27%, −20%; 1.6× MACs, 1.5× latency, 37% more time per step | Abstract, §12, §13 | Calculated from the rows above (rounded) |
| Latency range for one architecture across sessions: 57.2–58.5 ms | §12 | S3T (57.2), S2R §3 (58.5), S0R (57.5) |

### 2.3 Stage 0 (3 seeds, 800 steps)

| Value | Source |
|---|---|
| Table 1: A 26.72±0.10 / 24.07±0.07 / 18.72±0.28; C 26.72±0.14 / 24.07±0.06 / 18.80±0.24; O 26.73±0.15 / 24.09±0.09 / 18.81±0.31 | S0R §2, S0T |
| Table 2: all O−A, O−C, C−A Δ, CIs and Holm p | S0R §2 |
| O−A severe per-seed Δ: +0.038, +0.115, +0.126 | S0R §2, S0T |
| B−A medium −0.110, p = 4.9e-05; B−A LPIPS +0.0013 / +0.0024 / +0.0031; C−B +0.104 (medium), +0.111 (severe) | S0R §4 |
| Gates: B e3/e4 0.002–0.019; B bottleneck 0.83 (seed 42) vs. 0.16–0.23 (other seeds); C 0.28–0.44 (mild) → 0.79–0.85 (severe) | S0R §3 |
| Forcing G = 1 changes PSNR by ≤ 0.04 dB (largest: +0.041) | S0R §3 |
| Gated LPIPS penalty +0.0006 to +0.0031 | S0R §5.4 |
| 12 checkpoints pass integrity checks | S0R §0 |

### 2.4 Stage 1 (inference-time, 3 seeds)

| Value | Source |
|---|---|
| All translators off: −0.098 / −0.078 / −0.096 dB | S1R §3 |
| e3, e4, bn off: ≤ 0.01 dB each | S1R §3–4 |

### 2.5 Stage 2 (3 seeds, 800 steps)

| Value | Source |
|---|---|
| Table 3 (all means, SDs, SSIM and LPIPS) | S2T, S2R §1 |
| Table 4 (Δ, 90% CI, verdicts); severe PSNR 95% CI [+0.033, +0.266], Holm p = 8e-08 | S2R §2 |
| Te2 per-seed severe PSNR Δ +0.146, −0.286, −0.058 | S2R §2 notes |
| Seed SD up to 0.36 dB | S2R §7, S2T (Tnone severe ± 0.36) |
| Margins 0.10 dB / 0.0020 / 0.0050 | S2P |
| A severe LPIPS seed SD 0.0063 | S0T (A severe LPIPS ± 0.0063), S2P |

### 2.6 Stage 3 (seed 42 only, 20k steps)

| Value | Source |
|---|---|
| Table 5 (all 18 test numbers) | S3T, S3R §1 |
| Table 6 (Δ, 90% CI, d_z, %, Holm p, verdicts) | S3T, S3R §2 |
| Severe PSNR 95% CI [−0.148, +0.065] | S3T |
| Best epoch 20 for both models; best val 26.533 / 26.572 | S3T, `checkpoint_integrity.json` |
| Severe val PSNR Δ per epoch: +0.167, +0.103, ≈ 0, −0.047 to −0.101 (5k–8k), +0.005 to +0.011 (9k–13k), +0.020 (14k), +0.056 (16k), +0.075 (20k) | S3H (computed directly from the histories during this audit) |
| Mild val Tnone lead about 0.16–0.22 dB from about 5k, final 0.175 dB | S3H (computed) |
| Severe val SSIM at the final epoch 0.6831 vs. 0.6801 | S3H, S3R §3 |
| End-of-training val gain about 0.015 dB per epoch | S3R §3 (epochs 18→20: +0.030 / +0.033 dB over 2 epochs) |
| 750/750 inputs alignment-checked | `per_image/*_seed42.json` → `info.alignment_checked_inputs` |

## 3. Unresolved inconsistencies

| ID | Issue | Effect on the manuscript | Action needed |
|---|---|---|---|
| I-1 | **Crossover step. Corrected in the manuscript (QC pass 1).** S3R §3 says the severe curves "cross at about step 12k". The per-epoch S3H data give Tfull − Tnone severe val PSNR of +0.007, +0.010, +0.011, +0.005, +0.011 dB at 9k–13k, +0.020 at 14k, then rising to +0.075 at 20k; Tnone led by 0.047–0.101 dB at 5k–8k. | The manuscript now describes a *nominal* lead from about 9k that stays ≤ 0.011 dB through 13k and widens over the last ~6k steps. It gives no precise crossover point, and notes that even the final lead (0.075 dB) is below the 0.10 dB margin. | **Still open:** S3R itself (a committed historical report) still says "≈12k". Either add an erratum note to S3R or leave it and rely on the manuscript. No validation LPIPS was logged, so there is no LPIPS learning curve. |
| I-2 | **Hardware name. Resolved in the manuscript.** All 58 run manifests under `results/` record `"chip": "Apple M4 Pro"` (24 GB, torch 2.15.0). Reports that say only "Apple M4": `results/v7_stage0/STAGE0_REPORT.md:19` and `results/v7_corrected/REPORT.md:19`. README and AUDIT.md also say "M4". | The manuscript now says Apple M4 Pro, 24 GB, PyTorch 2.15.0 (MPS). | Historical reports and README were not modified, as instructed. Optionally add errata. |
| I-3 | **DSLT-A (Stage 0) vs. Tfull (Stage 2): mechanism verified; full causal attribution not tested.** Same inference architecture, but severe PSNR differs (3-seed means 18.72 vs. 18.81 dB; seed 2026: 18.69 vs. 18.91). **Verified (QC pass 1, CPU, no training):** constructing config A (gate_mode='film') and Tfull (gate_mode='none') with `seed_all(s)` for s ∈ {42, 123, 2026} gives *bit-identical* initial weights for the damage estimator, encoder and all four translators. The **decoder initialisation differs** (max \|Δ\| ≈ 0.69–0.70), because A additionally builds 24 gate-modulator tensors (87,788 params) after the translators and before the decoder, consuming RNG. **Also verified:** checkpoint metadata is otherwise identical (crop, epochs, steps/epoch, EMA, vgg_normalize, use_gating=False; same chip and torch). The gate modules get no gradient in A, so AdamW skips them. Selected best epochs differ (seed 42: A epoch 8 vs. Tfull epoch 6), a downstream consequence. | The manuscript (§5) states the verified mechanism and does not claim it explains the whole gap. | **Unresolved:** (a) whether decoder-init variation alone accounts for the gap; it is consistent with the observed seed-to-seed SD (up to 0.36 dB) but untested in isolation; (b) MPS run-to-run nondeterminism, not measured; (c) *checked, resolved:* `git diff e612ad1 e235240` and `git diff 2cc4a88 e235240` on `experiments/run_ablation.py`, `models/` and `utils/` show only new configs (O, Tfull, Te2, Tnone), the oracle path, the `translators` / `gate_mode` constructor arguments and extra metadata. The ungated training path is unchanged (`severity_map=sev.detach()` in both). A definitive attribution would need a re-run of A with the gate modules constructed after the decoder, which is training and was not authorised. |
| I-4 | **Latency for the same architecture:** 57.5 ms (S0R), 58.5 ms (S2R), 57.2 ms (S3T). | Disclosed in §12 as about ±1 ms session variation. | None beyond disclosure. |
| I-5 | **Holm p for C−A severe:** 6.9e-06 in S0R §2 (oracle family) vs. 5.5e-06 in S0R §4 (A/B/C family). Two different 27-test families give different adjustments. | The manuscript uses 6.9e-06 with the oracle family. | Name the family next to every Holm p in the final text. |
| I-6 | **Two "C − A severe" values:** +0.076 dB (3-seed Stage 0) vs. +0.071 dB (2-seed V7). Different seed sets. | Only the 3-seed value is used. | None. Do not cite the V7 2-seed numbers as the main result. |
| I-7 | **Gate–severity correlations 0.64–0.78** in README and AUDIT come from the legacy seed-42 ablation, trained with the un-normalised VGG loss. | Not used in the manuscript. Stage 0 gate values are used instead. | Re-measure on the Stage 0 checkpoints if gate–severity correlation is to be reported. |
| I-8 | **Stage 2 training times** for seeds 42 and 123 are invalid (machine sleep). Inference peak memory reads 2.29 GB for every model (not separable). | Stage 2 training time and inference memory are not used. Training cost comes only from Stage 3 (under caffeinate). | None. |
| I-9 | **Validation SSIM vs. test SSIM.** Validation SSIM is computed during training (S3H) and test SSIM by the benchmark evaluator. Both use skimage, but this was not cross-checked. | Validation SSIM is used only descriptively (0.6831 vs. 0.6801). | Optional: confirm both use identical SSIM settings. |
| I-10 | **Resolved.** AdamW β = (0.9, 0.99), wd 1e-4, grad clip 1.0 and cosine decay to 0.5% of lr (`min_lr_ratio` 0.005) were confirmed in `experiments/run_ablation.py` lines 17–18, 288–292 and 338–339. | §6 is updated. | None. |

## 4. Information still missing before submission

1. **Dataset provenance.** Source, licence, citation, collection and content (described only as "predominantly faces"), and the original resolution. The raw folder contains unnamed Kaggle-style archives (`archive (n).zip`, `MLRN (1).zip`); their origin must be documented.
2. **Method name.** "DSLT / Deep Space Latent Translation" clashes with Wan et al.'s "Deep Latent Space Translation" (CVPR 2020; TPAMI 2023, DOI 10.1109/TPAMI.2022.3163183). Recommended replacement: **RLT-Net**. See `MANUSCRIPT_REVIEW.md` §4. The authors must choose a name; no files have been renamed.
3. **References.** All 25 were checked on 24 Nov 2026 (`MANUSCRIPT_REVIEW.md` §5). The PromptIR title was corrected. 11 DOIs were confirmed from publisher or primary records (refs 2, 3, 4, 5, 8, 12, 15, 16, 21, 22, 25). 9 still need an IEEE Xplore, JSTOR or Crossref check (refs 1, 6, 9, 10, 13, 14, 18, 23, 24), and ref 11 has no DOI located. Refs 2, 4, 11, 12, 13 and 16 have minor open details (authors or pages). Refs 26–27 are TODO.
4. **External baselines.** None are valid under the current protocol. The legacy Restormer-tiny / PromptIR-tiny results (README benchmark table) were single-seed and trained with the un-normalised VGG loss, so they are not comparable and are excluded. Any venue will likely require fair baselines (Restormer, PromptIR, NAFNet; parameter- or MAC-matched; same protocol; 3 seeds).
5. **Stage 3 replication.** Seeds 123 and 2026 at the 20k-step protocol, about 17 h. The run was started and then stopped by the user before the first checkpoint, so there is no data. The runner (`experiments/run_pilot_seeds.sh`) and analysis script (`experiments/stage3_seeds.py`) exist but are uncommitted.
6. **Real-photograph evaluation.** At least qualitative results on real damaged photos (87 photos in `data/raw` per S0R §7), with no-reference metrics as supplementary evidence.
7. **Figures.**
   - Architecture diagram.
   - Learning curves: `results/v9_convergence_pilot/learning_curves.png` exists.
   - Visual comparisons: `results/v8_translator_training/visual_comparison_seed42.png` and `results/v7_translator_ablation/visual_comparison_seed42.png` exist.
   - A Stage 3 visual comparison has not been produced.
8. **Gating at the longer budget.** Stage 0 conclusions are only at 800 steps. Either re-test or keep the scope statement.
9. **Translator attribution at 20k steps.** It is unknown whether the Stage 3 severe effect is again concentrated in e2. This would need a Stage 1-style intervention on the Stage 3 Tfull checkpoint, which is inference only and cheap.
10. **Authors, affiliations, acknowledgements, data/code availability statement and target venue formatting.**
11. **Statistical methods text.** The bootstrap replicate count (10,000) and bootstrap RNG seed should be stated. Both are in the code (`experiments/stage2.py`, `experiments/stage3.py`) and not yet in the manuscript.

## 5. Claims that must remain exploratory or scoped

| Claim | Required wording |
|---|---|
| Full DSLT-A has a severe-damage perceptual advantage (LPIPS +0.0088, SSIM +0.0036, 69% of images, d_z 0.52) | **Exploratory, one seed, image-level only.** Must not be called reproducible or general. |
| Equivalence or non-inferiority of the encoder–decoder at 20k steps (mild, medium) | Exploratory (one seed). Partial: medium PSNR is inconclusive. |
| Late-training crossover on severe damage | Observed in one run only. Give the per-epoch numbers, not a single crossover step. |
| "Translators are redundant" | **Must not be claimed.** Supported only at 800 steps and 128 px (Stage 2), and contradicted on severe perceptual metrics at 20k steps. |
| "Translators are beneficial" | **Must not be claimed** in general. At most: a possible severe-damage perceptual benefit after long training, pending multi-seed confirmation. |
| Gating does not help | Scope it to the 800-step, 128-px budget and the FiLM-gate injection pathway. Not tested at convergence. |
| Oracle upper bound < 0.1 dB | Same scope as the gating claim. |
| Efficiency advantage of the encoder–decoder | Well supported: architectural, so seed-independent. Timings are specific to one Apple-silicon GPU in FP32. |
| Any comparison with the state of the art | Not supported. No valid external baselines. |
| Generalisation to real photographs or other datasets | Not supported. Synthetic, in-distribution degradation on one dataset. |

## 6. Excluded evidence (must not appear as benchmark results)

| Item | Reason |
|---|---|
| `results/best_model_v5.pth` and every v5 metric (for example 29.90 dB mild, "30.16 dB") | Trained on all images, including val/test: contaminated (AUD C1) |
| `results/metrics*.json`, `results/ablation.csv`, older U-Net and v4 rows | Unknown or inconsistent evaluation conditions (AUD M3) |
| Legacy seed-42 A/B/C and Restormer/PromptIR benchmark (README table) | Un-normalised VGG loss and a single seed; a separate series that cannot be pooled with v7+ |
| V7 two-seed gating series as the main result | Superseded by the 3-seed Stage 0. It may be cited only as part of Stage 0 provenance (seeds 123 and 2026 are reused) |
| Partial seed-123 Stage 3 run | Stopped before any checkpoint was saved; no data |
