# Rethinking Latent Feature Translation for Multi-Severity Old Photograph Restoration: An Empirical Study of Restoration Quality and Computational Efficiency

*First draft, 24 November 2026; quality-control pass 1 applied (see `paper/MANUSCRIPT_REVIEW.md`). Every number is traced to a repository file in `paper/RESULTS_AUDIT.md`. Items marked **[VERIFY]** need checking before submission, and items marked **[TODO]** are missing.*

> **Naming note (unresolved):** "DSLT" (Deep Space Latent Translation) is too close to Wan et al.'s "Deep Latent Space Translation" [1, 2] and must be replaced before submission. The recommended replacement is **RLT-Net** (Residual Latent Translation Network); see `MANUSCRIPT_REVIEW.md` §4. This draft keeps "DSLT" until the authors choose a name, so the text still matches the repository.

**Authors:** [TODO: author names, affiliations, corresponding author]

---

## Abstract

Old-photograph restoration networks often add specialised modules to a convolutional encoder–decoder, such as latent feature translators or damage-conditioned gates. Whether these modules justify their cost is not always tested under controlled, multi-seed conditions.

We study DSLT-A, a 44.4M-parameter multi-scale restoration U-Net. It has residual latent "translators" at three skip scales and at the bottleneck, plus an optional damage-conditioned FiLM-style gate. We evaluate it on a deduplicated split of 4,279 training, 250 validation and 250 test images, with synthetic degradation at three severities.

The study has three stages under one evaluation protocol: PSNR, SSIM and LPIPS on cached test inputs, validation-only model selection, paired image-level tests, and pre-registered equivalence margins.

- **Stage 0 (3 seeds, 800 steps):** damage-conditioned gating gives no practically meaningful gain. Even oracle ground-truth severity conditioning improves PSNR over the ungated model by at most +0.093 dB, on severe damage only. Unsupervised gates collapse.
- **Stage 2 (3 seeds, 800 steps):** an encoder–decoder trained without any translators (22.5M parameters) is non-inferior to DSLT-A on every severity and metric. Its PSNR on severe damage is higher by +0.137 dB (95% CI [+0.033, +0.266]).
- **Stage 3 (one seed, 20,000 steps at 256×256; exploratory):** this short-budget advantage disappears. The encoder–decoder is equivalent or non-inferior on every mild-damage metric and on medium SSIM and LPIPS. PSNR differences at medium and severe damage are inconclusive. On severe damage, DSLT-A has better test LPIPS in this run: 0.4264 vs. 0.4352, better on 69% of images, image-level d_z = 0.52. It also has slightly better SSIM. Both differences exceed the pre-registered margins at the image level.

The encoder–decoder needs 49% fewer parameters, 39% fewer multiply–accumulates, 34% lower latency, 27% less training time per step and 20% less peak training memory. Because Stage 3 uses a single training seed, the severe-damage advantage of the translators is exploratory.

Our results show three things about this architecture family:
1. Conclusions about module usefulness depend on the training budget.
2. Damage-conditioned gating is not supported at the short (800-step) budget where it was tested.
3. Latent translators trade a possible, not yet replicated, severe-damage perceptual benefit for a large computational cost.

**Keywords:** image restoration; old photograph restoration; ablation study; latent feature translation; damage-conditioned gating; equivalence testing; computational efficiency; encoder–decoder networks

---

## 1. Introduction

Photographs degrade through noise, blur, scratches, fading, colour casts and compression. Learning-based restoration has largely replaced hand-crafted pipelines.

Within that literature, a recurring design pattern adds specialised components to a convolutional encoder–decoder. One is *latent translation* modules, which map degraded-domain features toward clean-domain features. Wan et al. used latent-space translation for old-photo restoration [1, 2]. Another is *damage-aware conditioning*, which modulates features according to an estimate of degradation type or severity [8, 9].

Such components are usually justified by improvements over a baseline. Those improvements can be small, can come from a single training run, and are typically measured at a single training budget, sometimes without their computational cost reported next to them [TODO: reference needed, see ref. 27].

This paper started as the development of DSLT, a multi-scale latent-translation network with damage-conditioned adaptive gating. During development we found that our early evaluation was invalid. Reported metrics had been computed on training images, and the dataset contained near-duplicate images that would have leaked across a naive split. We rebuilt the protocol and then asked a narrower question:

> Under what degradation severity and training regime does latent feature translation provide a measurable restoration benefit, and is that benefit worth its computational cost?

We answer it in three controlled stages:
- **Stage 0** asks whether damage-conditioned gating helps, including an oracle upper bound.
- **Stage 2** trains architectures with and without translators independently, at a short budget, across three seeds.
- **Stage 3** repeats the decisive comparison at a 25× longer budget and higher resolution, with one seed. This was a pilot, not a confirmatory study.

An inference-time intervention study (Stage 1) motivates Stage 2 and is summarised briefly.

The results are mixed, and we report them as such. The short-budget conclusion, that translators are unnecessary, does not survive longer training without qualification. The longer-training advantage of the translators is limited to severe damage and to LPIPS and SSIM (not PSNR), and it rests on a single seed.

## 2. Research motivation and contributions

**Motivation.**
1. *Cost-blind component claims.* Translators make up 49% of DSLT-A's parameters and 39% of its compute (§12). Any benefit has to be weighed against that cost.
2. *Single-budget conclusions.* Ablations at one training length can mistake faster early learning for better final performance. Stages 2 and 3 show exactly this.
3. *Evaluation hygiene.* Our own earlier numbers were inflated by train/test contamination. A careful re-evaluation is a contribution in itself for a field that often reports single-run differences of a few hundredths of a dB.

**Contributions.**
1. A controlled, leakage-free evaluation protocol for multi-severity synthetic old-photo restoration:
   - near-duplicate-aware splits;
   - cached deterministic test degradations at three severities;
   - one shared metric implementation;
   - validation-only model selection;
   - paired statistics with Holm correction and pre-registered equivalence margins.
2. A negative result on damage-conditioned gating, at an 800-step budget with three seeds:
   - unsupervised gates collapse;
   - supervised gates track damage, but the network does not depend on them;
   - even oracle severity information gives < 0.1 dB PSNR.
3. An independently trained translator ablation (three seeds, 800 steps). At this budget, removing every translator halves the parameter count without loss of quality.
4. A convergence pilot (one seed, 20,000 steps). It shows the short-budget conclusion depends on the budget: in this single run, with longer training, the translator model recovers parity and gains a severe-damage perceptual advantage. Whether that advantage reproduces across seeds is untested.
5. A quantified efficiency comparison: parameters, MACs, latency, throughput, training time and peak memory.

We do **not** claim a new state of the art. We do not claim that translators are universally redundant, or that they are beneficial in general.

## 3. Related work

*Numbered citations refer to the reference list, which was checked on 24 Nov 2026. Remaining [TODO] items still need sources.*

**Old-photograph restoration.**
- Wan et al. [1, 2] proposed restoring old photos with two variational autoencoders that map old and clean photos into latent spaces, then learning the translation between those latent spaces from synthetic paired data. A global branch with a partial non-local block handles structured defects such as scratches and dust spots. Our model's working name, "Deep Space Latent Translation" (DSLT), is too close to their "Deep Latent Space Translation" and will be changed (see the naming note at the top).
- Our translators differ in mechanism: they are residual blocks `z + s·Blocks(z)` inside a single supervised encoder–decoder, not mappings between separately learned latent spaces.
- Other work addresses old-photo colourisation, scratch detection and face-specific restoration [TODO: references needed, see ref. 27].

**General image restoration architectures.**
- *Encoder–decoder.* U-Net [3] established the encoder–decoder with skip connections, which DSLT also builds on.
- *Residual CNNs.* Residual-in-residual dense blocks (RRDB) [4], attention modules such as CBAM [5], sub-pixel convolution (PixelShuffle) [6] and dilated convolutions [7] are standard components.
- *Simple baselines and transformers.* NAFNet [12] showed that a simple, carefully designed baseline can match more complex restorers. Restormer [10] is a common transformer-based reference.

**Degradation-aware and all-in-one restoration.**
- *Feature modulation.* Feature-wise linear modulation (FiLM) [8] and spatial feature transform (SFT) [9] condition network features on side information.
- *Degradation-type conditioning.* All-in-one methods such as AirNet [13] and PromptIR [11] condition restoration on learned representations or prompts of the degradation type.
- *Our gate.* Our damage-conditioned gate is a FiLM-style instance of this idea, applied to the strength of latent translation.

**Losses and metrics.**
- *Training losses.* We use a Charbonnier loss [18] and a VGG-based perceptual loss [16, 17].
- *Evaluation metrics.* We evaluate with PSNR, SSIM [15] and LPIPS [14].

**Rigorous evaluation in deep learning.**
- *Seed variance.* Several studies show that differences between runs (seed variance) can exceed reported architectural gains [TODO: reference needed, see ref. 27].
- *Equivalence testing.* Two one-sided tests (TOST) [21, 22] lets one claim the *absence* of a practically meaningful difference, which a non-significant test cannot.
- *Multiple comparisons.* We use Holm's step-down correction [23].
- *Paired tests and resampling.* We use the Wilcoxon signed-rank test [24] and bootstrap resampling [25].

## 4. Methodology

### 4.1 Research questions

- **RQ1 (Stage 0):** does damage-conditioned gating of latent translation improve restoration? How much could perfect damage information add?
- **RQ2 (Stages 1–2):** do latent translators contribute to restoration quality, (a) in a trained network (inference-time intervention) and (b) as an architectural choice (independent training)?
- **RQ3 (Stage 3):** does the RQ2 answer change with longer training at higher resolution?
- **RQ4:** what is the computational cost of each component?

### 4.2 Controlled-comparison principle

Within each stage, every configuration is identical except the component under study:
- same code path, data, loss, optimiser, schedule and number of optimizer steps;
- same per-epoch data order and degradation sampling;
- same validation set and selection rule;
- one evaluation on the test set.

Models are selected **only** on validation PSNR, the mean over the three severities. The test set is never used for selection.

### 4.3 Statistical procedure

All comparisons are paired over the same 250 test images.

- **Paired differences.** For each image, severity and metric we compute the paired difference between two models. With several seeds, per-image differences are averaged over seeds.
- **Significance.** Two-sided Wilcoxon signed-rank test on the per-image differences, Holm-corrected within each stage's family of tests.
- **Confidence intervals.** Multi-seed stages use a two-level bootstrap that resamples seeds and then images. Stage 3, with one seed, uses an image-level bootstrap.
- **Effect size.** Cohen's d_z on the paired differences.
- **Equivalence and non-inferiority** (Stages 2–3). Margins were fixed in a pre-registration committed before any Stage 2 training: PSNR ±0.10 dB, SSIM ±0.0020, LPIPS ±0.0050. Verdicts:
  - *equivalent*: the 90% CI lies entirely within the margin;
  - *non-inferior*: the CI does not cross the margin on the unfavourable side;
  - *worse beyond margin*: the whole 90% CI lies beyond the unfavourable margin;
  - *inconclusive*: any other case.
- **Two kinds of evidence.** Image-level significance says that *these trained models* differ consistently across test images. It does not say the difference would reappear with new training seeds. Seed-level robustness is judged from per-seed differences and seed-resampling CIs. With three seeds this evidence is limited, and with one seed (Stage 3) it is absent.
- **Practical threshold.** A PSNR difference counts as practically meaningful only if it is ≥ 0.1 dB and robust across seeds.

## 5. Architecture descriptions

All variants share one encoder–decoder backbone (`models/autoencoder.py`, `models/latent_translator.py`, `models/damage_estimator.py`).

**Encoder (8.95M parameters).**
- Four levels with strided-convolution downsampling, CBAM attention and residual-in-residual dense blocks at the bottleneck. Channel widths are 64→128→256→512.
- It produces skip features e1–e4 and a bottleneck feature.

**Decoder (13.57M parameters).**
- PixelShuffle upsampling with skip connections and instance normalisation.
- It predicts a residual correction: the output is `clamp(x + correction, 0, 1)`.

**Latent translators (21.89M parameters across four scales).**
- At skip scales e2, e3 and e4 and at the bottleneck (bn), a translator `T_k(z) = z + s_k · Blocks_k(z)` replaces the feature passed to the decoder.
- `Blocks_k` is a stack of 3–4 dilated residual blocks, and `s_k` is a learned scalar (initial value 0.2; trained values 0.18–0.21 at 800 steps).
- e1 is never translated.
- Per-scale cost at 256² (parameters / GMACs): e2 0.50M / 8.2; e3 2.67M / 10.9; e4 8.02M / 8.2; bn 10.69M / 2.7.

**Damage estimator (1.56M parameters).**
- A lightweight CNN that predicts a 4-channel per-pixel severity map (blur, noise, scratch, fade) from the input image.
- It is trained with an L1 loss (weight 0.15) against ground-truth severity maps produced by the degradation pipeline.
- It is trained in **all** configurations, including ungated ones, where its output does not affect the restoration. Inference parameter counts below exclude it.

**Damage-conditioned gate (gated variants only; +0.09M parameters).**
- The translated feature is `e + G ⊙ (T(e) − e)`, with `G = σ(logit(g_scale) + Δ_channel + Δ_spatial)`:
  - `g_scale` is a per-scale base gate predicted from pooled e3/e4 features and the predicted severity map;
  - `Δ_channel` is an SE-style per-channel offset;
  - `Δ_spatial` is a per-pixel offset computed from the resized severity map.
- Both offsets are zero-initialised.

**Configurations studied.**

| ID | Stage | Translators | Gating | Inference params |
|---|---|---|---|---:|
| DSLT-A | 0, 1 | e2, e3, e4, bn | none (G ≡ 1) | 44.41M |
| DSLT-B | 0 | e2, e3, e4, bn | learned FiLM gate, unsupervised | 46.06M* |
| DSLT-C | 0 | e2, e3, e4, bn | learned FiLM gate + gate severity supervision (w = 0.5) + sparsity (w = 0.01) | 46.06M* |
| DSLT-O | 0 | e2, e3, e4, bn | C with ground-truth severity maps in place of predicted ones (diagnostic oracle; not deployable) | 46.06M* |
| Tfull | 2, 3 | e2, e3, e4, bn | none; no gate modules built | 44.41M |
| Te2 | 2 | e2 only | none | 23.02M |
| Tnone (encoder–decoder) | 2, 3 | none (identity) | none | 22.52M |

\*Includes the damage estimator and gate modules, which gated models use at inference.

DSLT-A and Tfull have the same inference architecture and training settings. DSLT-A, however, also builds the (unused) gate modules, 87,788 parameters that are created after the translators and before the decoder.

We verified on CPU that, for every seed used, DSLT-A and Tfull get bit-identical initial weights for the encoder, all four translators and the damage estimator, but a **different decoder initialisation**: the extra modules consume random numbers first. The unused modules never receive gradients, so they do not change the optimisation directly. The two series are therefore effectively independent training runs of the same inference architecture, and their results (for example, severe PSNR 18.72 vs. 18.81 dB, three-seed means) should not be read as repeated measurements of one model. Each stage's internal comparisons are unaffected, because within a stage all configurations share the same construction path. Whether the different decoder initialisation fully explains the gap has not been tested in isolation (`RESULTS_AUDIT.md`, I-3).

## 6. Experimental setup

**Data.**
- A set of 5,000 clean 256×256 photographs, predominantly face images [TODO: dataset source, licence and citation].
- The set contained exact duplicates (368 md5 groups) and near-duplicates created by resizing and flipping. Flip-aware perceptual-hash clustering (pHash + dHash) found 621 multi-image clusters.
- Validation and test take at most one image per cluster. The 221 remaining cluster members are excluded from all splits, which gives 4,279 train / 250 validation / 250 test images (split seed 42).
- Some images, including their clean references, are stored rotated or flipped.

**Degradation.**
- Training uses online synthetic degradation. Each sample's severity is drawn uniformly from mild, medium and severe ("mixed").
- Validation and test inputs are rendered once, deterministically (per-image seeds), and cached, so every model sees identical inputs.

**Training protocols.**

| | Stages 0, 1, 2 | Stage 3 |
|---|---|---|
| Steps | 800 (8 × 100) | 20,000 (20 × 1,000) |
| Crop / batch | 128 px / 4 | 256 px / 4 |
| Optimiser | AdamW, lr 2e-4, β = (0.9, 0.99), wd 1e-4; linear warm-up, then cosine decay to 0.5% of lr; grad clip 1.0 | same |
| Warmup / EMA | 50 steps / 0.99 | 500 steps / 0.999 |
| Validation for selection | 24 images × 3 severities, each epoch | 250 images × 3 severities, each epoch |
| Seeds | 42, 123, 2026 | 42 only |
| Precision / hardware | FP32, Apple M4 Pro GPU via PyTorch MPS, 24 GB unified memory; PyTorch 2.15.0, Python 3.12 | same |

- **Precision.** Mixed precision gave no speed-up on this hardware, and BF16 gradients diverged, so all runs use FP32.
- **Loss.** Charbonnier + 0.01·VGG16 multi-scale perceptual (ImageNet-normalised inputs) + 0.25·(1 − SSIM) + 0.02·LAB + 0.05·FFT + 0.03·edge + 0.15·damage-severity L1. Gated variants add gate losses as listed in §5.
- **Reproducibility.** Every training and evaluation run writes a manifest recording the git commit, split checksums, environment, configuration and checkpoint SHA-256. Checkpoints are saved atomically. A stop-and-resume test reproduced an uninterrupted run exactly (max |Δ| = 0 for weights, EMA and optimiser state).

**Baselines.** This study compares architectural variants of one backbone. It does **not** include a fair comparison with external restoration methods. An earlier single-seed comparison with reduced-size re-implementations of Restormer and PromptIR used a different perceptual-loss normalisation and is not comparable; it is not reported as evidence here (§14).

## 7. Damage severity definitions

Degradations are composed from 13 operations. Each is applied with a severity-dependent probability `p` and parameter range (`utils/dataloader.py`):

| Operation | Mild | Medium | Severe |
|---|---|---|---|
| Gaussian noise: p; σ (/255) | 0.60; U(3, 20) | 0.68; U(3, 27) | 0.85; U(5, 50) |
| Salt & pepper: p; amount | 0.10; U(.002, .015) | 0.17; U(.002, .022) | 0.40; U(.005, .050) |
| Gaussian blur: p; kernel | 0.35; {3, 5} | 0.42; {3, 5, 7} | 0.65; {5, 7, 9} |
| Heavy blur: p; kernel | 0 | 0.08; {9, 11} | 0.30; {11, 13, 15, 17, 21} |
| Motion blur: p; length | 0.10; {3, 5} | 0.15; {3, 5, 7, 9} | 0.35; {7, 9, 11, 15} |
| JPEG: p; quality | 0.30; 35–70 | 0.35; 22–55 | 0.55; 5–30 |
| Sepia: p; intensity | 0.25; U(.10, .40) | 0.32; U(.15, .55) | 0.55; U(.30, .85) |
| Colour fade: p; amount | 0.20; U(.10, .35) | 0.25; U(.15, .47) | 0.45; U(.30, .80) |
| Contrast reduction: p; factor | 0.30; U(.60, .90) | 0.40; U(.50, .85) | 0.65; U(.30, .70) |
| Grayscale: p | 0.08 | 0.19 | 0.45 |
| Scratches: p; count | 0.10; 1–3 | 0.20; 1–5 | 0.50; 3–15 |
| Vignette: p; strength | 0.15; U(.10, .30) | 0.20; U(.15, .40) | 0.35; U(.30, .65) |
| Down/up-scaling: p; scale | 0.15; U(.50, .80) | 0.22; U(.40, .75) | 0.45; U(.20, .60) |

**Input quality.** On the test split, the unrestored degraded input scores (PSNR / SSIM / LPIPS):

| Severity | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Mild | 25.60 | 0.7847 | 0.2482 |
| Medium | 21.99 | 0.7074 | 0.3866 |
| Severe | 15.25 | 0.5111 | 0.6844 |

**Ground-truth severity maps** have four channels in [0, 1]:
- blur = max kernel / 15;
- noise = max(σ·255/35, 0.4 if salt-and-pepper);
- scratch = mask;
- fade = max(sepia, fade, 1 − contrast).

Severe values saturate at 1.0.

## 8. Evaluation metrics

- **PSNR** (dB, per image, RGB in [0, 1], capped at 50 dB). The cap prevents near-identical images from inflating means: some inputs receive no degradation at all, for example 8 of the 250 mild validation inputs.
- **SSIM:** `skimage` implementation.
- **LPIPS:** AlexNet backbone (`lpips` package 0.1.4). Lower is better.

All metrics are computed by one shared implementation on the cached test inputs. Before metrics are computed, every prediction is checked for input alignment against the expected cached image (750/750 inputs per evaluation in every stage).

**Efficiency metrics.**
- *Parameters:* inference modules.
- *MACs:* PyTorch FLOP counter at 128² and 256², counting convolutions and matrix multiplications only.
- *Latency:* median of 30 forward passes after 10 warm-up passes, batch 1, FP32 on the MPS GPU, one model per process.
- *Throughput:* batch 4 at 256².
- *Training cost:* seconds per step, wall-clock time and peak MPS driver memory, all from training manifests.

---

## 9. Stage 0: damage-conditioned gating (3 seeds, 800 steps)

**Setup.** DSLT-A, -B, -C and -O, seeds 42, 123 and 2026, trained under the 800-step protocol.
- The A/B/C runs for seeds 123 and 2026 come from the same pipeline (`results/v7_corrected`) and were re-verified by checksum.
- The seed-42 runs and all oracle runs were trained in Stage 0.
- All 12 checkpoints passed integrity checks.

**Table 1.** Test PSNR (dB), mean ± SD over three seeds.

| Model | Mild | Medium | Severe |
|---|---:|---:|---:|
| A (ungated) | 26.72 ± 0.10 | 24.07 ± 0.07 | 18.72 ± 0.28 |
| C (learned conditioning) | 26.72 ± 0.14 | 24.07 ± 0.06 | 18.80 ± 0.24 |
| O (oracle conditioning, diagnostic) | 26.73 ± 0.15 | 24.09 ± 0.09 | 18.81 ± 0.31 |

**Table 2.** PSNR differences: mean [95% seed × image bootstrap CI], Holm-adjusted p (27 tests in the oracle family).

| Severity | O − A | O − C | C − A |
|---|---|---|---|
| Mild | +0.004 [−0.070, +0.077], p = 1 | +0.010 [−0.046, +0.062], p = 1 | −0.005 [−0.073, +0.054], p = 1 |
| Medium | +0.022 [−0.061, +0.104], p = 1 | +0.028 [−0.035, +0.091], p = 0.72 | −0.006 [−0.106, +0.077], p = 1 |
| Severe | +0.093 [+0.036, +0.147], p = 1.4e-10 | +0.017 [−0.054, +0.101], p = 1 | +0.076 [+0.006, +0.139], p = 6.9e-06 |

**Findings.**
1. **Oracle upper bound below the practical threshold.**
   - Perfect severity information adds +0.004 / +0.022 / +0.093 dB over the ungated model.
   - Only the severe gain is consistent across seeds: per-seed +0.038, +0.115 and +0.126 dB.
   - It still falls below the pre-set 0.1 dB criterion.
2. **The severity estimator is not the bottleneck.** Learned conditioning (C) is statistically indistinguishable from the oracle: no O − C test is Holm-significant.
3. **Unsupervised gating does not help.**
   - B is worse than A at medium (−0.110 dB, Holm p = 4.9e-05) and has worse LPIPS at every severity (+0.0013 / +0.0024 / +0.0031).
   - C − B is +0.104 dB at medium and +0.111 dB at severe, both Holm-significant. Severity supervision therefore mainly repairs unsupervised gating rather than improving on A.
4. **Gates collapse without supervision, and the networks do not rely on their gates.**
   - In B, the e3 and e4 gates fall to about 0.002–0.019 for every seed and severity. The bottleneck gate varies across seeds (0.83 vs. 0.16–0.23).
   - In C, gates track severity, rising from 0.28–0.44 (mild) to 0.79–0.85 (severe).
   - Forcing G = 1 in trained B and C changes PSNR by at most 0.04 dB, and no PSNR change is Holm-significant.
5. **Perceptual cost.** Every gated variant, including the oracle, has slightly worse LPIPS than A (+0.0006 to +0.0031).

**Cost.** Gating adds 3.7% parameters, 8% MACs (82.5 vs. 76.3 GMACs at 256²) and 18% latency (68.1 vs. 57.5 ms at 256²).

**Stage 0 conclusion:** at this budget, damage-conditioned gating offers no practically meaningful benefit and costs compute. Gating was therefore not carried into Stages 2–3. Stage 0 did not test whether gating helps after longer training (§14).

## 10. Stage 2: independently trained translator ablation (3 seeds, 800 steps)

### 10.1 Motivation: inference-time intervention (Stage 1)

Before training new architectures, we switched translators off (`T_k = identity`) in the three trained DSLT-A checkpoints from Stage 0, without retraining.

- **All four removed** (49% of parameters, 39% of MACs): PSNR changes by −0.098 / −0.078 / −0.096 dB. Only the severe drop is Holm-significant and robust across seeds.
- **Where the effect sits:** almost entirely in the e2 translator (0.50M parameters). Removing e3, e4 or the bottleneck translator, 21.4M parameters together, changes PSNR by ≤ 0.01 dB.

Because these interventions act on networks trained *with* translators, they do not settle whether the architecture needs them. That requires independent training.

### 10.2 Design

- Three architectures (Tfull, Te2, Tnone; §5), ungated, with no gate modules built.
- Seeds 42, 123 and 2026; the 800-step protocol; one test evaluation each.
- The hypotheses, margins and decision rules were pre-registered and committed before training (`results/v8_translator_training/PREREGISTRATION.md`).
- Before training, forward hooks confirmed that only the intended translators execute.

**Table 3.** Test results, mean over three seeds (PSNR ± SD / SSIM / LPIPS).

| Model | Params | Mild | Medium | Severe |
|---|---:|---|---|---|
| Tfull | 44.41M | 26.75 ± 0.20 / 0.8023 / 0.2207 | 24.17 ± 0.23 / 0.7461 / 0.3317 | 18.81 ± 0.30 / 0.5966 / 0.5755 |
| Te2 | 23.02M | 26.74 ± 0.12 / 0.8023 / 0.2216 | 24.08 ± 0.13 / 0.7446 / 0.3352 | 18.74 ± 0.12 / 0.5954 / 0.5798 |
| Tnone | 22.52M | 26.81 ± 0.17 / 0.8030 / 0.2189 | 24.25 ± 0.22 / 0.7466 / 0.3293 | 18.94 ± 0.36 / 0.5974 / 0.5747 |

**Table 4.** Tnone − Tfull: Δ, 90% two-level bootstrap CI and pre-registered verdict (Holm over 18 tests).

| Metric | Mild | Medium | Severe |
|---|---|---|---|
| PSNR | +0.057 [−0.053, +0.156], non-inferior | +0.075 [−0.008, +0.153], non-inferior | +0.137 [+0.045, +0.241], non-inferior and better (95% CI [+0.033, +0.266]; Holm p = 8e-08; all 3 seeds positive) |
| SSIM | +0.0006, equivalent | +0.0006, equivalent | +0.0008 [−0.0006, +0.0023], non-inferior |
| LPIPS | −0.0018, equivalent | −0.0024 [−0.0052, +0.0003], non-inferior | −0.0008, equivalent |

**Findings.**
- At 800 steps the encoder–decoder without translators is non-inferior to the full model on every severity and metric, and better on severe PSNR.
- The e2-only model is inconclusive against Tfull on most comparisons: its per-seed differences change sign (for example, severe PSNR +0.146, −0.286 and −0.058 dB).

**Stage 2 conclusion (budget-specific):** under an 800-step, 128-px budget, translators are not needed for restoration quality. The next section shows that this conclusion depends on the budget.

## 11. Stage 3: convergence pilot (one seed, 20,000 steps)

**Design.**
- Tfull vs. Tnone, seed 42 only, 20,000 steps at 256 × 256. This is 25× the Stage 2 step count, at 4× the pixels per crop.
- Validation on all 250 validation images at each severity after every 1,000 steps. The best-validation checkpoint is evaluated once on the test set.
- All other settings are identical between the two models.

**Verification.**
- Both models completed all 20,000 steps without interruption.
- Checkpoint checksums match their manifests, and all weights are finite.
- 750/750 test inputs passed the alignment check.
- For both models the best-validation epoch was the last one (20).

> **This stage uses a single training seed.** All statistics below are image-level. They are conditional on one training run per architecture and do not establish that the differences reproduce across seeds. Stage 2 showed seed-to-seed standard deviations of up to 0.36 dB PSNR.

**Table 5.** Test results (250 images; PSNR / SSIM / LPIPS).

| Model | Params | Mild | Medium | Severe |
|---|---:|---|---|---|
| Tfull | 44.41M | 29.74 / 0.8771 / 0.1407 | 27.51 / 0.8254 / 0.2271 | 22.82 / 0.6897 / 0.4264 |
| Tnone | 22.52M | 29.79 / 0.8789 / 0.1374 | 27.43 / 0.8255 / 0.2265 | 22.78 / 0.6861 / 0.4352 |

Both models gain 3–4 dB over Stage 2. For example, Tfull on severe damage goes from 18.81 dB at 800 steps (three-seed mean) to 22.82 dB.

**Table 6.** Tnone − Tfull, image-level (n = 250, one seed; Holm over 9 tests).

| Severity | Metric | Δ | 90% CI | d_z | Images where Tnone is better | p (Holm) | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| Mild | PSNR | +0.050 | [−0.047, +0.150] | +0.05 | 55% | 1 | non-inferior |
| Mild | SSIM | +0.0018 | [+0.0006, +0.0031] | +0.16 | 62% | 0.002 | non-inferior (Tnone better) |
| Mild | LPIPS | −0.0033 | [−0.0044, −0.0022] | −0.31 | 69% | 1e-07 | equivalent (Tnone slightly better) |
| Medium | PSNR | −0.080 | [−0.175, +0.016] | −0.09 | 47% | 0.79 | inconclusive |
| Medium | SSIM | +0.0001 | [−0.0011, +0.0013] | +0.01 | 47% | 1 | equivalent |
| Medium | LPIPS | −0.0007 | [−0.0019, +0.0006] | −0.05 | 55% | 1 | equivalent |
| Severe | PSNR | −0.039 | [−0.131, +0.048] | −0.05 | 47% | 1 | inconclusive |
| Severe | SSIM | −0.0036 | [−0.0052, −0.0021] | −0.25 | 35% | 2e-06 | worse beyond margin (narrowly) |
| Severe | LPIPS | +0.0088 | [+0.0070, +0.0105] | +0.52 | 31% | 4e-13 | worse beyond margin |

On severe LPIPS, Tfull is better on 69% of test images.

**Learning curves** (`results/v9_convergence_pilot/learning_curves.png`).

Training loss is almost identical for the two models throughout. Validation curves differ by severity:

- **Mild.** Tnone is ahead from about step 5k, by roughly 0.16–0.22 dB, and ends 0.175 dB ahead.
- **Medium.** The curves are indistinguishable.
- **Severe.** The validation PSNR difference, Tfull − Tnone, at each 1,000-step mark:

  | Step | 1k | 2k | 3k–4k | 5k–8k | 9k–13k | 14k | 16k | 20k |
  |---|---:|---:|---:|---|---|---:|---:|---:|
  | Δ (dB) | +0.167 | +0.103 | ≈ 0 | −0.047 to −0.101 | +0.005 to +0.011 | +0.020 | +0.056 | +0.075 |

  - Tnone leads between steps 5k and 8k, by up to 0.101 dB.
  - From step 9k onward, Tfull is ahead at every validation point, so this is a nominal lead. Until step 13k, however, the lead is at most 0.011 dB, an order of magnitude below the 0.10 dB PSNR margin and well within plausible run-to-run variation. The curves are effectively tied over this interval.
  - The lead grows from step 14k (+0.020 dB) to step 20k (+0.075 dB) and was still widening when training ended. Even the final value is below the 0.10 dB practical-relevance margin.
  - We therefore do **not** identify a precise crossover point. The data support only this statement: in this run, a nominal Tfull lead begins at about step 9k and becomes larger, though still sub-margin, over the last ~6k steps.
  - Severe validation SSIM shows the same ordering at the end (0.6831 vs. 0.6801).
  - Validation and test agree on the direction of the severe-damage result.
  - Validation LPIPS was not logged during training, so no learning curve is available for the metric on which the test-set difference is largest.
- **Convergence.** For both models the best epoch is the last one, and validation PSNR still rose by about 0.015 dB per epoch at the end of a cosine schedule. Neither model shows overfitting, and both are slightly under-converged.

**Stage 3 findings (exploratory).**
1. **The Stage 2 advantage does not persist.** Tnone's +0.137 dB severe-PSNR advantage at 800 steps becomes −0.039 dB, 95% CI [−0.148, +0.065], at 20,000 steps.
2. **Equivalence is only partly established.** It holds or is exceeded on mild damage and holds for medium SSIM and LPIPS. Medium PSNR is inconclusive.
3. **Severe-damage perceptual advantage for translators (this run).** On severe damage, the full model has better test LPIPS (by 0.0088, beyond the 0.005 margin) and slightly better SSIM. This is the first result in this study that favours the translators. On validation PSNR, the corresponding lead is negligible before about step 13k and grows over the final ~6k steps.
4. **Single-seed caveat.** The severe LPIPS gap is about 1.4 times the seed-to-seed standard deviation of LPIPS observed for DSLT-A at 800 steps (0.0063). A single training run cannot separate a real architectural effect from seed variation.

## 12. Computational efficiency analysis

**Table 7.** Cost of Tfull vs. Tnone. Inference figures are from Stage 3 profiling; training figures are from the Stage 3 training manifests (one run each).

| Measure | Tfull | Tnone | Tnone vs. Tfull |
|---|---:|---:|---:|
| Inference parameters | 44.41M | 22.52M | −49% |
| GMACs @ 128² | 19.1 | 11.6 | −39% |
| GMACs @ 256² | 76.3 | 46.4 | −39% |
| Latency @ 256², batch 1 | 57.2 ms | 37.7 ms | −34% |
| Throughput @ 256², batch 4 | 24.8 img/s | 38.9 img/s | +57% |
| Training time per step (256², batch 4) | 0.841 s | 0.615 s | −27% |
| Training wall-clock, 20k steps + 20 validation passes | 296 min | 215 min | −27% |
| Peak training memory (MPS driver) | 7.19 GB | 5.77 GB | −20% |

**Cost of the other variants** (Stage 0 profiling):

| Variant | Params | GMACs @ 256² | Latency @ 256² | Throughput @ 256², batch 4 |
|---|---:|---:|---:|---:|
| Gated DSLT-B/C | 46.06M | 82.5 | 68.1 ms | 20.9 img/s |
| Oracle DSLT-O | 46.06M | 83.5 | 73.1 ms | 19.9 img/s |

The e2-only variant (Te2) has 23.02M parameters and 54.5 GMACs at 256² (Stage 2 profiling).

**Where the compute goes.** The translators account for 21.89M parameters and 29.9 GMACs at 256². By parameters the bottleneck translator is the largest (10.69M); by compute the e3 translator is the most expensive (10.9 GMACs).

**Training time** includes the damage estimator, which is trained in all configurations.

**Measurement caveats.**
- MACs exclude element-wise operations.
- Latencies were measured on one Apple M4 Pro GPU and vary by about ±1 ms between profiling sessions (57.2–58.5 ms for the same architecture).
- Inference peak memory could not be separated per model on MPS and is not reported.

## 13. Discussion

**Training budget changes the conclusion.**
- At 800 steps, the smaller encoder–decoder learns faster and comes out ahead on severe PSNR. The Stage 3 learning curves show the same early pattern.
- With 25× more training (one seed), the full model catches up on medium and severe damage. On severe damage it ends with better perceptual metrics in this run.
- Short-budget component ablations, our own Stage 2 included, can therefore confuse faster optimisation with better final capability. We recommend that component claims in restoration papers be checked at more than one training budget.

**Where translators might help.**
- The Stage 3 advantage of Tfull is confined to the most damaged inputs and appears mainly in LPIPS and SSIM rather than PSNR.
- One plausible explanation, not tested here, is that heavy degradation needs more extensive feature transformation than the encoder–decoder's skip pathways provide. Extra residual capacity in latent space would then be used only once the easier corrections have been learned.
- The Stage 1 analysis hints at a possible mechanism, but it was measured at 800 steps and may not transfer: almost all translator effect was concentrated in the cheapest, shallowest translator (e2).
- Translator-level attribution at the longer budget has not been measured.

**Is the benefit worth the cost?**
- If the Stage 3 severe-damage advantage reproduces across seeds, the choice is a real trade-off:
  - about 0.009 LPIPS and 0.004 SSIM on severely damaged inputs;
  - against twice the parameters, 1.6× the MACs, 1.5× the latency, and 37% more training time per step.
- For mild and moderate damage, the encoder–decoder is the better choice on current evidence.
- The answer therefore depends on the application's damage distribution and its compute budget.

**Damage conditioning.** Stage 0 shows that, at least at 800 steps, explicit severity information adds little to this backbone. Even an oracle gives < 0.1 dB. Gates trained without supervision collapse, and supervised gates are informative but ignored. This agrees with the observation that well-trained encoder–decoders infer degradation implicitly [TODO: reference needed, see ref. 27]. Our oracle tested only the existing FiLM-gate pathway; conditioning inside the translator blocks was not tested.

**Statistical lessons.**
- Many differences in this study are significant at the image level with Holm correction (p < 10⁻⁶) yet smaller than the seed-to-seed standard deviation.
- Image-level significance therefore says little about whether a result would reproduce with a new training run.
- Pre-registered equivalence margins let us make, and also withhold, claims of "no difference" in a disciplined way.

## 14. Limitations

1. **Synthetic, in-distribution degradation on one dataset.**
   - All degradations are synthetic, produced by the same pipeline for training and test, on a single collection of 256×256 images that are predominantly faces.
   - Results may not transfer to real damaged photographs, other image domains, other resolutions, or degradation types outside the pipeline.
   - No real-photo evaluation with ground truth was performed.
2. **Stage 3 has a single seed.** The severe-damage advantage of the translators and every other Stage 3 difference are exploratory. A three-seed replication was begun and stopped before any checkpoint was saved; it contributes no data.
3. **Training budgets.** Stages 0–2 use 800 steps at 128 px, far from convergence. Stage 3 uses 20,000 steps, and both models were still improving slightly at the end. Gating (Stage 0) and the e2-only variant were not re-tested at the longer budget.
4. **No fair external baselines.** No comparison with established restoration methods under this protocol is reported, so no state-of-the-art claim is possible.
5. **Margins.** Equivalence margins were chosen by the authors, pre-registered but not externally validated.
6. **Validation set size.** Stages 0–2 selected checkpoints using only 24 validation images per severity.
7. **Hardware-specific timing.** Efficiency measurements come from a single Apple M4 Pro GPU (MPS backend) in FP32; relative costs may differ on CUDA GPUs or other accelerators.
8. **Dataset artefacts.** Some images are stored rotated or flipped, and the original dataset contained duplicates (handled by the split procedure).
9. **Legacy results excluded.** Earlier checkpoints trained on all images, including validation and test images, are contaminated and are not used anywhere in this paper.

## 15. Conclusion

We ran a controlled, statistically careful study of two specialised components in a multi-scale old-photo restoration network.

- **Damage-conditioned gating** gives no practically meaningful benefit under the tested budget. Even oracle severity information adds less than 0.1 dB.
- **Latent feature translators**, in independently trained models at a short budget (three seeds, 800 steps), give no measurable benefit: a plain encoder–decoder with half the parameters is non-inferior or better.
- **After 25× longer training** (one seed), the picture changes:
  - the encoder–decoder's early advantage disappears;
  - the encoder–decoder is equivalent or non-inferior on mild damage and on medium SSIM and LPIPS, while medium and severe PSNR are inconclusive;
  - in this single run, the translator model achieves better perceptual quality on severely damaged images.

The encoder–decoder remains substantially cheaper in parameters, compute, latency, training time and memory. Whether the severe-damage benefit of latent translation is reproducible, and therefore worth its cost, needs a multi-seed study at the longer budget. That is the immediate next step. More broadly, our results suggest that conclusions about restoration components can change with training budget and should be tested at more than one budget, across seeds, and against explicit cost.

## Acknowledgements

[TODO]

## Data and code availability

[TODO: repository URL, dataset source and licence, and checkpoint availability. Checkpoints are 169–180 MB and are not in the git repository.]

---

## References

*Status after the 24 November 2026 check (details in `MANUSCRIPT_REVIEW.md` §5):*
- ✔ means the bibliographic details were confirmed against a publisher, proceedings, CVF, PubMed Central, Project Euclid or JSTOR record found during the check.
- A DOI is given only where a source showed it.
- "DOI not confirmed" means no DOI was located. **Do not add one without checking it on IEEE Xplore or Crossref.**
- Page numbers come from the proceedings pages listed.

1. ✔ Z. Wan, B. Zhang, D. Chen, P. Zhang, D. Chen, J. Liao, F. Wen. "Bringing Old Photos Back to Life." *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 2747–2757. IEEE DOI not confirmed; arXiv:2004.09484.
2. ✔ (partial) Z. Wan, B. Zhang, D. Chen, P. Zhang, F. Wen, J. Liao. "Old Photo Restoration via Deep Latent Space Translation." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 2, pp. 2071–2087, 2023 (early access 2022). DOI: 10.1109/TPAMI.2022.3163183. **[VERIFY: author list and order. A library record shows 6 authors; the CVPR version has 7.]**
3. ✔ O. Ronneberger, P. Fischer, T. Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015*, LNCS vol. 9351, pp. 234–241, Springer, 2015. DOI: 10.1007/978-3-319-24574-4_28.
4. ✔ X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, C. C. Loy. "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks." *ECCV Workshops 2018*, LNCS vol. 11133, pp. 63–79, 2019. DOI: 10.1007/978-3-030-11021-5_5. **[VERIFY: the arXiv version also lists X. Tang; follow the published version.]**
5. ✔ S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. "CBAM: Convolutional Block Attention Module." *ECCV 2018*, LNCS vol. 11211, pp. 3–19. DOI: 10.1007/978-3-030-01234-2_1.
6. ✔ W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, Z. Wang. "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." *CVPR 2016*, pp. 1874–1883. IEEE DOI not confirmed.
7. ✔ F. Yu, V. Koltun. "Multi-Scale Context Aggregation by Dilated Convolutions." *International Conference on Learning Representations (ICLR)*, 2016. arXiv:1511.07122. (No DOI; ICLR does not issue them.)
8. ✔ E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville. "FiLM: Visual Reasoning with a General Conditioning Layer." *Proc. AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018. DOI: 10.1609/aaai.v32i1.11671.
9. ✔ X. Wang, K. Yu, C. Dong, C. C. Loy. "Recovering Realistic Texture in Image Super-Resolution by Deep Spatial Feature Transform." *CVPR 2018*, pp. 606–615. IEEE DOI not confirmed.
10. ✔ S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang. "Restormer: Efficient Transformer for High-Resolution Image Restoration." *CVPR 2022*, pp. 5728–5739. IEEE DOI not confirmed.
11. ✔ V. Potlapalli, S. W. Zamir, S. Khan, F. S. Khan. "PromptIR: Prompting for All-in-One Image Restoration." *Advances in Neural Information Processing Systems (NeurIPS)* 36, 2023. **Corrected title:** the NeurIPS version omits "Blind", which appears only in the arXiv title (arXiv:2306.13090). Pages and DOI not confirmed.
12. ✔ L. Chen, X. Chu, X. Zhang, J. Sun. "Simple Baselines for Image Restoration." *ECCV 2022*, LNCS vol. 13667. DOI: 10.1007/978-3-031-20071-7_2. **[VERIFY: pages.]**
13. ✔ B. Li, X. Liu, P. Hu, Z. Wu, J. Lv, X. Peng. "All-in-One Image Restoration for Unknown Corruption." *CVPR 2022*, pp. 17452–17462 per CVF (DBLP gives 17431–17441). **[VERIFY: pages, and DOI 10.1109/CVPR52688.2022.01693, which comes from a secondary index (mlanthology); confirm on IEEE Xplore.]**
14. ✔ R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." *CVPR 2018*, pp. 586–595. **[VERIFY: DOI 10.1109/CVPR.2018.00068 was reported by a search summary; confirm on IEEE Xplore.]**
15. ✔ Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. "Image Quality Assessment: From Error Visibility to Structural Similarity." *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004. DOI: 10.1109/TIP.2003.819861.
16. ✔ J. Johnson, A. Alahi, L. Fei-Fei. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution." *ECCV 2016*. DOI: 10.1007/978-3-319-46475-6_43. **[VERIFY: LNCS volume and pages.]**
17. ✔ K. Simonyan, A. Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*, 2015. arXiv:1409.1556.
18. ✔ P. Charbonnier, L. Blanc-Féraud, G. Aubert, M. Barlaud. "Two Deterministic Half-Quadratic Regularization Algorithms for Computed Imaging." *Proc. IEEE International Conference on Image Processing (ICIP)*, vol. 2, pp. 168–172, 1994. DOI not confirmed.
19. ✔ I. Loshchilov, F. Hutter. "Decoupled Weight Decay Regularization." *ICLR*, 2019. arXiv:1711.05101.
20. ✔ I. Loshchilov, F. Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*, 2017. arXiv:1608.03983.
21. ✔ D. J. Schuirmann. "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, vol. 15, pp. 657–680, 1987. DOI: 10.1007/BF01068419.
22. ✔ D. Lakens. "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, vol. 8, no. 4, pp. 355–362, 2017. DOI: 10.1177/1948550617697177.
23. ✔ S. Holm. "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979. JSTOR 4615733. **[VERIFY: DOI 10.2307/4615733 follows the JSTOR pattern but was not shown by a primary source.]**
24. ✔ F. Wilcoxon. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945. DOI not confirmed.
25. ✔ B. Efron. "Bootstrap Methods: Another Look at the Jackknife." *The Annals of Statistics*, vol. 7, no. 1, pp. 1–26, 1979. DOI: 10.1214/aos/1176344552.
26. [TODO] Dataset source citation.
27. [TODO] References still needed: seed variance and reproducibility in deep learning; implicit degradation estimation (cited in §13); further old-photo restoration and colourisation work (§3). A candidate found during the check, still to be read and verified: H. Touvron et al., "Powers of layers for image-to-image translation", arXiv:2008.05763 (residual blocks applied in a latent space).
