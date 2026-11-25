# RealOld (Pik-Fix) Inference-Only Evaluation: Licence Check and Test Protocol

**Status:** plan only. Nothing downloaded, no dataset access yet, no code run, no commits. Research done 25 Nov 2026 from primary sources only.
**Anything not confirmed by a primary source is marked UNCONFIRMED.**

## 1. Recommendation (read this first)

**NO-GO today. Conditional GO after written permission.**

- There is **no dataset licence**. The only stated terms are: access "only upon personal inquiry" by email, "due to copyright restrictions" (section 2).
- The MIT licence in the repository covers the code. It is not described as covering the data.
- The paper says the dataset is "publicly available", but the data is not downloadable. The repository gives no link, and users report emails going unanswered (section 3).
- Becomes **GO** if the authors reply with the data link and the reply (or an explicit follow-up) permits use for a published paper. Keep that email in the repository as the permission record.
- Even then, the scope is **metrics only**. Do not put RealOld images (inputs, GT or our outputs) in the paper, repository or supplementary material without separate explicit permission. The images are scans of old photographs, and the authors cite copyright as the reason for restricted access.
- If no reply arrives within about 2 weeks (the README promises a reply in 3 days), treat RealOld as unavailable. Do not use mirrors or third-party copies. Report this honestly as a limitation, and keep the existing qualitative real-photo panel (`REAL_PHOTO_PANEL_PLAN.md`) as the only real-photo evidence.

## 2. Licence / terms of use

**No licence is stated for the RealOld data.** There is no dataset licence file, no dataset card, no terms-of-use page, and no licence sentence in the paper or the supplement.

Verbatim from the official README (https://github.com/DerrickXuNu/Pik-Fix, section "Real Old Photo Data / Download"):

> "Due to copyright restrictions, our real old photo dataset is only available upon personal inquiry. To request access, please email derrickxu1994@gmail.com with the subject "Pik-Fix Data Inquiry" and include your name and affiliation in the message.
> Then we will reply to you with the download link in 3 days."

Verbatim from the paper (CVF open access, Section 4.1; https://openaccess.thecvf.com/content/WACV2023/papers/Xu_Pik-Fix_Restoring_and_Colorizing_Old_Photos_WACV_2023_paper.pdf):

> "We are making this dataset publicly available to allow other researchers to develop advanced algorithms that can both colorize and repair old photos impaired by scratches, blur, cracks, wear, film grain noise, and physical and capture distortions."

The supplement (https://openaccess.thecvf.com/content/WACV2023/supplemental/Xu_Pik-Fix_Restoring_and_WACV_2023_supplemental.pdf) repeats "We are making this dataset publicly available to allow other researchers to develop advanced algorithms ...". It adds no terms.

Findings on the terms:

- **Repository licence:** MIT, "Copyright (c) 2022 Runsheng Xu" (https://github.com/DerrickXuNu/Pik-Fix/blob/main/LICENSE). It is the code licence. The README does not apply it to the data, and the data is described as restricted by copyright, so do not assume the MIT terms cover it.
- **Requirements that are stated:** access by personal email request, with name and affiliation. The README does not mention a citation or acknowledgement requirement, research-only use, a no-redistribution clause, or whether results may be published. All of these are UNCONFIRMED. The paper is the natural thing to cite regardless.
- **Inconsistency:** the paper says "publicly available"; the README says "only available upon personal inquiry... due to copyright restrictions". The README is the operative statement.
- **Provenance:** the paper does not say where the old photographs came from or who holds the rights (UNCONFIRMED). The GT was made by "Photoshop experts" (paper), and the terms covering that work are UNCONFIRMED.
- **Not checked:** the CVF HTML landing page returned HTTP 403 to the fetch tool. The paper and supplement PDFs on the same site were read in full, and the GitHub README, LICENSE and file tree were read directly. No project page or request form beyond the email address was found.

## 3. Access method and its risk

1. Email derrickxu1994@gmail.com, subject exactly `Pik-Fix Data Inquiry`, giving name and affiliation (the user supplies these; nothing is stored in this plan). Suggested extra sentences, to settle the open terms: say the use is inference-only evaluation of two trained restoration models for a research paper; that only aggregate and per-image metric values would be published, not images; and ask (a) whether this is permitted, (b) how to cite, (c) whether the data may be stored on a university or personal machine only.
2. The README says the reply (a download link) arrives "in 3 days". It does not say what the link points to (a shared drive; UNCONFIRMED).
3. **Access risk, observed:** GitHub issue #2 ("Need Pik-Fix Data", https://github.com/DerrickXuNu/Pik-Fix/issues/2). A user wrote of waiting for an answer; one commenter (22 Oct 2024) asked "Have you received Pik-Fix data? I have the same question."; another (1 Nov 2026) wrote "I sent an email, and a follow up email, but didn't receive RealOld Dataset". The issue has no maintainer reply. Issue #3 reports the `texture2` folder (dust textures) missing. So access may not succeed.
4. The maintainer wrote in issue #1 (30 Jan 2023) that the pre-trained models might not be released "as it is related to commercialization" (a statement about models, not data, but it signals cautious terms).

## 4. Dataset facts (as stated by the primary sources)

| Fact | Value | Source |
|---|---|---|
| Content | "digitized copies of 200 real old black & white photographs" | paper 4.1; supp 1 |
| Pairs | Each photo is "paired with digitally manually restored and colorized by Photoshop experts" (paper intro: "paired with a 'pristine' version of it that was manually restored and colorized by Adobe Photoshop editors") | paper; supp |
| Aligned | "aligned 'ground truth' 'pristine' photos to enable pixel-to-pixel processing and comparison using full-reference quality metrics" | supp 1 |
| Input type | **Greyscale** ("200 authentic old grayscale photos") | paper intro |
| GT type | **Restored AND colourised** (not restoration only) | paper, supp |
| Degradations | "scratches, blur, cracks, wear, film grain noise, and physical and capture distortions" | paper, supp |
| Release layout | `real_old_data` ("200 image pair with old photos and the repared ones"), `real_old_ref` (reference images used in the paper, for Pik-Fix's reference-based colourisation), `texture2` (texture files for generating fake old photos) | README |
| Split | Test only: "RealOld is used for testing models that have been trained on Pascal". No train/val split. | paper 4.1 |
| Resolutions | **UNCONFIRMED.** Not stated in the paper, supplement or README. Images are probably not uniform in size, and our protocol must not assume otherwise. | n/a |
| File format | **UNCONFIRMED.** The repo's `RealOldPhotoDataset` class suggests GT files whose name contains `t` (and not `m`) with extension of 3 characters (`.png` / `.jpg`), and an input file with the same name but `o` in place of the last `t` (`gt_name[:-5] + 'o.' + ext`). The input is read with `cv2.imread(path, 0)` (forced greyscale). This is inferred from code, not documented. | `datasets/OldPhotoDataset.py` |
| Other | `hypes_yaml/config.yaml` points `real_file` at `data/real_old_resize/`, which suggests the authors used a resized copy for evaluation (size unknown). The README says the shared folder is `real_old_data`. UNCONFIRMED whether `real_old_data` is the original or the resized version. | repo |

## 5. Official evaluation protocol (what Pik-Fix did)

- Models trained on Pascal (10,000 train / 1,000 test), then tested on RealOld. RealOld is not used for training (paper 4.1).
- Metrics: "PSNR and SSIM [49] scores between the reference and restored/colorized pictures. As an alternative, ... LPIPS". Table 2 of the paper: Pix2pix 16.80 / 0.684 / 0.320; DeOldify 17.14 / 0.723 / 0.287; He et al. 16.72 / 0.707 / 0.314; InstColorization 16.86 / 0.715 / 0.312; Wan et al. 16.99 / 0.709 / 0.303; **Pik-Fix 17.20 / 0.758 / 0.258** (PSNR / SSIM / LPIPS).
- Not stated anywhere: the test resolution, resize or crop rule, colour space for PSNR, LPIPS backbone, PSNR cap, whether all 200 images were used. The training patches were random 256x256 crops. The user study used 100 randomly selected RealOld photos (not the metrics). The repository contains no test or evaluation script (issue #1 asked for one). So the 200-image count for the metrics is **UNCONFIRMED**, though the paper reports Table 2 as the "RealOld dataset" without a subset.
- Pik-Fix's reference-based colourisation uses 2,000 RGB portraits downloaded from Google Images as reference candidates. Our models take no reference.
- **Consequence:** Pik-Fix's Table 2 numbers are not reproducible for us and not comparable to ours (different task, unknown resolution). Do not put them in the same table as our results. At most, cite them in text as context for a colourising model.

## 6. What affects comparability for us

1. **Colourisation dominates RGB metrics.** The GT is colourised from greyscale. Our models are trained to restore synthetic degradation and are not designed to colourise. 40% of training inputs get `convert_to_grayscale` (mean of channels, replicated to 3) with a colour clean target (`utils/degradation.py`), so the models may partly learn average colour. Whether they do on real scans is unknown. Therefore a full-RGB PSNR/SSIM/LPIPS against colour GT mixes two things: restoration quality (luminance) and chroma guessing. The GT colours are also an arbitrary choice by an editor, not the single truth.
2. **Handling (decided here, before any data is seen):**
   - **Primary: luminance-only comparison.** Convert both model output and GT to luma Y (BT.601: 0.299 R + 0.587 G + 0.114 B, in float [0,1]), then replicate Y to 3 channels and score with the **unchanged** `psnr_np`, `ssim_np` and `lpips_batch`. Replicating the channel gives the same PSNR and SSIM as a single-channel computation (SSIM is the mean over identical channels), and LPIPS gets the 3-channel input it needs. This keeps one implementation for every number.
   - **Secondary: full RGB vs colour GT**, exactly as `utils/benchmark.py`. Label it "restoration + colourisation, our models do not colourise by design". It is exploratory and reported separately (never pooled with the primary).
   - **Reference rows for scale (same protocol):** the unprocessed input (`identity_fn`: greyscale input replicated to RGB) and, optionally, bicubic (`bicubic_fn`). The identity row is the floor that shows how much of the RGB score is simply "grey vs colour".
   - **Optional chroma-free diagnostic:** report the mean absolute chroma of each model output (e.g. mean |a|,|b| in CIELAB) next to the GT's, so readers can see if the models colourise at all. This is descriptive only.
3. **Domain shift:** real scans of the 1900s to 1960s versus our synthetic degradation of colour face-centric images. Models were never trained on real degradation. State this as a limitation. No training, fine-tuning or hyper-parameter tuning on RealOld.
4. **Greyscale input handling:** inputs are 1-channel (UNCONFIRMED whether stored as 1- or 3-channel files). Load as uint8 greyscale, replicate to 3 identical channels. Do not apply any colour transform before the model.
5. **Aligned GT may still differ slightly** (editor crops, brightness and contrast edits). The paper calls it aligned but does not give a registration tolerance (UNCONFIRMED). Check at the start that every pair has identical H x W; if any pair differs, **do not resize to fit**. Log it, exclude the pair, and report the exclusion count.
6. **Unknown licence also limits what we can show**, not just what we can run (section 1).

## 7. Proposed test protocol (pre-registered; fix all choices before looking at any output)

**7.1 Data and storage**
- Place the downloaded folder under `data/realold/` (git-ignored; add `data/realold/` to `.gitignore` before downloading). Never commit it, never copy images into `paper/`, `results/` or any bundle.
- Use `real_old_data` only. `real_old_ref`, `texture2` are not needed.
- Pair discovery: pair GT and input by the file naming found in the folder (inferred rule `...t.ext` / `...o.ext`; UNCONFIRMED, verify on the real file list). Assert that there are exactly 200 pairs, that each pair has the same height and width, and that each file decodes. Any deviation is reported, not silently fixed.
- **Images used: all pairs found (expected 200).** No selection, no filtering by difficulty, no removal after seeing scores.
- Record in `results/v11_realold/data_manifest.json`: file names, sizes, SHA-256 of every file, original H x W, channel count, extension. No pixels.

**7.2 Preprocessing (identical for input and GT, deterministic, no TTA)**
1. Load with OpenCV, convert BGR to RGB (`cv2.imread`, then `cv2.cvtColor`), keep uint8. Input: 1 channel replicated to 3 (or the 3-channel file as given; log which).
2. **Primary scale: long side 512.** If the long side > 512, resize both images with `cv2.resize(..., interpolation=cv2.INTER_AREA)` so that the long side is exactly 512 and the aspect ratio is kept (short side = round(S x 512 / L)). If the long side <= 512, leave the image unchanged (never upscale). Same scale factor for input and GT, so they stay aligned. This matches the planned real-photo panel.
3. Convert to float32 /255 in [0,1]. Do not quantise model outputs (same as `utils/benchmark.py`: clip only).
4. Reflect-pad **bottom and right** to a multiple of 16 with `pad_to_multiple(restore_fn, 16, "reflect")`, crop back to H x W after the forward pass.
5. Batch size 1 (images differ in size). One forward pass, no flips or rotations, no tiling (if memory forces tiling, use the same tiling for every model and say so).
6. **Sensitivity scale: long side 256** (the training scale; the same code with 256 in place of 512). Reported as a secondary analysis only. **No centre-crop**, since it discards image content (scratches near the borders) and the crop choice would be arbitrary.

**7.3 Models (validation-selected checkpoints, already trained; no new training)**
- `results/v9_convergence_pilot/seed42/Tfull_best.pth`, `.../seed42/Tnone_best.pth`
- `results/v9_convergence_pilot/seed123/Tfull_best.pth`, `.../seed123/Tnone_best.pth`
- Load with `experiments.stage2.load(cfg, seed, device)` after setting `s2.OUT` to the v9 directory (as `experiments/stage3.py` does); the forward pass is `x + dec(...)` clamped to [0,1]. Before inference, check each checkpoint's SHA-256 against its `*_manifest.json` (same integrity check as `stage3.py`).
- Tfull = DSLT-A full (44.4M parameters); Tnone = encoder-decoder without translators (22.5M).
- Seed 2026 and the baselines (v10) are out of scope for this evaluation unless asked.
- Hardware: one MPS job at a time with the existing watermark settings (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`, `LOW=0.4`). Estimated cost: 200 images x 4 checkpoints at <= 512 px, a few minutes. Wait until no other MPS job is running.

**7.4 Metrics (exactly as `utils/benchmark.py`; one implementation)**
- PSNR: `psnr_np(out, ref)`: output clipped to [0,1], float64 MSE, 10 log10(1/MSE), capped at 50 dB per image.
- SSIM: `ssim_np(out, ref)`: `skimage.metrics.structural_similarity(ref, out, data_range=1.0, channel_axis=2)`, default 7x7 window.
- LPIPS: `lpips_batch(out, ref, net="alex")`: AlexNet, `lpips` package, inputs scaled to [-1,1], lower is better.
- Computed on the full (resized, unpadded) image. Mean over images, plus per-image values saved.
- **Primary family:** luma-replicated variant (section 6.2), 3 metrics. **Secondary:** full-RGB variant and the 256 sensitivity scale, labelled exploratory.
- Reference rows: input identity (and optionally bicubic) scored the same way.

**7.5 Statistics**
- **Paired, per seed, no cross-seed pooling.** For each seed in {42, 123}, per-image difference d_i = metric(Tnone) - metric(Tfull) over the same 200 images (assert identical image order, as `stage3.py` does).
- Report for each seed and metric: the mean of each model, the mean difference, a 95% image-level bootstrap CI (10,000 resamples, fixed seed 0), the two-sided Wilcoxon signed-rank p-value, and the paired effect size.
- **Multiplicity:** Holm correction within the primary family (3 metrics x 2 seeds = 6 tests). The secondary RGB and 256-px results get their own labelled family and are described as exploratory.
- Inference is **image-level and conditional on each training run** (one network per seed). Do not claim anything about the population of training runs. If the two seeds disagree in sign, report that, do not average them away. If the existing equivalence margins (PSNR 0.10 dB, SSIM 0.0020, LPIPS 0.0050 from Stage 2/3) are reused, say so, and apply them to the luma-primary family only.
- The 200 images are scans, not independent draws from a defined population; the images may include near-duplicate content (UNCONFIRMED). Say "image-level" and avoid generalising beyond this set.

**7.6 Output paths (all under `results/v11_realold/`)**
- `data_manifest.json` (file list and hashes, no pixels), `protocol.json` (this protocol's constants, git commit, library versions, device)
- `per_image/{Tfull,Tnone}_seed{42,123}.json` (per-image PSNR / SSIM / LPIPS for each variant and scale, image names) and matching `*_eval_manifest.json` (checkpoint SHA-256 and environment)
- `reference_rows.json` (identity, optional bicubic)
- `stats.json` and `realold_tables.md` (paired tests, per seed)
- `chroma_diagnostic.json` (optional)
- **No predicted images saved to a committed location.** If outputs are cached for debugging, keep them outside git (e.g. under a git-ignored `results/v11_realold/_cache/`) and delete after the run.
- The script is `experiments/realold_eval.py` (to be written), reusing the functions in `utils/benchmark.py` without modification. Nothing in the v9 results is altered.

**7.7 What the paper may say**
- Report as "evaluation on RealOld (Xu et al., WACV 2023), 200 real old photographs with colourised expert-restored references, obtained from the authors on request", with the citation, and state that the models are inference-only (no training or tuning on RealOld) and that the primary score is luminance-only because our models do not colourise by design.
- No claim of state of the art, and no comparison with Pik-Fix's Table 2 (different task, unknown resolution).
- Show no RealOld images unless the authors explicitly allow it.

## 8. Open items

1. Send the access email (section 3). Ask for the terms explicitly (publication of metrics, figures, storage, citation).
2. After access: confirm file naming, file format, resolution distribution, channel count of inputs, and whether the folder holds the original or the resized version. Update sections 4 and 7.1 with the confirmed facts and mark the plan as final **before** running inference.
3. Decide whether to keep the 512 primary scale or switch to 256. Fix the choice **before** looking at any score. The plan uses 512, with 256 as sensitivity.
4. Record the permission email (date, sender, content) in `paper/DATASET_PROVENANCE.md` when it arrives.

## Sources

- Paper: https://openaccess.thecvf.com/content/WACV2023/papers/Xu_Pik-Fix_Restoring_and_Colorizing_Old_Photos_WACV_2023_paper.pdf (also arXiv 2205.01902)
- Supplement: https://openaccess.thecvf.com/content/WACV2023/supplemental/Xu_Pik-Fix_Restoring_and_WACV_2023_supplemental.pdf
- Official repository (README, LICENSE, `datasets/OldPhotoDataset.py`, `hypes_yaml/config.yaml`): https://github.com/DerrickXuNu/Pik-Fix
- Issues #1, #2, #3: https://github.com/DerrickXuNu/Pik-Fix/issues
