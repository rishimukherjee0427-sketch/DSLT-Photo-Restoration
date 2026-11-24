# Manuscript Review: Quality-Control Pass 1

*Date: 24 November 2026. Files reviewed: `paper/DSLT_research_paper_draft.md` and `paper/RESULTS_AUDIT.md`, at repository state `5d01d02` (uncommitted).*

**What this pass did not do:** no training, no architecture or configuration changes, no commits, and no changes to README.md or any historical report. The only computation was CPU model construction to compare initial weights, plus read-only `git diff`.

---

## 1. Learning-curve interpretation (Stage 3)

**Evidence.** Per-epoch Tfull − Tnone validation PSNR on severe damage (250 images; `results/v9_convergence_pilot/seed42/*_history.json`):

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k | 16k | 17k | 18k | 19k | 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ (dB) | +.167 | +.103 | +.005 | +.001 | −.086 | −.101 | −.060 | −.047 | +.007 | +.010 | +.011 | +.005 | +.011 | +.020 | +.032 | +.056 | +.060 | +.061 | +.073 | +.075 |

**Corrections made.**
- §11 now separates:
  - a **nominal lead** from about step 9k, which stays ≤ 0.011 dB through step 13k (effectively tied);
  - a lead that grows over the last ~6k steps.
- No precise crossover point is claimed.
- The manuscript notes that even the final lead (0.075 dB) is below the 0.10 dB margin.
- It notes that validation LPIPS was never logged, so there is no learning curve for the metric where the test difference is largest.
- Stage 3 finding 3 and the Discussion were reworded to match.

**Unresolved.** `results/v9_convergence_pilot/REPORT.md` §3 still says the curves "cross at about step 12k". It is a historical report and was not edited. Add an erratum if it will be shared.

## 2. Stage 0 (DSLT-A) vs. Stage 2 (Tfull): score difference

**Question.** Why do the same inference architecture and the same seeds give different results? Severe PSNR is 18.72 vs. 18.81 dB on the 3-seed means, and 18.69 vs. 18.91 dB for seed 2026.

**Checks performed.**

| Check | Result |
|---|---|
| Architecture definitions (`run_ablation.py` CONFIGS, `Nets`, `MultiScaleDSLT`) | A uses `gate_mode='film'`, so it **builds** the gate modulators even with `use_gating=False`. Tfull uses `gate_mode='none'` and builds none. The construction order is damage estimator → encoder → DSLT (translators, then gate modules) → decoder |
| Initialisation (CPU, `seed_all(s)` for s = 42, 123, 2026) | Damage estimator, encoder and all 4 translators are **bit-identical**. The **decoder differs** (max \|Δ\| 0.69–0.70) for every seed. A has 24 extra tensors (87,788 parameters) |
| Parameter counts (checkpoint `n_params`) | A 46,060,826 vs. Tfull 45,973,038 total; the difference is exactly the unused gate modules. Both have 44.41M inference parameters |
| Checkpoint metadata | Identical except config name, `gate_mode`, `translators` and the selected best epoch (seed 42: A epoch 8, Tfull epoch 6) |
| Training configuration and environment | Same crop, steps, EMA, `vgg_normalize`, chip (Apple M4 Pro) and torch 2.15.0 |
| Training code between commits (`git diff e612ad1 / 2cc4a88 → e235240`) | Only new configs, the oracle path and constructor arguments. The ungated training path is unchanged |
| Gradient path | The unused gate modules get no gradient, so AdamW skips them. They cannot affect optimisation directly |

**Conclusion (as stated in the manuscript, §5).**
- *Verified:* the two series start from **different decoder initialisations** because of RNG consumption by the unused modules. Everything else checked is identical. They are therefore effectively independent training runs.
- *Not verified:* that this alone explains the gap. It is consistent with the observed seed-to-seed SD (up to 0.36 dB), but MPS nondeterminism was not measured.
- A definitive test, retraining A with the modules built after the decoder, would require training. It was not done.

## 3. Hardware description

- All **58** run manifests in `results/` record `"chip": "Apple M4 Pro"`, 24 GB (25,769,803,776 bytes), torch 2.15.0 and Python 3.12.10.
- **Manuscript corrected** to say "Apple M4 Pro GPU via PyTorch MPS, 24 GB unified memory" in §6, §12 and §14.
- **Not corrected** (historical, outside scope): `results/v7_stage0/STAGE0_REPORT.md:19`, `results/v7_corrected/REPORT.md:19`, README.md and AUDIT.md, which say "Apple M4".

## 4. Method name

**Collision.**
- "DSLT, Deep Space Latent Translation" is almost the same as Wan et al.'s "Old Photo Restoration via **Deep Latent Space Translation**" (TPAMI 2023, extended from CVPR 2020). That is the best-known method in the *same* application area.
- The approaches also differ in substance. Wan et al. translate between two separately trained VAE latent spaces. Ours inserts residual blocks `z + s·Blocks(z)` on encoder features inside one supervised network.
- The near-identical name would mislead reviewers and cause citation confusion.

**Candidates.** Each was checked with one web search on 24 Nov 2026. This is only a quick check for obvious conflicts, not a full search of Google Scholar, IEEE or arXiv.

| Name | Meaning | Conflict check | Assessment |
|---|---|---|---|
| **RLT-Net** (recommended) | Residual Latent Translation Network | No method by this name found. The nearest concept is Touvron et al., "Powers of layers for image-to-image translation" (residual blocks applied in a latent space, a different task), which is worth citing | Describes the mechanism exactly and distinguishes it from Wan et al. The ablated model becomes "RLT-Net without translators" or simply "the encoder–decoder" |
| MLT-UNet | Multi-scale Latent Translation U-Net | No method by this name found | Accurate, but "Latent Translation" still echoes Wan et al. and "MLT" is a common acronym (multi-task learning, machine translation) |
| SkipTrans-UNet | Skip-path Translation U-Net | No "SkipTrans" found. However, "Skip-Transformer (DL-ST)" exists in blind face restoration (Frontiers in Signal Processing, 2023), and "Trans" suggests a transformer | Not recommended: likely to be misread as transformer-based |

**Recommendation:** **RLT-Net.**
- No files, experiment IDs or results were renamed.
- After the authors approve, rename in the manuscript text only. Keep the repository identifiers (DSLT-A, Tfull, Tnone) and add a mapping table in the supplementary material.
- Before committing to the name, run a Google Scholar and arXiv search for "RLT-Net" and "RLTNet".

## 5. References: verification results

Checked on 24 Nov 2026 by web search against publisher, proceedings, CVF, PubMed Central, Project Euclid and JSTOR records.

- "Confirmed" means the title, authors, venue and year matched a primary or authoritative source.
- No DOI was written unless a source showed it.

| # | Reference | Title | Authors | Venue / year | DOI | Action |
|---|---|---|---|---|---|---|
| 1 | Wan et al., Bringing Old Photos Back to Life | ✔ | ✔ (7) | ✔ CVPR 2020, pp. 2747–2757 | not found (arXiv 2004.09484) | Get the IEEE DOI from Xplore |
| 2 | Wan et al., Old Photo Restoration via Deep Latent Space Translation | ✔ | ⚠ catalogue lists 6 authors | **Corrected:** TPAMI **vol. 45 no. 2, 2023**, pp. 2071–2087 (early access 2022; the draft said 2022) | 10.1109/TPAMI.2022.3163183 | Confirm the author list on Xplore |
| 3 | Ronneberger et al., U-Net | ✔ | ✔ | ✔ MICCAI 2015, LNCS 9351, pp. 234–241 | 10.1007/978-3-319-24574-4_28 | — |
| 4 | Wang et al., ESRGAN | ✔ | ⚠ the arXiv version adds X. Tang | ✔ ECCV Workshops 2018, pp. 63–79 | 10.1007/978-3-030-11021-5_5 | Use the published author list |
| 5 | Woo et al., CBAM | ✔ | ✔ | ✔ ECCV 2018, LNCS 11211, pp. 3–19 | 10.1007/978-3-030-01234-2_1 | — |
| 6 | Shi et al., sub-pixel CNN | ✔ | ✔ (8) | ✔ CVPR 2016, pp. 1874–1883 | not found | Get the IEEE DOI |
| 7 | Yu & Koltun, dilated convolutions | ✔ | ✔ | ✔ ICLR 2016 | none (ICLR) | — |
| 8 | Perez et al., FiLM | ✔ | ✔ | ✔ AAAI 2018, vol. 32 no. 1 | 10.1609/aaai.v32i1.11671 | — |
| 9 | Wang et al., SFT | ✔ | ✔ | ✔ CVPR 2018, pp. 606–615 | not found | Get the IEEE DOI |
| 10 | Zamir et al., Restormer | ✔ | ✔ (6) | ✔ CVPR 2022, pp. 5728–5739 | not found | Get the IEEE DOI |
| 11 | Potlapalli et al., PromptIR | **Corrected:** the NeurIPS title is "PromptIR: Prompting for All-in-One Image Restoration" ("Blind" appears only on arXiv) | ✔ | ✔ NeurIPS 2023 | not found | Add the proceedings pages |
| 12 | Chen et al., NAFNet | ✔ | ✔ | ✔ ECCV 2022, LNCS 13667 | 10.1007/978-3-031-20071-7_2 | Add pages |
| 13 | Li et al., AirNet | ✔ | ✔ (6) | ✔ CVPR 2022; ⚠ pages 17452–17462 per CVF vs. 17431–17441 per DBLP | 10.1109/CVPR52688.2022.01693 (secondary source only) | Confirm the DOI and pages on Xplore |
| 14 | Zhang et al., LPIPS | ✔ | ✔ | ✔ CVPR 2018, pp. 586–595 | 10.1109/CVPR.2018.00068 (search summary only) | Confirm on Xplore |
| 15 | Wang et al., SSIM | ✔ | ✔ | ✔ IEEE TIP 13(4):600–612, 2004 | 10.1109/TIP.2003.819861 | — |
| 16 | Johnson et al., perceptual losses | ✔ | ✔ | ✔ ECCV 2016 | 10.1007/978-3-319-46475-6_43 | Add LNCS volume and pages |
| 17 | Simonyan & Zisserman, VGG | ✔ | ✔ | ✔ ICLR 2015 | none (arXiv 1409.1556) | — |
| 18 | Charbonnier et al. | ✔ | ✔ | ✔ ICIP 1994, vol. 2, pp. 168–172 | not found | Get the IEEE DOI |
| 19 | Loshchilov & Hutter, AdamW | ✔ | ✔ | ✔ ICLR 2019 | none (arXiv 1711.05101) | — |
| 20 | Loshchilov & Hutter, SGDR | ✔ | ✔ | ✔ ICLR 2017 | none (arXiv 1608.03983) | — |
| 21 | Schuirmann, TOST | ✔ | ✔ | ✔ J. Pharmacokinet. Biopharm. 15:657–680, 1987 | 10.1007/BF01068419 | — |
| 22 | Lakens, equivalence tests | ✔ | ✔ | ✔ SPPS 8(4):355–362, 2017 | 10.1177/1948550617697177 | — |
| 23 | Holm, step-down procedure | ✔ | ✔ | ✔ Scand. J. Stat. 6(2):65–70, 1979 | JSTOR 4615733; DOI 10.2307/4615733 not shown by a primary source | Confirm the DOI |
| 24 | Wilcoxon, ranking methods | ✔ | ✔ | ✔ Biometrics Bulletin 1(6):80–83, 1945 | not found (the commonly cited 10.2307/3001968 was **not** confirmed) | Confirm on JSTOR or Crossref |
| 25 | Efron, bootstrap | ✔ | ✔ | ✔ Ann. Statist. 7(1):1–26, 1979 | 10.1214/aos/1176344552 | — |

**Summary of references**
- **Incorrect or wrong references:** none of the 25. Every work exists with the stated title, authors and venue.
- **Corrected:**
  - #2: the year is now 2023 (vol. 45), with a note about early access in 2022.
  - #11: the title now matches the NeurIPS version.
- **DOIs confirmed:** 11 from publisher or primary records (#2, 3, 4, 5, 8, 12, 15, 16, 21, 22, 25). Four ICLR papers (#7, 17, 19, 20) have arXiv IDs instead, since ICLR does not issue DOIs.
- **DOIs still to obtain or confirm:**
  - IEEE DOIs not found: #1, 6, 9, 10, 18.
  - Secondary source only: #13 and #14.
  - Not confirmed: #23 and #24.
  - No DOI could be confirmed for #11.
- **In-text citations:** `[VERIFY: …]` placeholders were replaced with numbered citations. Three places where no reference exists yet are now `[TODO: reference needed, see ref. 27]`:
  - §1, the claim about single-run evaluation in the literature;
  - §3, old-photo and colourisation work and seed variance;
  - §13, implicit degradation estimation.

## 6. Scientific-claims check

| Check | Result | Corrections |
|---|---|---|
| Stages 0 and 2 described as three-seed, 800-step | ✔ Everywhere: abstract, §1, §2, §9 and §10 headings, Tables 1 and 3, conclusion | The conclusion now says "(three seeds, 800 steps)" for translators |
| Stage 3 described as one-seed, 20,000-step, exploratory | ✔ Abstract (now labelled "exploratory"), §1, §2, §11 heading and warning box, Table 6, §13, §14, conclusion | Added "in this run" / "in this single run" to contribution 4, Stage 3 finding 3, the Discussion and the conclusion |
| No generalisation from one seed | ✔ after fixes | **Fixed:** the abstract and conclusion said the models were "equivalent or non-inferior on mild and medium damage". That was wrong, because medium PSNR is *inconclusive*. Both now say: equivalent or non-inferior on all mild metrics and on medium SSIM and LPIPS; PSNR at medium and severe is inconclusive |
| Translators not called universally redundant or beneficial | ✔ | The abstract now says "possible, not yet replicated" severe-damage benefit. §1 now says "LPIPS and SSIM (not PSNR)" instead of "perceptual metrics". The conclusion scopes "no measurable benefit" to independently trained models at the short budget, because the Stage 1 intervention did find a small, significant severe-PSNR effect |
| Gating conclusion scoped | ✔ after fix | Abstract point 2 now says "at the short (800-step) budget where it was tested" |
| Synthetic-degradation limitation disclosed | ✔ Abstract ("synthetic degradation"), §6, §7 and §14 (limitation 1, first in the list) | — |
| No unsupported external-baseline superiority claims | ✔ §6 says no fair external comparison exists; §14.4 says no state-of-the-art claim is possible. The legacy Restormer/PromptIR numbers are not used | — |
| Unsupported statements about the literature | ⚠ Two found | §1 "improvements are often small, come from a single training run" became hedged wording plus [TODO ref]. The abstract's "rarely tested" became "not always tested under controlled, multi-seed conditions" |
| Internal cross-references | ⚠ Three broken | §16 → §14 (twice) and §11 → §12 (cost section) |

## 7. Complete list of changes in this pass

**`paper/DSLT_research_paper_draft.md`**
1. Header: QC-pass note and a naming note recommending RLT-Net (no renaming done).
2. Abstract:
   - Stage 3 labelled exploratory;
   - medium-damage equivalence corrected;
   - "in this run" and "image-level" qualifiers added;
   - "rarely" softened;
   - gating point scoped to 800 steps;
   - translator benefit marked "not yet replicated".
3. §1: literature claim hedged and marked [TODO ref]; "perceptual metrics" changed to "LPIPS and SSIM (not PSNR)".
4. §2: contribution 4 qualified to a single run.
5. §3:
   - placeholder note replaced;
   - Wan et al. description made accurate (two VAEs plus latent mapping; partial non-local block for scratches);
   - mechanism difference stated;
   - all [VERIFY] tags replaced with numbered citations.
6. §5: the A vs. Tfull initialisation finding stated (verified mechanism, unverified completeness).
7. §6: hardware corrected to Apple M4 Pro, 24 GB, PyTorch 2.15.0; LR schedule floor (0.5%) added in the previous pass.
8. §11: learning-curve interpretation rewritten (nominal lead vs. growing lead; no crossover point; sub-margin final lead; no validation LPIPS curve); finding 3 reworded.
9. §12 and §14: hardware wording corrected.
10. §13: "overtakes" replaced with a one-seed, in-this-run statement.
11. §15: translator statement scoped; "medium" wording corrected; "in this single run"; "can reverse" became "our results suggest… can change".
12. Cross-references fixed (§16 → §14 twice; §11 → §12).
13. References rewritten with verified details, DOIs where confirmed, explicit flags, two corrections (#2 year, #11 title), and a candidate related work (Touvron et al.) noted under TODO #27.

**`paper/RESULTS_AUDIT.md`**
- I-1, I-2 and I-3 rewritten with the evidence above.
- Missing-information items 2 (name) and 3 (references) updated.
- QC date added.
- I-10 had already been resolved in the previous pass.

**`paper/MANUSCRIPT_REVIEW.md`:** new; this file.

## 8. Unresolved issues

1. The **Stage 3 report** still says "≈12k" (historical file, not edited).
2. **I-3 attribution:** the decoder-initialisation difference is verified, but whether it explains the whole 0.09 dB gap is not tested.
3. The **method name** must be chosen by the authors. RLT-Net is recommended, pending a deeper literature search.
4. **References:**
   - 9 DOIs need an IEEE Xplore, JSTOR or Crossref check: #1, 6, 9, 10, 13, 14, 18, 23, 24;
   - #2 author list;
   - pages for #11, 12, 13 and 16.
5. **References not yet sourced:** seed-variance literature, implicit degradation estimation, old-photo and colourisation work, dataset citation.
6. **No validation LPIPS curve:** it was not logged, and producing it would need re-evaluating saved checkpoints. Only the best and last checkpoints exist per run, so a full curve cannot be produced without retraining.
7. **Historical reports and README** still say "Apple M4", by instruction.

## 9. Remaining submission requirements

1. **Dataset provenance:** source, licence, citation, collection description and original resolution. This is mandatory.
2. **Stage 3 replication:** seeds 123 and 2026 (about 17 h) before the severe-damage effect can be stated as more than exploratory. The runner and analysis script exist but are uncommitted.
3. **Fair external baselines** (Restormer, PromptIR, NAFNet) under the same protocol and seeds, or an explicit statement that the paper is a within-architecture study. Reviewers will likely ask for this.
4. **Real-photograph qualitative evaluation:** 87 photos in `data/raw`.
5. **Figures:**
   - architecture diagram;
   - Stage 3 learning curves (`learning_curves.png` exists);
   - visual comparisons (Stage 1 and 2 grids exist; no Stage 3 grid).
6. **Statistical details in the text:** bootstrap replicate count (10,000) and RNG seed.
7. **Front and back matter:** authors, affiliations, acknowledgements, data/code availability (checkpoints of 169–180 MB are not in git) and the target venue's template.
8. **Final reference pass:** complete DOIs and pages (§5) and resolve the TODO citations.
9. **Optional:**
   - inference-only translator attribution on the Stage 3 Tfull checkpoint;
   - re-test of gating at the long budget.
