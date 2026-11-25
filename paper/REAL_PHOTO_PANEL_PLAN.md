# Real-Photo Qualitative Panel: Plan (Task 5; not run yet)

**Status:** plan only. The panel needs the GPU for inference, so it waits until the Stage 3 seed runs finish (one MPS job at a time on this machine).

## Inputs

- **Source:** the 29 unique real old photographs in `data/raw/archive (2)/` (= `data/archive (8).zip`, Kaggle bundle 4045250). They have no ground truth, so the panel is qualitative only.
- **Licence:** must be confirmed before any image appears in the paper (see `DATASET_PROVENANCE.md`).
- **Excluded:**
  - `archive (4)` (identifiable people with demographic metadata);
  - any training or test image.

## Selection rule (fixed before looking at outputs)

1. Sort the 29 filenames and drop any that are not photographs (inspect only the *inputs*, for example drawings or text).
2. Pick 10 inputs to cover the damage types visible in the *input*, judged on the input alone and before any model is run:
   - 4 with scratches or tears;
   - 3 with fading or colour cast;
   - 3 with blur, noise or grain.
   If a category has too few images, fill it in filename order.
3. Record the chosen filenames in `paper/build/out/real_panel_selection.json` **before** running any model.

## Models (validation-selected checkpoints only; no test-set use)

1. Input (unprocessed).
2. Full model, Stage 3 (`results/v9_convergence_pilot/seed42/Tfull_best.pth`).
3. Encoder–decoder, Stage 3 (`.../seed42/Tnone_best.pth`).
4. Optional: DSLT-C, Stage 0 seed 42 (gated, 800 steps), to show the gating variant. This is a different training budget and must be labelled as such.

If seeds 123/2026 finish, keep seed 42 for the panel (the pre-set choice) and do not pick the seed that looks best.

## Inference protocol

- RGB in [0, 1].
- Resize the long side to 512 px with area interpolation, then reflect-pad to a multiple of 16. Crop back after inference.
- Single forward pass: **no TTA, no tiling** unless memory requires it, and then the same tiling for all models.
- Save PNGs to `paper/build/out/real_panel/`.
- Generate the figure (rows = photos, columns = models; 300 dpi PDF/PNG) with a script under `paper/build/`, not by hand.

## Caption requirements

- "Real scanned photographs, no ground truth; models trained only on synthetic degradation. Quantitative results (Tables …) are on synthetic degradation."
- State the training budget of each model and that the selection rule was fixed in advance.
- No claims of perceptual superiority from this panel.

**Estimated cost:** under 5 minutes of GPU time for 10 images × 3 models.
