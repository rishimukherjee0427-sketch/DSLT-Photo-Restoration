# Dataset Provenance Trace

*Traced 24 Nov 2026 from local evidence only: zip listings (nothing extracted), macOS download metadata (`kMDItemWhereFroms`), build scripts and git history. Signed download URLs are not reproduced here because they contain access tokens; only dataset identifiers are given.*

## 1. Where `data/train_clean` comes from: established

| Step | Evidence |
|---|---|
| `data/train_clean/img_0000.jpg` … `img_4999.jpg` (5,000 files) | Written by `utils/degradation.py::generate_training_pairs()` (in the repository since the initial commit `b22c557`, 2026-04-12) |
| Source folder | `CLEAN_SOURCE = "data/clean_raw/BnW vs Colored/Colored"`, `MAX_IMAGES = 5000`. The folder holds exactly 5,000 files (`COLOR (n).jpg/png`), so no subsampling took place |
| Archive | `data/archive (5).zip`: 10,000 files, `BnW vs Colored/BnW` (5,000) + `BnW vs Colored/Colored` (5,000); 4,163,033,167 bytes uncompressed; file timestamps 2025-04-24 |
| Download origin (macOS metadata on the zip) | Kaggle, dataset bundle **ID 7238026**, version ID 11541649. Downloaded 2026-03-26 (signed-URL date) |
| Processing applied | Resized to 256×256. **Random horizontal and vertical flips baked into the stored clean images** (`augment_image`, `random.seed(42)` set only on the subsample path). This explains the rotated or flipped references seen in Stage 1/2 visual checks. The degraded twins in `data/train_degraded` are not used by the current pipeline, which degrades online or uses cached deterministic benchmark inputs |
| Stale comment | `generate_training_pairs` says "Collect ALL images from DAVIS subdirectories". The code reads the Colored folder, not DAVIS. The comment is outdated, not evidence of a DAVIS origin |

## 2. Identity and licence of Kaggle #7238026: **not established**

- **The zip contains no licence, README or citation file.**
- The Kaggle dataset's title, owner and licence could not be read locally. Only the numeric ID is recorded.
- **Candidate match (unconfirmed):** Hugging Face `prithivMLmods/BnW-vs-Colored-10K`.
  - Same name, 10,000 images, labels "B & W" and "Colored", 4.15 GB.
  - Its card image is dated 24 Apr 2025, the same date as the timestamps in our zip.
  - The card states `license: apache-2.0`.
  - **Caveat:** the card gives no original source or collection method for the images. An Apache-2.0 tag from the curator does not establish that the curator had rights to the underlying photographs.
- **Content:** the images are predominantly faces/portraits; the earlier audit noted 368 exact-duplicate groups and many flip/resize near-duplicates.

**Action needed (cannot be done from this machine without your Kaggle login):**
1. Open your Kaggle account's download history, or the dataset page for ID 7238026.
2. Record its title, owner, licence and URL.
3. Check whether it is the same as the Hugging Face dataset above and whether either states an image source.

If no licence or source is stated, the paper must say so. Consider whether a dataset with a clear licence is needed for publication (for example FFHQ, CelebA-HQ or DIV2K, each with its own licence terms to check).

## 3. Other archives in `data/` (not used for training)

| Archive | Origin (download metadata) | Content | Use in the paper |
|---|---|---|---|
| `archive (8).zip` | Kaggle bundle ID 4045250 | 29 real old photographs (`old_photo_01…29`, `k.jpg`). These are the **29 unique** images behind the 87 files in `data/raw` (duplicated copies) | **Candidate for the real-photo qualitative panel.** Licence unknown; check the Kaggle page |
| `MLRN (1).zip` | Mendeley Data `r4zm9sj6cf` v1 (Qingbo Hao, Dec 2023), includes `readme.pdf` | Old-photo restoration set with noisy and original images (1,416 files) | Possible extra real or paired source. Licence: read `readme.pdf` and the Mendeley page (Mendeley shows a licence per dataset) |
| `archive (4).zip` | Kaggle bundle ID 9117807 | "Historical Facial Images Dataset Public Sample": **selfies of identifiable people with age, gender, ethnicity and device metadata** | **Do not use in figures.** Contains personal data |
| `archive (3).zip` | Kaggle bundle ID 275283 | Face images (selfies, celebrities) | Not used |
| `archive (6).zip` | Kaggle bundle ID 3590572 | Landscape colour/grey images (16,328 files) | Not used |
| `archive (7).zip` | Kaggle bundle ID 8902293 | "Clean candidates GT" low-resolution photographs (201 files) | Not used |

## 4. What the paper can state now

> "Clean training and test images are 5,000 colour images taken from the 'BnW vs Colored' collection (Kaggle dataset ID 7238026, the colour half), resized to 256×256 [licence: TODO]. The stored images include random horizontal and vertical flips applied during preparation. Near-duplicates were removed across splits by perceptual hashing."

Do not cite a licence until step 2 above is done.
