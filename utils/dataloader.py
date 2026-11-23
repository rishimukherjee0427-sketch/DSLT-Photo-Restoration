"""
Dataset classes for DSLT v5 training and inference.
Features:
- Online degradation (infinite variety, no pre-generated pairs needed)
- Random crop at multiple scales for progressive training
- Strong augmentation: flips, rotations, color jitter, mixup-ready
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random
import zlib


class VintageDataset(Dataset):
    """Single-image dataset for inference/testing."""

    def __init__(self, folder):
        self.paths = []
        for root, dirs, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            return self.__getitem__((idx + 1) % len(self.paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256)).astype('float32') / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)


# ═══════════════════════════════════════════════════════════════════════
#  Degradation severity presets
# ═══════════════════════════════════════════════════════════════════════
#
#  Each entry: probability of applying the op (p) + its parameter range.
#  'mild' and 'heavy' are EXACTLY the legacy v5 values (mild is bit-identical
#  to the pre-refactor code for a given RNG stream; 'heavy' likewise).
#  'medium' lies between mild and heavy; 'severe' is stronger than heavy.
#
#  op                      | mild              | medium            | heavy (legacy)     | severe
#  ------------------------+-------------------+-------------------+--------------------+--------------------
#  gaussian noise p        | 0.60              | 0.68              | 0.75               | 0.85
#    sigma (/255)          | U(3, 20)          | U(3, 27)          | U(3, 35)           | U(5, 50)
#  salt&pepper p           | 0.10              | 0.17              | 0.25               | 0.40
#    amount                | U(.002, .015)     | U(.002, .022)     | U(.002, .030)      | U(.005, .050)
#  gaussian blur p         | 0.35              | 0.42              | 0.50               | 0.65
#    kernel k (odd)        | {3, 5}            | {3, 5, 7}         | {3, 5, 7}          | {5, 7, 9}
#  heavy blur p            | 0 (off)           | 0.08              | 0.15               | 0.30
#    kernel k              | -                 | {9, 11}           | {11, 13, 15}       | {11, 13, 15, 17, 21}
#  motion blur p           | 0.10              | 0.15              | 0.20               | 0.35
#    kernel size           | {3, 5}            | {3, 5, 7, 9}      | {5, 7, 9, 11}      | {7, 9, 11, 15}
#  JPEG p                  | 0.30              | 0.35              | 0.40               | 0.55
#    quality               | U{35..70}         | U{22..55}         | U{10..40}          | U{5..30}
#  sepia p                 | 0.25              | 0.32              | 0.40               | 0.55
#    intensity             | U(.10, .40)       | U(.15, .55)       | U(.20, .70)        | U(.30, .85)
#  colour fade p           | 0.20              | 0.25              | 0.30               | 0.45
#    amount                | U(.10, .35)       | U(.15, .47)       | U(.20, .60)        | U(.30, .80)
#  contrast reduction p    | 0.30              | 0.40              | 0.50               | 0.65
#    factor                | U(.60, .90)       | U(.50, .85)       | U(.40, .80)        | U(.30, .70)
#  grayscale p             | 0.08              | 0.19              | 0.30               | 0.45
#  scratches p             | 0.10              | 0.20              | 0.30               | 0.50
#    count                 | U{1..3}           | U{1..5}           | U{1..8}            | U{3..15}
#  vignette p              | 0.15              | 0.20              | 0.25               | 0.35
#    strength              | U(.10, .30)       | U(.15, .40)       | U(.20, .50)        | U(.30, .65)
#  down/up-scale p         | 0.15              | 0.22              | 0.30               | 0.45
#    scale                 | U(.50, .80)       | U(.40, .75)       | U(.30, .70)        | U(.20, .60)
#
#  severity='mixed': each sample picks mild / medium / severe uniformly.
#  Ground-truth severity map (4 ch, [0,1]): blur=max(k/15), noise=max(sigma*255/35, 0.4 if S&P),
#  scratch mask, fade=max(sepia, fade, 1-contrast). The normalisation constants
#  are the same for all levels, so 'severe' values saturate at 1.0 (clipped).

SEVERITY_PRESETS = {
    'mild': dict(
        noise_p=0.60, noise_sigma=(3, 20), sp_p=0.10, sp_amount=(0.002, 0.015),
        blur_p=0.35, blur_k=[3, 5], hblur_p=None, hblur_k=[11, 13, 15],
        motion_p=0.10, motion_k=[3, 5], jpeg_p=0.30, jpeg_q=(35, 70),
        sepia_p=0.25, sepia_i=(0.1, 0.4), fade_p=0.20, fade_a=(0.1, 0.35),
        contrast_p=0.30, contrast_f=(0.6, 0.9), gray_p=0.08,
        scratch_p=0.10, scratch_n=(1, 3), vig_p=0.15, vig_s=(0.1, 0.3),
        down_p=0.15, down_s=(0.5, 0.8)),
    'medium': dict(
        noise_p=0.68, noise_sigma=(3, 27), sp_p=0.17, sp_amount=(0.002, 0.022),
        blur_p=0.42, blur_k=[3, 5, 7], hblur_p=0.08, hblur_k=[9, 11],
        motion_p=0.15, motion_k=[3, 5, 7, 9], jpeg_p=0.35, jpeg_q=(22, 55),
        sepia_p=0.32, sepia_i=(0.15, 0.55), fade_p=0.25, fade_a=(0.15, 0.47),
        contrast_p=0.40, contrast_f=(0.5, 0.85), gray_p=0.19,
        scratch_p=0.20, scratch_n=(1, 5), vig_p=0.20, vig_s=(0.15, 0.4),
        down_p=0.22, down_s=(0.4, 0.75)),
    'heavy': dict(  # legacy heavy
        noise_p=0.75, noise_sigma=(3, 35), sp_p=0.25, sp_amount=(0.002, 0.03),
        blur_p=0.50, blur_k=[3, 5, 7], hblur_p=0.15, hblur_k=[11, 13, 15],
        motion_p=0.20, motion_k=[5, 7, 9, 11], jpeg_p=0.40, jpeg_q=(10, 40),
        sepia_p=0.40, sepia_i=(0.2, 0.7), fade_p=0.30, fade_a=(0.2, 0.6),
        contrast_p=0.50, contrast_f=(0.4, 0.8), gray_p=0.30,
        scratch_p=0.30, scratch_n=(1, 8), vig_p=0.25, vig_s=(0.2, 0.5),
        down_p=0.30, down_s=(0.3, 0.7)),
    'severe': dict(
        noise_p=0.85, noise_sigma=(5, 50), sp_p=0.40, sp_amount=(0.005, 0.05),
        blur_p=0.65, blur_k=[5, 7, 9], hblur_p=0.30, hblur_k=[11, 13, 15, 17, 21],
        motion_p=0.35, motion_k=[7, 9, 11, 15], jpeg_p=0.55, jpeg_q=(5, 30),
        sepia_p=0.55, sepia_i=(0.3, 0.85), fade_p=0.45, fade_a=(0.3, 0.8),
        contrast_p=0.65, contrast_f=(0.3, 0.7), gray_p=0.45,
        scratch_p=0.50, scratch_n=(3, 15), vig_p=0.35, vig_s=(0.3, 0.65),
        down_p=0.45, down_s=(0.2, 0.6)),
}
MIXED_LEVELS = ('mild', 'medium', 'severe')
VALID_SEVERITIES = ('mild', 'medium', 'heavy', 'severe', 'mixed')


class OnlineDegradationDataset(Dataset):
    """
    Clean-image-only dataset with online (on-the-fly) degradation.
    Every epoch generates unique degraded versions -> effectively infinite data.
    Supports progressive training with different crop sizes.

    Args:
        clean_dir:   folder of clean images.
        crop_size:   square crop (int) or None for full image resolution.
        augment:     random flips / rot90 (+ random crop) when True.
        severity:    'mild' | 'medium' | 'severe' | 'mixed' | 'heavy' (legacy alias of
                     the original heavy preset). See the preset table above.
        file_list:   optional txt file (one filename per line, relative to clean_dir)
                     restricting the dataset (e.g. data/splits/train.txt).
        deterministic: if True, sample idx ALWAYS returns the identical degraded image
                     (RNG seeded from (seed, idx, severity)), centre crop and no flips.
                     Intended for val/test/benchmark use. Does not touch global RNG state.
        seed:        base seed used when deterministic=True.

    Legacy behaviour kept: augment=False with deterministic=False seeds the global
    RNG with idx*31337 (old 'validation' mode, random crop included).
    """

    def __init__(self, clean_dir, crop_size=256, augment=True, severity='mild',
                 file_list=None, deterministic=False, seed=0):
        if severity not in VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {VALID_SEVERITIES}, got {severity!r}")
        self.clean_dir = clean_dir
        self.crop_size = crop_size
        self.augment = augment
        self.severity = severity
        self.deterministic = deterministic
        self.seed = seed
        if file_list is not None:
            with open(file_list) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
            missing = [n for n in names if not os.path.isfile(os.path.join(clean_dir, n))]
            if missing:
                raise FileNotFoundError(f"{len(missing)} files from {file_list} missing in {clean_dir}, e.g. {missing[:3]}")
            self.filenames = names
        else:
            self.filenames = sorted([
                f for f in os.listdir(clean_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        print(f"OnlineDegradationDataset: {len(self.filenames)} images, crop={crop_size}, "
              f"severity={severity}{', deterministic' if deterministic else ''}")

    def __len__(self):
        return len(self.filenames)

    def set_crop_size(self, size):
        """Called between phases for progressive training."""
        self.crop_size = size
        print(f"  Crop size updated to {size}x{size}")

    def _load_and_crop(self, path, rng=random):
        """Load image and crop at current crop_size (random, or centre if deterministic).
        crop_size=None returns the full image."""
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.crop_size is None:
            return img
        h, w = img.shape[:2]

        # If image is smaller than crop size, resize up
        min_side = min(h, w)
        if min_side < self.crop_size:
            scale = self.crop_size / min_side * 1.1  # slightly larger to allow cropping
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        if self.deterministic:
            y = (h - self.crop_size) // 2
            x = (w - self.crop_size) // 2
        else:
            y = rng.randint(0, h - self.crop_size)
            x = rng.randint(0, w - self.crop_size)
        return img[y:y+self.crop_size, x:x+self.crop_size]

    def _degrade_online(self, img, rng=random, nrng=np.random, level=None):
        """Apply random degradation pipeline on-the-fly and track severity.

        rng / nrng: python `random`-like and numpy `np.random`-like objects
        (module or instance). `level` overrides self.severity (used by 'mixed').
        Returns:
            degraded (np.ndarray): (H, W, 3) degraded image
            severity_map (np.ndarray): (H, W, 4) ground truth severity target
        """
        if level is None:
            level = self.severity
            if level == 'mixed':
                level = rng.choice(MIXED_LEVELS)
        P = SEVERITY_PRESETS[level]
        h, w = img.shape[:2]
        degraded = img.copy()

        blur_sev = 0.0
        noise_sev = 0.0
        scratch_mask = np.zeros((h, w), dtype=np.float32)
        fade_sev = 0.0

        # -- Noise ---------------------------------------------------------
        if rng.random() < P['noise_p']:
            sigma = rng.uniform(P['noise_sigma'][0], P['noise_sigma'][1]) / 255.0
            noise = nrng.normal(0, sigma, degraded.shape).astype(np.float32)
            degraded = np.clip(degraded + noise, 0, 1)
            noise_sev = max(noise_sev, float(sigma * 255.0 / 35.0))

        # Salt-and-pepper noise
        if rng.random() < P['sp_p']:
            amount = rng.uniform(P['sp_amount'][0], P['sp_amount'][1])
            n_salt = int(amount * degraded.size / 2)
            coords = [nrng.randint(0, d - 1, n_salt) for d in degraded.shape[:2]]
            degraded[coords[0], coords[1]] = 1.0
            coords = [nrng.randint(0, d - 1, n_salt) for d in degraded.shape[:2]]
            degraded[coords[0], coords[1]] = 0.0
            noise_sev = max(noise_sev, 0.4)

        # -- Blur ----------------------------------------------------------
        if rng.random() < P['blur_p']:
            k = rng.choice(P['blur_k'])
            degraded = cv2.GaussianBlur(degraded, (k, k), 0)
            blur_sev = max(blur_sev, float(k / 15.0))

        # Heavy blur (disabled in mild; no RNG draw consumed there, as before)
        if P['hblur_p'] is not None and rng.random() < P['hblur_p']:
            k = rng.choice(P['hblur_k'])
            degraded = cv2.GaussianBlur(degraded, (k, k), 0)
            blur_sev = max(blur_sev, float(k / 15.0))

        # Motion blur
        if rng.random() < P['motion_p']:
            size = rng.choice(P['motion_k'])
            kernel = np.zeros((size, size))
            if rng.random() < 0.5:
                kernel[size // 2, :] = 1.0  # Horizontal
            else:
                kernel[:, size // 2] = 1.0  # Vertical
            kernel /= size
            degraded = cv2.filter2D(degraded, -1, kernel)
            blur_sev = max(blur_sev, float(size / 15.0))

        # -- JPEG compression ----------------------------------------------
        if rng.random() < P['jpeg_p']:
            quality = rng.randint(P['jpeg_q'][0], P['jpeg_q'][1])
            img_uint8 = (degraded * 255).astype(np.uint8)
            _, encoded = cv2.imencode('.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # -- Color degradations --------------------------------------------
        if rng.random() < P['sepia_p']:
            intensity = rng.uniform(P['sepia_i'][0], P['sepia_i'][1])
            sepia = np.array([0.272, 0.534, 0.131], dtype=np.float32)
            degraded = degraded * (1 - intensity) + np.ones_like(degraded) * sepia * intensity
            degraded = np.clip(degraded, 0, 1)
            fade_sev = max(fade_sev, float(intensity))

        if rng.random() < P['fade_p']:
            fade = rng.uniform(P['fade_a'][0], P['fade_a'][1])
            gray = np.mean(degraded, axis=2, keepdims=True)
            degraded = degraded * (1 - fade) + np.repeat(gray, 3, axis=2) * fade
            degraded = np.clip(degraded, 0, 1).astype(np.float32)
            fade_sev = max(fade_sev, float(fade))

        # Contrast reduction
        if rng.random() < P['contrast_p']:
            factor = rng.uniform(P['contrast_f'][0], P['contrast_f'][1])
            mean = degraded.mean()
            degraded = np.clip(mean + factor * (degraded - mean), 0, 1).astype(np.float32)
            fade_sev = max(fade_sev, float(1.0 - factor))

        # Grayscale conversion
        if rng.random() < P['gray_p']:
            gray = np.mean(degraded, axis=2, keepdims=True)
            degraded = np.repeat(gray, 3, axis=2).astype(np.float32)

        # -- Scratches -----------------------------------------------------
        if rng.random() < P['scratch_p']:
            n_scratches = rng.randint(P['scratch_n'][0], P['scratch_n'][1])
            for _ in range(n_scratches):
                x1, y1 = rng.randint(0, w-1), rng.randint(0, h-1)
                x2, y2 = rng.randint(0, w-1), rng.randint(0, h-1)
                color = rng.uniform(0.6, 1.0)
                cv2.line(degraded, (x1, y1), (x2, y2), (color, color, color), 1)
                cv2.line(scratch_mask, (x1, y1), (x2, y2), 1.0, 1)

        # Vignette
        if rng.random() < P['vig_p']:
            strength = rng.uniform(P['vig_s'][0], P['vig_s'][1])
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
            vignette = 1 - strength * (dist / np.sqrt((w/2)**2 + (h/2)**2))**2
            degraded = np.clip(degraded * vignette[:, :, np.newaxis], 0, 1).astype(np.float32)

        # -- Downscale + upscale (simulate low resolution) -----------------
        if rng.random() < P['down_p']:
            scale = rng.uniform(P['down_s'][0], P['down_s'][1])
            small = cv2.resize(degraded, (int(w * scale), int(h * scale)))
            degraded = cv2.resize(small, (w, h)).astype(np.float32)

        severity_map = np.zeros((h, w, 4), dtype=np.float32)
        severity_map[:, :, 0] = np.clip(blur_sev, 0, 1)
        severity_map[:, :, 1] = np.clip(noise_sev, 0, 1)
        severity_map[:, :, 2] = np.clip(scratch_mask, 0, 1)
        severity_map[:, :, 3] = np.clip(fade_sev, 0, 1)

        return degraded.astype(np.float32), severity_map

    def _det_seed(self, idx):
        """Stable 32-bit seed from (seed, idx, severity) (platform independent)."""
        return zlib.crc32(f"{self.seed}|{idx}|{self.severity}".encode()) & 0xFFFFFFFF

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.clean_dir, fname)

        if self.deterministic:
            # Private RNGs: reproducible regardless of workers / call order.
            s = self._det_seed(idx)
            rng, nrng = random.Random(s), np.random.RandomState(s)
            clean = self._load_and_crop(path, rng)
            if clean is None:
                return self.__getitem__((idx + 1) % len(self.filenames))
            degraded, severity_map = self._degrade_online(clean, rng, nrng)
            return self._to_tensors(clean, degraded, severity_map)

        # Legacy validation mode: seed global RNG per-image
        if not self.augment:
            random.seed(idx * 31337)
            np.random.seed(idx * 31337 % (2**31))

        clean = self._load_and_crop(path)
        if clean is None:
            return self.__getitem__((idx + 1) % len(self.filenames))

        # Apply online degradation
        degraded, severity_map = self._degrade_online(clean)

        if self.augment:
            # Synchronized augmentations
            if random.random() > 0.5:
                clean = clean[:, ::-1, :].copy()
                degraded = degraded[:, ::-1, :].copy()
                severity_map = severity_map[:, ::-1, :].copy()
            if random.random() > 0.5:
                clean = clean[::-1, :, :].copy()
                degraded = degraded[::-1, :, :].copy()
                severity_map = severity_map[::-1, :, :].copy()
            # Random 90 deg rotations
            k = random.randint(0, 3)
            if k > 0:
                clean = np.rot90(clean, k).copy()
                degraded = np.rot90(degraded, k).copy()
                severity_map = np.rot90(severity_map, k).copy()

        # Restore random state after deterministic validation
        if not self.augment:
            random.seed()
            np.random.seed()

        return self._to_tensors(clean, degraded, severity_map)

    @staticmethod
    def _to_tensors(clean, degraded, severity_map):
        clean = torch.from_numpy(np.ascontiguousarray(clean)).permute(2, 0, 1)
        degraded = torch.from_numpy(np.ascontiguousarray(degraded)).permute(2, 0, 1)
        severity_map = torch.from_numpy(np.ascontiguousarray(severity_map)).permute(2, 0, 1)
        return degraded, clean, severity_map


class PairedDataset(Dataset):
    """
    Paired (degraded, clean) dataset for supervised DSLT training.
    Kept for backward compatibility and evaluation.
    """

    def __init__(self, clean_dir, degraded_dir, augment=True, crop_size=256):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.augment = augment
        self.crop_size = crop_size
        self.filenames = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"PairedDataset: {len(self.filenames)} pairs loaded")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        clean = cv2.imread(os.path.join(self.clean_dir, fname))
        degraded = cv2.imread(os.path.join(self.degraded_dir, fname))

        if clean is None or degraded is None:
            return self.__getitem__((idx + 1) % len(self.filenames))

        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        clean = cv2.resize(clean, (self.crop_size, self.crop_size)).astype('float32') / 255.0
        degraded = cv2.resize(degraded, (self.crop_size, self.crop_size)).astype('float32') / 255.0

        if self.augment:
            # Synchronized flips
            if random.random() > 0.5:
                clean = clean[:, ::-1, :].copy()
                degraded = degraded[:, ::-1, :].copy()
            if random.random() > 0.5:
                clean = clean[::-1, :, :].copy()
                degraded = degraded[::-1, :, :].copy()
            # Random 90° rotations
            k = random.randint(0, 3)
            if k > 0:
                clean = np.rot90(clean, k).copy()
                degraded = np.rot90(degraded, k).copy()

        clean = torch.from_numpy(np.ascontiguousarray(clean)).permute(2, 0, 1)
        degraded = torch.from_numpy(np.ascontiguousarray(degraded)).permute(2, 0, 1)
        return degraded, clean
