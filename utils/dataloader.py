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


class OnlineDegradationDataset(Dataset):
    """
    Clean-image-only dataset with online (on-the-fly) degradation.
    Every epoch generates unique degraded versions → effectively infinite data.
    Supports progressive training with different crop sizes.
    """

    def __init__(self, clean_dir, crop_size=256, augment=True):
        self.clean_dir = clean_dir
        self.crop_size = crop_size
        self.augment = augment
        self.filenames = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"OnlineDegradationDataset: {len(self.filenames)} images, crop={crop_size}")

    def __len__(self):
        return len(self.filenames)

    def set_crop_size(self, size):
        """Called between phases for progressive training."""
        self.crop_size = size
        print(f"  Crop size updated to {size}×{size}")

    def _load_and_crop(self, path):
        """Load image and take a random crop at current crop_size."""
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = img.shape[:2]

        # If image is smaller than crop size, resize up
        min_side = min(h, w)
        if min_side < self.crop_size:
            scale = self.crop_size / min_side * 1.1  # slightly larger to allow cropping
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        # Random crop
        y = random.randint(0, h - self.crop_size)
        x = random.randint(0, w - self.crop_size)
        return img[y:y+self.crop_size, x:x+self.crop_size]

    def _degrade_online(self, img):
        """Apply random degradation pipeline on-the-fly."""
        degraded = img.copy()

        # Noise (applied with high probability)
        if random.random() < 0.75:
            sigma = random.uniform(5, 35) / 255.0
            noise = np.random.normal(0, sigma, degraded.shape).astype(np.float32)
            degraded = np.clip(degraded + noise, 0, 1)

        # Salt-and-pepper noise
        if random.random() < 0.25:
            amount = random.uniform(0.005, 0.03)
            n_salt = int(amount * degraded.size / 2)
            coords = [np.random.randint(0, d - 1, n_salt) for d in degraded.shape[:2]]
            degraded[coords[0], coords[1]] = 1.0
            coords = [np.random.randint(0, d - 1, n_salt) for d in degraded.shape[:2]]
            degraded[coords[0], coords[1]] = 0.0

        # Blur
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            degraded = cv2.GaussianBlur(degraded, (k, k), 0)

        # Heavy blur (rare)
        if random.random() < 0.15:
            k = random.choice([11, 13, 15])
            degraded = cv2.GaussianBlur(degraded, (k, k), 0)

        # Motion blur
        if random.random() < 0.2:
            size = random.choice([5, 7, 9, 11])
            kernel = np.zeros((size, size))
            if random.random() < 0.5:
                kernel[size // 2, :] = 1.0  # Horizontal
            else:
                kernel[:, size // 2] = 1.0  # Vertical
            kernel /= size
            degraded = cv2.filter2D(degraded, -1, kernel)

        # JPEG compression
        if random.random() < 0.4:
            quality = random.randint(10, 40)
            img_uint8 = (degraded * 255).astype(np.uint8)
            _, encoded = cv2.imencode('.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # Color degradations
        if random.random() < 0.4:
            # Sepia tint
            intensity = random.uniform(0.2, 0.7)
            sepia = np.array([0.272, 0.534, 0.131], dtype=np.float32)
            degraded = degraded * (1 - intensity) + np.ones_like(degraded) * sepia * intensity
            degraded = np.clip(degraded, 0, 1)

        if random.random() < 0.3:
            # Color fading
            fade = random.uniform(0.2, 0.6)
            gray = np.mean(degraded, axis=2, keepdims=True)
            degraded = degraded * (1 - fade) + np.repeat(gray, 3, axis=2) * fade
            degraded = np.clip(degraded, 0, 1).astype(np.float32)

        # Contrast reduction
        if random.random() < 0.5:
            factor = random.uniform(0.4, 0.8)
            mean = degraded.mean()
            degraded = np.clip(mean + factor * (degraded - mean), 0, 1).astype(np.float32)

        # Grayscale conversion (important for colorization task)
        if random.random() < 0.3:
            gray = np.mean(degraded, axis=2, keepdims=True)
            degraded = np.repeat(gray, 3, axis=2).astype(np.float32)

        # Scratches
        if random.random() < 0.3:
            h, w = degraded.shape[:2]
            for _ in range(random.randint(2, 8)):
                x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
                x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
                color = random.uniform(0.6, 1.0)
                cv2.line(degraded, (x1, y1), (x2, y2), (color, color, color), random.randint(1, 2))

        # Vignette
        if random.random() < 0.25:
            strength = random.uniform(0.2, 0.5)
            h, w = degraded.shape[:2]
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
            vignette = 1 - strength * (dist / np.sqrt((w/2)**2 + (h/2)**2))**2
            degraded = np.clip(degraded * vignette[:, :, np.newaxis], 0, 1).astype(np.float32)

        # Downscale then upscale (simulate low resolution)
        if random.random() < 0.3:
            scale = random.uniform(0.3, 0.7)
            h, w = degraded.shape[:2]
            small = cv2.resize(degraded, (int(w * scale), int(h * scale)))
            degraded = cv2.resize(small, (w, h)).astype(np.float32)

        return degraded.astype(np.float32)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.clean_dir, fname)

        clean = self._load_and_crop(path)
        if clean is None:
            return self.__getitem__((idx + 1) % len(self.filenames))

        # Apply online degradation
        degraded = self._degrade_online(clean)

        if self.augment:
            # Synchronized augmentations
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
