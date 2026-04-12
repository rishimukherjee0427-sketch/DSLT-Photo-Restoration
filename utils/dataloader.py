"""
Dataset classes for DSLT training and inference.
Includes strong augmentation for training: flips, color jitter, random blur, erasing.
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


class PairedDataset(Dataset):
    """
    Paired (degraded, clean) dataset for supervised DSLT training.
    Strong augmentation: random flips, color jitter, Gaussian blur, random erasing.
    """

    def __init__(self, clean_dir, degraded_dir, augment=True):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.augment = augment
        self.filenames = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"PairedDataset: {len(self.filenames)} pairs loaded")

    def __len__(self):
        return len(self.filenames)

    def _color_jitter(self, img, brightness=0.15, contrast=0.15, saturation=0.1):
        """Apply random color jitter to clean image only (simulates target variation)."""
        img = img.astype(np.float32)
        # Brightness
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-brightness, brightness)
            img = np.clip(img * factor, 0, 1)
        # Contrast
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = img.mean()
            img = np.clip(mean + factor * (img - mean), 0, 1)
        return img

    def _random_erase(self, img, p=0.15, scale=(0.02, 0.08)):
        """Random erasing — small rectangular region filled with noise."""
        if random.random() > p:
            return img
        h, w = img.shape[:2]
        area = h * w
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(0.5, 2.0)
        rh = int(np.sqrt(target_area * aspect))
        rw = int(np.sqrt(target_area / aspect))
        if rh < h and rw < w:
            y = random.randint(0, h - rh)
            x = random.randint(0, w - rw)
            img[y:y+rh, x:x+rw] = np.random.rand(rh, rw, 3).astype(np.float32)
        return img

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        clean = cv2.imread(os.path.join(self.clean_dir, fname))
        degraded = cv2.imread(os.path.join(self.degraded_dir, fname))

        if clean is None or degraded is None:
            return self.__getitem__((idx + 1) % len(self.filenames))

        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        clean = cv2.resize(clean, (256, 256)).astype('float32') / 255.0
        degraded = cv2.resize(degraded, (256, 256)).astype('float32') / 255.0

        if self.augment:
            # Synchronized flips
            if random.random() > 0.5:
                clean = clean[:, ::-1, :].copy()
                degraded = degraded[:, ::-1, :].copy()
            if random.random() > 0.5:
                clean = clean[::-1, :, :].copy()
                degraded = degraded[::-1, :, :].copy()

            # Color jitter on clean only (subtle)
            if random.random() > 0.7:
                clean = self._color_jitter(clean)

            # Random erasing on degraded only (extra corruption)
            degraded = self._random_erase(degraded)

            # Random Gaussian blur on degraded only
            if random.random() > 0.8:
                k = random.choice([3, 5])
                degraded = cv2.GaussianBlur(degraded, (k, k), 0)

        clean = torch.from_numpy(np.ascontiguousarray(clean)).permute(2, 0, 1)
        degraded = torch.from_numpy(np.ascontiguousarray(degraded)).permute(2, 0, 1)
        return degraded, clean
