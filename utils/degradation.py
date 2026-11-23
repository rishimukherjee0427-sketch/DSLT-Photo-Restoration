"""
Enhanced Synthetic Degradation Pipeline — v3
Generates 2500 paired images with comprehensive degradation.
Adds: salt-pepper noise, color fading, vignette, higher B&W ratio.
"""

import cv2
import numpy as np
import os
import random
import sys

CLEAN_SOURCE = "data/clean_raw/BnW vs Colored/Colored"
OUTPUT_CLEAN = "data/train_clean"
OUTPUT_DEGRADED = "data/train_degraded"
TARGET_SIZE = (256, 256)
MAX_IMAGES = 5000


def add_gaussian_noise(img, sigma_range=(10, 40)):
    sigma = random.uniform(*sigma_range) / 255.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1).astype(np.float32)


def add_salt_pepper_noise(img, amount=None):
    if amount is None:
        amount = random.uniform(0.01, 0.05)
    result = img.copy()
    # Salt
    n_salt = int(amount * img.size / 2)
    salt_coords = [np.random.randint(0, d - 1, n_salt) for d in img.shape[:2]]
    result[salt_coords[0], salt_coords[1]] = 1.0
    # Pepper
    n_pepper = int(amount * img.size / 2)
    pepper_coords = [np.random.randint(0, d - 1, n_pepper) for d in img.shape[:2]]
    result[pepper_coords[0], pepper_coords[1]] = 0.0
    return result.astype(np.float32)


def add_gaussian_blur(img, k_range=(3, 9)):
    k = random.choice(range(k_range[0], k_range[1] + 1, 2))
    return cv2.GaussianBlur(img, (k, k), 0)


def add_heavy_blur(img):
    k = random.choice([11, 13, 15, 17])
    return cv2.GaussianBlur(img, (k, k), 0)


def add_sepia_tint(img, intensity=None):
    if intensity is None:
        intensity = random.uniform(0.3, 0.8)
    sepia = np.array([0.272, 0.534, 0.131], dtype=np.float32)
    tinted = img * (1 - intensity) + np.ones_like(img) * sepia * intensity
    return np.clip(tinted, 0, 1).astype(np.float32)


def reduce_contrast(img, factor=None):
    if factor is None:
        factor = random.uniform(0.3, 0.7)
    mean = img.mean()
    return np.clip(mean + factor * (img - mean), 0, 1).astype(np.float32)


def add_jpeg_artifacts(img, quality_range=(5, 30)):
    quality = random.randint(*quality_range)
    img_uint8 = (img * 255).astype(np.uint8)
    _, encoded = cv2.imencode('.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded.astype(np.float32) / 255.0


def add_scratches(img, num_scratches=None):
    scratched, _ = add_scratches_with_mask(img, num_scratches)
    return scratched


def add_scratches_with_mask(img, num_scratches=None):
    if num_scratches is None:
        num_scratches = random.randint(3, 12)
    h, w = img.shape[:2]
    result = img.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_scratches):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = random.randint(0, w - 1)
        y2 = random.randint(0, h - 1)
        color = random.uniform(0.6, 1.0)
        thickness = random.randint(1, 2)
        cv2.line(result, (x1, y1), (x2, y2), (color, color, color), thickness)
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness)
    return result, mask


def convert_to_grayscale(img):
    """Convert to B&W — the model must learn to colorize."""
    gray = np.mean(img, axis=2, keepdims=True)
    return np.repeat(gray, 3, axis=2).astype(np.float32)


def add_color_fading(img, fade=None):
    """Reduce color saturation to simulate faded vintage photos."""
    if fade is None:
        fade = random.uniform(0.3, 0.8)
    gray = np.mean(img, axis=2, keepdims=True)
    gray3 = np.repeat(gray, 3, axis=2)
    return np.clip(img * (1 - fade) + gray3 * fade, 0, 1).astype(np.float32)


def add_vignette(img, strength=None):
    """Dark corners seen in old photos."""
    if strength is None:
        strength = random.uniform(0.3, 0.7)
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1 - strength * (dist / max_dist) ** 2
    vignette = np.expand_dims(vignette, 2)
    return np.clip(img * vignette, 0, 1).astype(np.float32)


def add_broken_pixels(img, ratio=None):
    """Simulate dead/stuck pixels — bright or dark spots."""
    if ratio is None:
        ratio = random.uniform(0.001, 0.005)
    result = img.copy()
    h, w = img.shape[:2]
    n_broken = int(ratio * h * w)
    for _ in range(n_broken):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        if random.random() > 0.5:
            result[y, x] = [1.0, 0.0, 1.0]  # Hot pixel (magenta)
        else:
            result[y, x] = [0.0, 0.0, 0.0]  # Dead pixel
    return result.astype(np.float32)


def degrade_image(img):
    """Apply random combination of degradations."""
    degraded, _ = degrade_image_with_severity(img)
    return degraded


def degrade_image_with_severity(img):
    """
    Apply random combination of degradations and return 4-channel ground truth severity map:
    - Channel 0: Blur severity [0, 1] (broadcast scalar)
    - Channel 1: Noise severity [0, 1] (broadcast scalar)
    - Channel 2: Scratch severity mask [0, 1] (2D spatial mask matching scratches)
    - Channel 3: Fade severity [0, 1] (broadcast scalar)
    """
    h, w = img.shape[:2]
    degraded = img.copy()
    
    blur_sev = 0.0
    noise_sev = 0.0
    scratch_mask = np.zeros((h, w), dtype=np.float32)
    fade_sev = 0.0

    # 1. Blur
    if random.random() < 0.5:
        k = random.choice([3, 5, 7, 9])
        degraded = cv2.GaussianBlur(degraded, (k, k), 0)
        blur_sev = max(blur_sev, float(k / 17.0))
    elif random.random() < 0.2:
        k = random.choice([11, 13, 15, 17])
        degraded = cv2.GaussianBlur(degraded, (k, k), 0)
        blur_sev = max(blur_sev, float(k / 17.0))

    # 2. Noise
    if random.random() < 0.7:
        sigma = random.uniform(10, 40)
        noise = np.random.normal(0, sigma / 255.0, degraded.shape).astype(np.float32)
        degraded = np.clip(degraded + noise, 0, 1).astype(np.float32)
        noise_sev = max(noise_sev, float(sigma / 40.0))
    if random.random() < 0.3:
        degraded = add_salt_pepper_noise(degraded)
        noise_sev = max(noise_sev, 0.4)

    # 3. Fade / Tint / Contrast
    if random.random() < 0.5:
        intensity = random.uniform(0.3, 0.8)
        degraded = add_sepia_tint(degraded, intensity=intensity)
        fade_sev = max(fade_sev, float(intensity))
    if random.random() < 0.6:
        factor = random.uniform(0.3, 0.7)
        degraded = reduce_contrast(degraded, factor=factor)
        fade_sev = max(fade_sev, float(1.0 - factor))
    if random.random() < 0.5:
        fade = random.uniform(0.3, 0.8)
        degraded = add_color_fading(degraded, fade=fade)
        fade_sev = max(fade_sev, float(fade))

    # 4. Scratches
    if random.random() < 0.35:
        degraded, s_mask = add_scratches_with_mask(degraded)
        scratch_mask = np.maximum(scratch_mask, s_mask)

    # Other non-tracked degradations
    if random.random() < 0.4:
        degraded = add_jpeg_artifacts(degraded)
    if random.random() < 0.4:
        degraded = convert_to_grayscale(degraded)
    if random.random() < 0.3:
        degraded = add_vignette(degraded)
    if random.random() < 0.2:
        degraded = add_broken_pixels(degraded)

    # Construct severity map (H, W, 4)
    severity_map = np.zeros((h, w, 4), dtype=np.float32)
    severity_map[:, :, 0] = np.clip(blur_sev, 0, 1)
    severity_map[:, :, 1] = np.clip(noise_sev, 0, 1)
    severity_map[:, :, 2] = np.clip(scratch_mask, 0, 1)
    severity_map[:, :, 3] = np.clip(fade_sev, 0, 1)

    return degraded, severity_map


def augment_image(img):
    """Random augmentation: flips + slight rotation."""
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # Vertical flip
    return img


def generate_training_pairs():
    os.makedirs(OUTPUT_CLEAN, exist_ok=True)
    os.makedirs(OUTPUT_DEGRADED, exist_ok=True)

    # Collect ALL images from DAVIS subdirectories
    all_images = []
    if os.path.exists(CLEAN_SOURCE):
        for fname in sorted(os.listdir(CLEAN_SOURCE)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(CLEAN_SOURCE, fname))

    # Subsample
    if len(all_images) > MAX_IMAGES:
        random.seed(42)
        all_images = random.sample(all_images, MAX_IMAGES)

    print(f"Generating {len(all_images)} training pairs...")

    count = 0
    for i, img_path in enumerate(all_images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, TARGET_SIZE)
        img_float = img.astype(np.float32) / 255.0

        # Apply augmentation to clean image
        img_float = augment_image(img_float)

        # Generate degraded version
        degraded = degrade_image(img_float)

        # Save
        cv2.imwrite(
            os.path.join(OUTPUT_CLEAN, f"img_{count:04d}.jpg"),
            (img_float * 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DEGRADED, f"img_{count:04d}.jpg"),
            (degraded * 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        count += 1

        if count % 100 == 0:
            print(f"  [{count}/{len(all_images)}] pairs generated")

    print(f"✅ Done! {count} pairs saved to {OUTPUT_CLEAN}/ and {OUTPUT_DEGRADED}/")


if __name__ == "__main__":
    generate_training_pairs()
