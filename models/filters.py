"""
Post-processing beauty filters for DSLT restored images.
Applied AFTER restoration: Vivid, Warm, Cool, B&W, Sharpen, Colorize.
"""

import cv2
import numpy as np


def apply_vivid(img_rgb):
    """Boost saturation + contrast for vivid, punchy colors."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_warm(img_rgb):
    """Warm amber/golden color temperature shift."""
    r = img_rgb.copy().astype(np.float32)
    r[:, :, 0] = np.clip(r[:, :, 0] * 1.1 + 10, 0, 255)
    r[:, :, 1] = np.clip(r[:, :, 1] * 1.03 + 5, 0, 255)
    r[:, :, 2] = np.clip(r[:, :, 2] * 0.9 - 5, 0, 255)
    return r.astype(np.uint8)


def apply_cool(img_rgb):
    """Cool blue/teal color temperature shift."""
    r = img_rgb.copy().astype(np.float32)
    r[:, :, 0] = np.clip(r[:, :, 0] * 0.9 - 5, 0, 255)
    r[:, :, 1] = np.clip(r[:, :, 1] * 1.03 + 3, 0, 255)
    r[:, :, 2] = np.clip(r[:, :, 2] * 1.12 + 10, 0, 255)
    return r.astype(np.uint8)


def apply_bw(img_rgb):
    """Artistic black & white with dramatic contrast."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def apply_sharpen(img_rgb):
    """Unsharp mask for enhanced edge detail."""
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), 3)
    sharpened = cv2.addWeighted(img_rgb, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_colorize(img_rgb):
    """
    Intelligent colorization: enhances faded/desaturated images
    by boosting saturation + applying warm toning on low-color areas.
    For true B&W → uses sepia tone + saturation boost.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    avg_sat = hsv[:, :, 1].mean()

    if avg_sat < 30:
        # Near-grayscale: apply warm sepia-like colorization
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # Sepia mapping (warm brown tones)
        r = np.clip(gray * 240 + 20, 0, 255).astype(np.uint8)
        g = np.clip(gray * 200 + 10, 0, 255).astype(np.uint8)
        b = np.clip(gray * 150, 0, 255).astype(np.uint8)
        result = np.stack([r, g, b], axis=2)
        # CLAHE for contrast
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Has some color: boost saturation strongly
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6 + 15, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Filter registry
FILTERS = {
    "vivid": apply_vivid,
    "warm": apply_warm,
    "cool": apply_cool,
    "bw": apply_bw,
    "sharpen": apply_sharpen,
    "colorize": apply_colorize,
}


def apply_filter(img_rgb_uint8, filter_name):
    """Apply a named filter to RGB uint8 image."""
    if filter_name not in FILTERS:
        return img_rgb_uint8
    return FILTERS[filter_name](img_rgb_uint8)
