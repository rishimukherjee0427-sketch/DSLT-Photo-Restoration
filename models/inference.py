"""
DSLT v5 Inference — Self-Ensemble TTA + Tile-Based Processing
Self-ensemble: Run 8 augmented versions, average → +0.3-0.5 dB free PSNR boost.
Tile-based: Process large images in overlapping tiles to avoid OOM.
"""

import torch
import torch.nn.functional as F
import numpy as np


def self_ensemble_inference(encoder, decoder, dslt, input_tensor, device):
    """
    Self-ensemble (Test-Time Augmentation) for higher quality.
    Runs inference on 8 augmented versions (4 rotations × 2 flips),
    then averages the results.

    Typically adds +0.3-0.5 dB PSNR for free (no retraining needed).

    Args:
        encoder, decoder, dslt: Model components
        input_tensor: [1, 3, H, W] tensor in [0, 1]
        device: torch device

    Returns:
        output: [1, 3, H, W] tensor in [0, 1]
    """
    augmented_outputs = []

    for flip in [False, True]:
        for k in range(4):  # 0°, 90°, 180°, 270°
            # Augment
            x = input_tensor.clone()
            if flip:
                x = torch.flip(x, dims=[3])  # Horizontal flip
            if k > 0:
                x = torch.rot90(x, k, dims=[2, 3])

            # Forward pass
            x = x.to(device)
            with torch.no_grad():
                bottleneck, skips = encoder(x)
                bn_t, skips_t = dslt(bottleneck, skips)
                correction = decoder(bn_t, skips_t)
                out = torch.clamp(x + correction, 0, 1)

            # Reverse augmentation
            if k > 0:
                out = torch.rot90(out, 4 - k, dims=[2, 3])
            if flip:
                out = torch.flip(out, dims=[3])

            augmented_outputs.append(out.cpu())

    # Average all 8 outputs
    result = torch.stack(augmented_outputs, dim=0).mean(dim=0)
    return result


def tile_inference(encoder, decoder, dslt, input_tensor, device,
                   tile_size=256, overlap=32):
    """
    Tile-based inference for processing large images without OOM.
    Splits image into overlapping tiles, processes each, and blends results.

    Args:
        encoder, decoder, dslt: Model components
        input_tensor: [1, 3, H, W] tensor in [0, 1]
        device: torch device
        tile_size: Size of each tile
        overlap: Overlap between tiles for smooth blending

    Returns:
        output: [1, 3, H, W] tensor in [0, 1]
    """
    _, _, h, w = input_tensor.shape
    stride = tile_size - overlap

    # Pad image so tiles fit evenly
    pad_h = (stride - (h - tile_size) % stride) % stride
    pad_w = (stride - (w - tile_size) % stride) % stride
    padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, ph, pw = padded.shape

    output = torch.zeros_like(padded)
    weight_map = torch.zeros_like(padded)

    # Create blending weight (cosine feathering at edges)
    blend_1d = torch.ones(tile_size)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap)
        blend_1d[:overlap] = ramp
        blend_1d[-overlap:] = ramp.flip(0)
    blend_2d = blend_1d.unsqueeze(0) * blend_1d.unsqueeze(1)
    blend_2d = blend_2d.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            tile = padded[:, :, y:y+tile_size, x:x+tile_size].to(device)

            with torch.no_grad():
                bottleneck, skips = encoder(tile)
                bn_t, skips_t = dslt(bottleneck, skips)
                correction = decoder(bn_t, skips_t)
                tile_out = torch.clamp(tile + correction, 0, 1).cpu()

            output[:, :, y:y+tile_size, x:x+tile_size] += tile_out * blend_2d
            weight_map[:, :, y:y+tile_size, x:x+tile_size] += blend_2d

    # Normalize by weight
    output = output / (weight_map + 1e-8)

    # Remove padding
    output = output[:, :, :h, :w]
    return torch.clamp(output, 0, 1)


def restore_image(encoder, decoder, dslt, input_tensor, device,
                  use_tta=True, use_tiles=False, tile_size=256, overlap=32):
    """
    High-level restoration function.
    Chooses between simple, TTA, or tiled inference based on image size and options.

    Args:
        encoder, decoder, dslt: Model components (should be in eval mode)
        input_tensor: [1, 3, H, W] tensor in [0, 1]
        device: torch device
        use_tta: Whether to use self-ensemble (slower but +0.3-0.5 dB)
        use_tiles: Whether to use tile-based processing
        tile_size: Tile size for tiled inference
        overlap: Overlap for tiled inference

    Returns:
        output: [1, 3, H, W] tensor in [0, 1]
    """
    _, _, h, w = input_tensor.shape

    # Auto-enable tiles for large images
    if h > 512 or w > 512:
        use_tiles = True

    encoder.eval()
    decoder.eval()
    dslt.eval()

    if use_tiles:
        return tile_inference(encoder, decoder, dslt, input_tensor, device,
                            tile_size=tile_size, overlap=overlap)
    elif use_tta:
        return self_ensemble_inference(encoder, decoder, dslt, input_tensor, device)
    else:
        # Simple single-pass inference
        with torch.no_grad():
            x = input_tensor.to(device)
            bottleneck, skips = encoder(x)
            bn_t, skips_t = dslt(bottleneck, skips)
            correction = decoder(bn_t, skips_t)
            output = torch.clamp(x + correction, 0, 1)
        return output.cpu()
