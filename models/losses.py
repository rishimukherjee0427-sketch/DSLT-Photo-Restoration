"""
==== GATE-SUPERVISION API (added for adaptive gating; all OFF by default) ====
CombinedLoss(device, ..., w_gate_sup=0.0, w_gate_sparsity=0.0, w_gate_bimodal=0.0,
             gate_target_a=0.2, gate_target_b=0.8, gate_sup_type='l1')
  * CombinedLoss.forward(output, target, pred_severity=None, target_severity=None,
                         pred_gates=None)  -> SAME 6-tuple as before
        (total, loss_char, loss_p, loss_ssim, loss_adv, loss_damage).
    If pred_gates (B,4) and target_severity (B,4,H,W) are given and any gate weight > 0,
    the weighted gate loss is ADDED to `total`; the terms are exposed (detached) as
        self.last_gate_loss       weighted total gate loss (tensor, 0 if off)
        self.last_gate_terms      dict(sup=, sparsity=, bimodal=)  UNweighted scalars (floats)
  * CombinedLoss.compute_gate_loss(pred_gates, target_severity) -> (weighted_loss, terms_dict)
        callable on its own (e.g. if you add the term to `total` yourself).
  * CombinedLoss.gate_targets(target_severity) / module fn `gate_targets_from_severity(...)`
        -> (B,4) target gates for [e2,e3,e4,bottleneck].
  Mapping: per-image damage scalars n_c = clip(max_over_space(sev_c)/norm_c, 0, 1) with
  max (not mean) so sparse scratches count; norm = (blur 0.5, noise 0.6, scratch 1, fade 0.5)
  ~ the max values the 'mild' pipeline can emit. Per-scale severity
        s_scale = clip(sum_c M[scale, c] * n_c, 0, 1),
  M rows [e2,e3,e4,bn] over [blur,noise,scratch,fade]:
        e2 (.5,.5,.3,0)  e3 (.4,.4,.5,.2)  e4 (.2,.2,.4,.5)  bn (0,.1,.2,.8)
  (fine scales for blur/noise, deep scales for global fade/contrast), and
        target_gate = clip(a + b * s_scale, 0, 1),  a=gate_target_a, b=gate_target_b.
  Terms: sup = L1 (or BCE if gate_sup_type='bce') between pred and target gates;
         sparsity = mean(pred_gates) (L1 on gate mean); bimodal = mean(g*(1-g)) (pushes to 0/1).
==============================================================================

Loss Functions for DSLT v5 — Charbonnier + FFT Frequency + Multi-Scale Perceptual + SSIM + LAB
Designed for maximum PSNR/SSIM while maintaining perceptual quality.
MPS-compatible, optimized for vintage image restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss — robust variant of L1 that avoids over-smoothing.
    More stable gradients near zero compared to L1.
    """
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon_sq = epsilon ** 2

    def forward(self, output, target):
        diff = output - target
        return torch.mean(torch.sqrt(diff * diff + self.epsilon_sq))


class FFTFrequencyLoss(nn.Module):
    """
    Frequency-domain loss using FFT.
    Penalizes differences in frequency space to preserve edges, textures,
    and high-frequency details that pixel-space losses tend to miss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # 2D FFT on spatial dimensions
        fft_output = torch.fft.rfft2(output, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')

        # Loss on magnitude spectrum (amplitude)
        mag_loss = F.l1_loss(fft_output.abs(), fft_target.abs())

        # Loss on phase (important for structure)
        phase_loss = F.l1_loss(
            torch.angle(fft_output),
            torch.angle(fft_target)
        )

        return mag_loss + 0.1 * phase_loss


class SSIMLoss(nn.Module):
    """Differentiable SSIM Loss (1 - SSIM)."""
    def __init__(self, window_size=11, channels=3, device="cpu"):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer('window', window)

    def forward(self, output, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        mu1 = F.conv2d(output, self.window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channels)
        sigma1_sq = F.conv2d(output ** 2, self.window, padding=pad, groups=self.channels) - mu1 ** 2
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=pad, groups=self.channels) - mu2 ** 2
        sigma12 = F.conv2d(output * target, self.window, padding=pad, groups=self.channels) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class MultiScalePerceptualLoss(nn.Module):
    """
    Multi-scale VGG16 perceptual loss.
    Extracts features from conv1_2, conv2_2, conv3_3, conv4_3 for multi-scale matching.
    Much more effective than single-layer perceptual loss.
    """
    def __init__(self, device, normalize=True):
        super().__init__()
        # torchvision VGG16 weights expect ImageNet-normalised input. Inputs here are in [0, 1].
        # normalize=False reproduces the legacy (pre-v7, un-normalised) behaviour exactly.
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        features = list(vgg.features)

        # Feature extraction at multiple scales
        # conv1_2: index 3, conv2_2: index 8, conv3_3: index 15, conv4_3: index 22
        self.slice1 = nn.Sequential(*features[:4]).to(device)   # conv1_2
        self.slice2 = nn.Sequential(*features[4:9]).to(device)  # conv2_2
        self.slice3 = nn.Sequential(*features[9:16]).to(device) # conv3_3
        self.slice4 = nn.Sequential(*features[16:23]).to(device)# conv4_3

        for module in [self.slice1, self.slice2, self.slice3, self.slice4]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        # Weights for each scale (deeper features weighted less — they're more abstract)
        self.weights = [1.0, 0.8, 0.6, 0.4]

    def forward(self, output, target):
        loss = 0.0
        if self.normalize:
            x_out = (output - self.mean) / self.std
            x_tgt = (target - self.mean) / self.std
        else:
            x_out = output
            x_tgt = target

        for i, (slice_module, w) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4],
            self.weights
        )):
            x_out = slice_module(x_out)
            x_tgt = slice_module(x_tgt)
            loss += w * F.l1_loss(x_out, x_tgt)

        return loss


class LABColorLoss(nn.Module):
    """
    Computes L1 loss on the continuous A and B color channels of LAB space.
    Approximated cleanly in tensor operations to fight pink/magenta artifacts.
    Assumes inputs are RGB [0, 1].
    """
    def __init__(self):
        super().__init__()
        # RGB to XYZ matrix
        M = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]).view(3, 3, 1, 1)
        self.register_buffer('M', M)

    def rgb_to_xyz(self, rgb):
        # Inverse sRGB gamma safely
        rgb = torch.clamp(rgb, 0.0, 1.0)
        mask = (rgb > 0.04045).float()
        val = torch.clamp((rgb + 0.055) / 1.055, min=1e-6)
        rgb_linear = mask * torch.pow(val, 2.4) + (1 - mask) * (rgb / 12.92)
        xyz = F.conv2d(rgb_linear, self.M)
        return xyz

    def xyz_to_lab(self, xyz):
        # D65 white point
        xyz_norm = xyz / torch.tensor([0.950456, 1.0, 1.088754], device=xyz.device).view(1, 3, 1, 1)
        xyz_norm = torch.clamp(xyz_norm, min=1e-6)
        mask = (xyz_norm > 0.008856).float()
        f_xyz = mask * torch.pow(xyz_norm, 1/3) + (1 - mask) * (7.787 * xyz_norm + 16/116)

        L = 116 * f_xyz[:, 1:2] - 16
        a = 500 * (f_xyz[:, 0:1] - f_xyz[:, 1:2])
        b = 200 * (f_xyz[:, 1:2] - f_xyz[:, 2:3])
        return torch.cat([L, a, b], dim=1)

    def forward(self, output, target):
        lab_out = self.xyz_to_lab(self.rgb_to_xyz(output))
        lab_tgt = self.xyz_to_lab(self.rgb_to_xyz(target))
        # We only care about color channels (a, b) to penalize color bleed
        return F.l1_loss(lab_out[:, 1:], lab_tgt[:, 1:])


class EdgeLoss(nn.Module):
    """
    Edge-aware loss using Laplacian operator.
    Ensures the model preserves sharp edges and fine structures.
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel for edge detection
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Repeat for 3 channels
        self.register_buffer('kernel', kernel.repeat(3, 1, 1, 1))

    def forward(self, output, target):
        edge_out = F.conv2d(output, self.kernel, padding=1, groups=3)
        edge_tgt = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(edge_out, edge_tgt)


# Rows: [e2, e3, e4, bottleneck]; cols: [blur, noise, scratch, fade]
GATE_MIX = ((0.5, 0.5, 0.3, 0.0),
            (0.4, 0.4, 0.5, 0.2),
            (0.2, 0.2, 0.4, 0.5),
            (0.0, 0.1, 0.2, 0.8))
GATE_NORM = (0.5, 0.6, 1.0, 0.5)


def severity_scalars(target_severity):
    """(B,4,H,W) GT severity map -> (B,4) per-image damage scalars in [0,1] (max over space / norm)."""
    mx = target_severity.flatten(2).amax(dim=2)
    norm = torch.tensor(GATE_NORM, device=mx.device, dtype=mx.dtype).view(1, 4)
    return (mx / norm).clamp(0.0, 1.0)


def gate_targets_from_severity(target_severity, a=0.2, b=0.8):
    """(B,4,H,W) GT severity -> (B,4) target gates [e2,e3,e4,bn]; see header for the mapping."""
    n = severity_scalars(target_severity)                                   # (B,4)
    mix = torch.tensor(GATE_MIX, device=n.device, dtype=n.dtype)            # (4 scales, 4 damages)
    s = (n @ mix.t()).clamp(0.0, 1.0)                                       # (B,4 scales)
    return (a + b * s).clamp(0.0, 1.0)


class CombinedLoss(nn.Module):
    """
    v5 Combined Loss:
    Charbonnier + Multi-Scale Perceptual + SSIM + LAB + FFT Frequency + Edge
    + Optional Adversarial (GAN) loss for sharpness.
    Balanced for maximum PSNR while maintaining perceptual quality.

    Set use_discriminator=True and pass a discriminator to enable GAN sharpening.
    Discriminator is OPTIONAL — set use_discriminator=False to disable anytime.
    """
    def __init__(self, device, use_discriminator=False, discriminator=None, mode='psnr',
                 w_gate_sup=0.0, w_gate_sparsity=0.0, w_gate_bimodal=0.0,
                 gate_target_a=0.2, gate_target_b=0.8, gate_sup_type='l1', vgg_normalize=True):
        super().__init__()
        self.charbonnier = CharbonnierLoss(epsilon=1e-3)
        # vgg_normalize=True: corrected ImageNet normalisation (v7). False = legacy behaviour.
        self.vgg_normalize = vgg_normalize
        self.perceptual = MultiScalePerceptualLoss(device, normalize=vgg_normalize)
        self.ssim_loss = SSIMLoss(device=device)
        self.lab_loss = LABColorLoss().to(device)
        self.fft_loss = FFTFrequencyLoss().to(device)
        self.edge_loss = EdgeLoss().to(device)

        # Set loss weights based on mode
        self.set_mode(mode)
        self.w_damage = 0.15

        # Optional gate supervision (all off by default; see header)
        self.w_gate_sup = w_gate_sup
        self.w_gate_sparsity = w_gate_sparsity
        self.w_gate_bimodal = w_gate_bimodal
        self.gate_target_a = gate_target_a
        self.gate_target_b = gate_target_b
        assert gate_sup_type in ('l1', 'bce')
        self.gate_sup_type = gate_sup_type
        self.last_gate_loss = torch.tensor(0.0)
        self.last_gate_terms = {'sup': 0.0, 'sparsity': 0.0, 'bimodal': 0.0}

        # GAN components (optional)
        self.use_discriminator = use_discriminator
        self.discriminator = discriminator
        self.adv_criterion = nn.BCEWithLogitsLoss()

    def set_mode(self, mode='psnr'):
        """
        Switch between loss weight presets:
          'psnr':     Maximum PSNR — minimize perceptual, boost pixel-level losses
          'balanced': Original balanced weights for perceptual+pixel quality
          'gan':      GAN fine-tuning mode — adversarial loss enabled
        """
        if mode == 'psnr':
            # PSNR-maximizing: Charbonnier dominates, minimal perceptual
            self.w_perceptual = 0.01     # Perceptual hurts PSNR — keep minimal
            self.w_ssim = 0.25           # SSIM strongly correlated with PSNR
            self.w_lab = 0.02            # Light color correction
            self.w_fft = 0.05            # Light frequency supervision
            self.w_edge = 0.03           # Light edge preservation
            self.w_adversarial = 0.0     # No GAN for PSNR mode
        elif mode == 'balanced':
            # Original balanced weights
            self.w_perceptual = 0.05
            self.w_ssim = 0.15
            self.w_lab = 0.05
            self.w_fft = 0.10
            self.w_edge = 0.05
            self.w_adversarial = 0.0
        elif mode == 'gan':
            # GAN fine-tuning: perceptual + adversarial for sharpness
            self.w_perceptual = 0.05
            self.w_ssim = 0.15
            self.w_lab = 0.05
            self.w_fft = 0.10
            self.w_edge = 0.05
            self.w_adversarial = 0.02

    def gate_targets(self, target_severity):
        return gate_targets_from_severity(target_severity, self.gate_target_a, self.gate_target_b)

    def compute_gate_loss(self, pred_gates, target_severity):
        """
        Args:
            pred_gates: (B,4) in [0,1] from DamageEstimator
            target_severity: (B,4,H,W) GT severity map from OnlineDegradationDataset
        Returns: (weighted_loss tensor, dict of UNweighted float terms)
        """
        pg = pred_gates.float()
        zero = pg.new_zeros(())
        l_sup, l_sp, l_bi = zero, zero, zero
        if self.w_gate_sup > 0 and target_severity is not None:
            tgt = self.gate_targets(target_severity.float()).detach()
            if self.gate_sup_type == 'bce':
                l_sup = F.binary_cross_entropy(pg.clamp(1e-6, 1 - 1e-6), tgt)
            else:
                l_sup = F.l1_loss(pg, tgt)
        if self.w_gate_sparsity > 0:
            l_sp = pg.mean()
        if self.w_gate_bimodal > 0:
            l_bi = (pg * (1.0 - pg)).mean()
        total = (self.w_gate_sup * l_sup + self.w_gate_sparsity * l_sp
                 + self.w_gate_bimodal * l_bi)
        return total, {'sup': float(l_sup.detach()), 'sparsity': float(l_sp.detach()),
                       'bimodal': float(l_bi.detach())}

    def forward(self, output, target, pred_severity=None, target_severity=None, pred_gates=None):
        loss_char = self.charbonnier(output, target)
        loss_p = self.perceptual(output, target)
        loss_ssim = self.ssim_loss(output, target)
        loss_lab = self.lab_loss(output, target)
        loss_fft = self.fft_loss(output, target)
        loss_edge = self.edge_loss(output, target)

        total = (loss_char
                 + self.w_perceptual * loss_p
                 + self.w_ssim * loss_ssim
                 + self.w_lab * loss_lab
                 + self.w_fft * loss_fft
                 + self.w_edge * loss_edge)

        loss_damage = torch.tensor(0.0, device=output.device)
        if (pred_severity is not None) and (target_severity is not None):
            loss_damage = F.l1_loss(pred_severity, target_severity)
            total = total + self.w_damage * loss_damage

        # Optional gate supervision / regularisation (no-op unless weights > 0)
        self.last_gate_loss = torch.zeros((), device=output.device)
        self.last_gate_terms = {'sup': 0.0, 'sparsity': 0.0, 'bimodal': 0.0}
        if pred_gates is not None and (self.w_gate_sup > 0 or self.w_gate_sparsity > 0
                                       or self.w_gate_bimodal > 0):
            loss_gate, terms = self.compute_gate_loss(pred_gates, target_severity)
            total = total + loss_gate
            self.last_gate_loss = loss_gate.detach()
            self.last_gate_terms = terms

        # Adversarial loss: generator tries to fool discriminator
        loss_adv = torch.tensor(0.0, device=output.device)
        if self.use_discriminator and self.discriminator is not None and self.w_adversarial > 0:
            fake_preds = self.discriminator(output)
            # Handle multi-scale discriminator (list of outputs)
            if isinstance(fake_preds, list):
                loss_adv = sum(
                    self.adv_criterion(pred, torch.ones_like(pred))
                    for pred in fake_preds
                ) / len(fake_preds)
            else:
                loss_adv = self.adv_criterion(fake_preds, torch.ones_like(fake_preds))
            total = total + self.w_adversarial * loss_adv

        return total, loss_char, loss_p, loss_ssim, loss_adv, loss_damage

    def discriminator_loss(self, real_images, fake_images):
        """
        Compute discriminator loss separately.
        Uses label smoothing (0.9 for real) for training stability.

        Args:
            real_images: [B, 3, H, W] clean ground-truth images
            fake_images: [B, 3, H, W] generator outputs (detached!)

        Returns:
            d_loss: scalar discriminator loss
            d_real_acc: accuracy on real images (for monitoring)
            d_fake_acc: accuracy on fake images (for monitoring)
        """
        if self.discriminator is None:
            return torch.tensor(0.0), 0.0, 0.0

        real_preds = self.discriminator(real_images)
        fake_preds = self.discriminator(fake_images.detach())

        # Handle multi-scale discriminator
        if isinstance(real_preds, list):
            loss_real = sum(
                self.adv_criterion(pred, torch.ones_like(pred) * 0.9)  # Label smoothing
                for pred in real_preds
            ) / len(real_preds)
            loss_fake = sum(
                self.adv_criterion(pred, torch.zeros_like(pred))
                for pred in fake_preds
            ) / len(fake_preds)
            # Accuracy from first scale
            d_real_acc = (real_preds[0] > 0).float().mean().item()
            d_fake_acc = (fake_preds[0] < 0).float().mean().item()
        else:
            loss_real = self.adv_criterion(real_preds, torch.ones_like(real_preds) * 0.9)
            loss_fake = self.adv_criterion(fake_preds, torch.zeros_like(fake_preds))
            d_real_acc = (real_preds > 0).float().mean().item()
            d_fake_acc = (fake_preds < 0).float().mean().item()

        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss, d_real_acc, d_fake_acc
