"""
Restormer (Zamir et al., CVPR 2022) -- clean from-scratch re-implementation.

Faithful to the paper's architecture: 4-level U-shaped transformer built from
Transformer blocks = [LayerNorm -> MDTA (multi-Dconv-head transposed attention)]
+ [LayerNorm -> GDFN (gated-Dconv feed-forward network)], pixel-unshuffle /
pixel-shuffle down/up-sampling, 1x1 conv fusion of skip connections, a
refinement stage at full resolution and a global residual output.

This is a REDUCED-SIZE re-implementation (see SIZE_CONFIGS); it is NOT the
official checkpoint and is trained from scratch on this project's data.

forward(x): x in [0,1] (B,3,H,W) -> restored in [0,1]; H, W must be divisible
by 8 (use `restore_full` / `pad_to_multiple` for arbitrary sizes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# size -> kwargs.  'base' is the official Restormer configuration (~26M params).
SIZE_CONFIGS = {
    'tiny':  dict(dim=24, num_blocks=(2, 2, 3, 4), num_refinement_blocks=2, heads=(1, 2, 4, 8)),
    'small': dict(dim=32, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2, heads=(1, 2, 4, 8)),
    'base':  dict(dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4, heads=(1, 2, 4, 8)),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm with bias ('WithBias' variant), applied on (B,C,H,W)."""
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, 1e-5)
        return x.permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """Multi-Dconv head Transposed Attention: attention across channels (C x C map)."""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        hd = self.num_heads
        q = q.reshape(b, hd, c // hd, h * w)
        k = k.reshape(b, hd, c // hd, h * w)
        v = v.reshape(b, hd, c // hd, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-forward Network."""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.use_checkpoint = False  # activation checkpointing (saves memory, ~+30% time)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, dim, 3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    """conv (C -> C/2) + PixelUnshuffle(2): C -> 2C, resolution / 2."""
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """conv (C -> 2C) + PixelShuffle(2): C -> C/2, resolution x 2."""
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def _blocks(dim, n, heads, ffn):
    return nn.Sequential(*[TransformerBlock(dim, heads, ffn) for _ in range(n)])


# --------------------------------------------------------------------------
# Restormer
# --------------------------------------------------------------------------
class Restormer(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=48, num_blocks=(4, 6, 6, 8),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66):
        super().__init__()
        ffn = ffn_expansion_factor
        self.patch_embed = OverlapPatchEmbed(in_ch, dim)

        self.encoder_level1 = _blocks(dim, num_blocks[0], heads[0], ffn)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = _blocks(dim * 2, num_blocks[1], heads[1], ffn)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = _blocks(dim * 4, num_blocks[2], heads[2], ffn)
        self.down3_4 = Downsample(dim * 4)
        self.latent = _blocks(dim * 8, num_blocks[3], heads[3], ffn)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = _blocks(dim * 4, num_blocks[2], heads[2], ffn)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = _blocks(dim * 2, num_blocks[1], heads[1], ffn)

        self.up2_1 = Upsample(dim * 2)  # no channel reduction at level 1 (as in the paper)
        self.decoder_level1 = _blocks(dim * 2, num_blocks[0], heads[0], ffn)

        self.refinement = _blocks(dim * 2, num_refinement_blocks, heads[0], ffn)
        self.output = nn.Conv2d(dim * 2, out_ch, 3, padding=1, bias=False)

    def forward_residual(self, x):
        e1 = self.encoder_level1(self.patch_embed(x))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        lat = self.latent(self.down3_4(e3))

        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], 1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))
        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))
        return self.output(self.refinement(d1))

    def forward(self, x):
        return torch.clamp(x + self.forward_residual(x), 0.0, 1.0)


def set_grad_checkpoint(model, enabled=True):
    """Enable activation checkpointing on every TransformerBlock of a baseline model."""
    for m in model.modules():
        if isinstance(m, TransformerBlock):
            m.use_checkpoint = enabled
    return model


def build_restormer(size='small', **overrides):
    cfg = dict(SIZE_CONFIGS[size])
    cfg.update(overrides)
    return Restormer(**cfg)


# --------------------------------------------------------------------------
# Full-image helpers (shared by PromptIR)
# --------------------------------------------------------------------------
def pad_to_multiple(x, multiple=16, mode='reflect'):
    """Reflect-pad (B,C,H,W) on bottom/right so H, W are multiples of `multiple`.
    Returns (padded, (H, W)). Falls back to replicate where reflect is invalid."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    m = mode
    if m == 'reflect' and (ph >= h or pw >= w):
        m = 'replicate'
    return F.pad(x, (0, pw, 0, ph), mode=m), (h, w)


@torch.no_grad()
def restore_full(model, img, multiple=16):
    """Run `model` on a full image tensor (B,3,H,W) or (3,H,W) in [0,1] of any size:
    reflect-pad to a multiple of 16, forward, crop back."""
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)
    padded, (h, w) = pad_to_multiple(img, multiple)
    out = model(padded)[..., :h, :w]
    return out.squeeze(0) if squeeze else out
