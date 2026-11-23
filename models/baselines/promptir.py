"""
PromptIR (Potlapalli et al., NeurIPS 2023) -- clean from-scratch re-implementation.

PromptIR = Restormer backbone + Prompt Generation Modules (PGM) and Prompt
Interaction Modules (PIM) inserted at the three decoder stages (1/8, 1/4, 1/2
resolution).  PGM: input-conditioned weights (global-avg-pool -> linear ->
softmax) mix N learnable prompt components, which are bilinearly resized to the
feature resolution and refined by a 3x3 conv.  PIM: concatenate prompt with the
decoder feature, run a Transformer block, and reduce channels with a 1x1 conv.
The prompt makes the single network degradation-adaptive ("all-in-one" blind
restoration) without an explicit degradation label.

REDUCED-SIZE re-implementation (see SIZE_CONFIGS); not the official checkpoint.
The official prompt dims (64/128/320 at dim=48) are scaled with `dim`.
forward(x): x in [0,1] (B,3,H,W) -> restored in [0,1]; H, W divisible by 8.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.restormer import (
    TransformerBlock, OverlapPatchEmbed, Downsample, Upsample, _blocks,
    pad_to_multiple, restore_full,
)

SIZE_CONFIGS = {
    'tiny':  dict(dim=24, num_blocks=(2, 2, 3, 4), num_refinement_blocks=2, heads=(1, 2, 4, 8)),
    'small': dict(dim=32, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2, heads=(1, 2, 4, 8)),
    'base':  dict(dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4, heads=(1, 2, 4, 8)),
}


class PromptGenBlock(nn.Module):
    """Prompt Generation Module: input-conditioned mixture of learnable prompt components."""
    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=False)

    def forward(self, x):
        b, _, h, w = x.shape
        emb = x.mean(dim=(-2, -1))                                   # (B, lin_dim)
        weights = F.softmax(self.linear_layer(emb), dim=1)           # (B, L)
        prompt = (weights[:, :, None, None, None] * self.prompt_param.expand(b, -1, -1, -1, -1)).sum(1)
        prompt = F.interpolate(prompt, (h, w), mode='bilinear', align_corners=False)
        return self.conv3x3(prompt)


class PromptIR(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=48, num_blocks=(4, 6, 6, 8),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66,
                 prompt_len=5, prompt_sizes=(16, 32, 64)):
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

        # prompt dims follow the official 64/128/320 @ dim=48 ratios (~4/3, 8/3, 20/3 x dim)
        r8 = lambda v: max(8, int(round(v / 8)) * 8)  # keep channels divisible by head counts
        pd3, pd2, pd1 = r8(dim * 20 / 3), r8(dim * 8 / 3), r8(dim * 4 / 3)
        # level 3 (1/8): PGM on latent, PIM (transformer + 1x1 reduce)
        self.prompt3 = PromptGenBlock(pd3, prompt_len, prompt_sizes[0], dim * 8)
        self.noise_level3 = TransformerBlock(dim * 8 + pd3, heads[3], ffn)
        self.reduce_noise_level3 = nn.Conv2d(dim * 8 + pd3, dim * 8, 1, bias=False)
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = _blocks(dim * 4, num_blocks[2], heads[2], ffn)

        # level 2 (1/4)
        self.prompt2 = PromptGenBlock(pd2, prompt_len, prompt_sizes[1], dim * 4)
        self.noise_level2 = TransformerBlock(dim * 4 + pd2, heads[2], ffn)
        self.reduce_noise_level2 = nn.Conv2d(dim * 4 + pd2, dim * 4, 1, bias=False)
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = _blocks(dim * 2, num_blocks[1], heads[1], ffn)

        # level 1 (1/2 -> 1)
        self.prompt1 = PromptGenBlock(pd1, prompt_len, prompt_sizes[2], dim * 2)
        self.noise_level1 = TransformerBlock(dim * 2 + pd1, heads[1], ffn)
        self.reduce_noise_level1 = nn.Conv2d(dim * 2 + pd1, dim * 2, 1, bias=False)
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = _blocks(dim * 2, num_blocks[0], heads[0], ffn)

        self.refinement = _blocks(dim * 2, num_refinement_blocks, heads[0], ffn)
        self.output = nn.Conv2d(dim * 2, out_ch, 3, padding=1, bias=False)

    def forward_residual(self, x):
        e1 = self.encoder_level1(self.patch_embed(x))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        lat = self.latent(self.down3_4(e3))

        lat = self.reduce_noise_level3(self.noise_level3(torch.cat([lat, self.prompt3(lat)], 1)))
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], 1)))

        d3 = self.reduce_noise_level2(self.noise_level2(torch.cat([d3, self.prompt2(d3)], 1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))

        d2 = self.reduce_noise_level1(self.noise_level1(torch.cat([d2, self.prompt1(d2)], 1)))
        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))
        return self.output(self.refinement(d1))

    def forward(self, x):
        return torch.clamp(x + self.forward_residual(x), 0.0, 1.0)


def build_promptir(size='small', **overrides):
    cfg = dict(SIZE_CONFIGS[size])
    cfg.update(overrides)
    return PromptIR(**cfg)
