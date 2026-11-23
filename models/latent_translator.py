"""
Multi-Scale DSLT v5 Latent Translator — 4-Scale with Dilated Convolutions
4-scale translation: e2(128ch) + e3(256ch) + e4(512ch) + bottleneck(512ch)
Uses dilated convolutions for larger receptive field and channel shuffle for better mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = self.sigmoid(avg_out + max_out) * x
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return mask * x


class DilatedResidualBlock(nn.Module):
    """
    Residual block with mixed dilated convolutions for larger receptive field.
    Uses dilation rates [1, 2] alternating for multi-scale feature capture.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
        self.cbam = CBAM(channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(x + self.cbam(self.block(x)))


class ScaleTranslator(nn.Module):
    """
    Multi-block translator for a single scale.
    Alternates between dilation=1 and dilation=2 for multi-scale receptive fields.
    """
    def __init__(self, channels, num_blocks):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            dilation = 2 if i % 2 == 1 else 1  # Alternate dilations
            blocks.append(DilatedResidualBlock(channels, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)
        # Residual scaling for training stability
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, z):
        return z + self.blocks(z) * self.scale


class GateModulator(nn.Module):
    """
    Turns the (B,4) damage-estimator gates into an expressive gate for ONE scale.

      logit(g) = logit(g_scalar[:, s])                       # supervised per-image scalar
               + SE(avgpool(z), all 4 gates)  -> (B, C, 1, 1) # FiLM-style per-channel offset
               + conv1x1(avg/max-pool(severity_map))  -> (B, 1, h, w)   # optional spatial offset

    Both offset branches are zero-initialised, so at init the gate equals the scalar gate
    exactly (and the module is a no-op relative to the scalar design); they only learn to
    deviate if that helps the restoration loss.
    """
    def __init__(self, channels, n_gates=4):
        super().__init__()
        hid = max(channels // 8, 16)
        self.fc1 = nn.Linear(channels + n_gates, hid)
        self.fc2 = nn.Linear(hid, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.spatial = nn.Conv2d(8, 1, 1)   # 4 severity ch x (avg, max)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, z, g_scalar, all_gates, severity_map=None):
        eps = 1e-4
        g = g_scalar.clamp(eps, 1 - eps)
        logit = torch.log(g / (1 - g)).view(-1, 1, 1, 1)
        pooled = z.mean(dim=(2, 3))
        h = F.relu(self.fc1(torch.cat([pooled, all_gates], dim=1)))
        logit = logit + self.fc2(h).view(z.size(0), -1, 1, 1)
        if severity_map is not None:
            size = z.shape[2:]
            sev = torch.cat([F.adaptive_avg_pool2d(severity_map, size),
                             F.adaptive_max_pool2d(severity_map, size)], dim=1)
            logit = logit + self.spatial(sev)
        return torch.sigmoid(logit)  # (B,C,1,1) or (B,C,h,w)


class MultiScaleDSLT(nn.Module):
    """
    v5: 4-Scale DSLT with dilated convolutions:
      - Level e2 (96ch, 1/2 res):      3 blocks
      - Level e3 (192ch, 1/4 res):     4 blocks
      - Level e4 (384ch, 1/8 res):     3 blocks
      - Bottleneck (384ch, 1/16 res):  4 blocks

    Gating (only active when scale_gates is passed and use_gating=True):
      gate_mode='scalar' : legacy. out = g*raw + (1-g)*orig with g (B,1,1,1) per scale.
      gate_mode='film'   : (default) out = orig + G*(raw-orig), G = GateModulator(...) is a
                           per-channel (and, if `severity_map` is passed, per-pixel) gate built
                           on top of the supervised scalar gate. Adds small extra params
                           (`gate_mods.*`); old checkpoints without them still load strictly
                           (a load pre-hook fills the missing keys with their init values).
    The e1 skip is never translated, hence never gated (identical in gated/ungated configs).
    Ungated path / call `dslt(bottleneck, skips)` is bit-identical to before.
    After a gated forward with `record_gates=True`, `self.last_gate_info` holds
      'eff_gates': (B,4) mean effective gate per scale, 'rel_delta': (B,4) ||raw-orig||/||orig||.
    """
    def __init__(self, gate_mode='film'):
        super().__init__()
        assert gate_mode in ('scalar', 'film')
        self.gate_mode = gate_mode
        self.translator_e2 = ScaleTranslator(96, num_blocks=3)
        self.translator_e3 = ScaleTranslator(192, num_blocks=4)
        self.translator_e4 = ScaleTranslator(384, num_blocks=3)
        self.translator_bn = ScaleTranslator(384, num_blocks=4)
        if gate_mode == 'film':
            self.gate_mods = nn.ModuleList([GateModulator(c) for c in (96, 192, 384, 384)])
            self._register_load_state_dict_pre_hook(self._fill_missing_gate_keys)
        self.record_gates = False
        self.last_gate_info = None

    def _fill_missing_gate_keys(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        """Backward compat: checkpoints saved before gate_mods existed load with strict=True."""
        for k, v in self.gate_mods.state_dict(prefix=prefix + 'gate_mods.').items():
            if k not in state_dict:
                state_dict[k] = v.clone()

    def forward(self, bottleneck, skips, scale_gates=None, use_gating=True, severity_map=None):
        """
        Args:
            bottleneck: (B, 384, H/16, W/16)
            skips: [e1, e2, e3, e4]
            scale_gates (Tensor, optional): (B, 4) gating values in [0, 1] [e2, e3, e4, bottleneck]
            use_gating (bool): Enable soft adaptive gating
            severity_map (Tensor, optional): (B, 4, H, W) predicted severity; enables the
                per-pixel gate branch in gate_mode='film'.
        """
        e1, e2, e3, e4 = skips

        raw_e2 = self.translator_e2(e2)
        raw_e3 = self.translator_e3(e3)
        raw_e4 = self.translator_e4(e4)
        raw_bn = self.translator_bn(bottleneck)

        if (not use_gating) or (scale_gates is None):
            # Ungated baseline path (Full translation)
            return raw_bn, [e1, raw_e2, raw_e3, raw_e4]

        origs = [e2, e3, e4, bottleneck]
        raws = [raw_e2, raw_e3, raw_e4, raw_bn]
        outs, effs = [], []
        for i, (orig, raw) in enumerate(zip(origs, raws)):
            if self.gate_mode == 'scalar':
                g = scale_gates[:, i].view(-1, 1, 1, 1)
                # Soft Adaptive Gating: gate * translated + (1 - gate) * original
                out = g * raw + (1.0 - g) * orig
            else:
                g = self.gate_mods[i](orig, scale_gates[:, i], scale_gates, severity_map)
                out = orig + g * (raw - orig)
            outs.append(out)
            if self.record_gates:
                effs.append(g.detach().float().mean(dim=tuple(range(1, g.dim()))))

        if self.record_gates:
            with torch.no_grad():
                rel = torch.stack([
                    (raw - orig).flatten(1).norm(dim=1) / (orig.flatten(1).norm(dim=1) + 1e-8)
                    for orig, raw in zip(origs, raws)], dim=1)
            self.last_gate_info = {'eff_gates': torch.stack(effs, dim=1), 'rel_delta': rel}

        return outs[3], [e1, outs[0], outs[1], outs[2]]
