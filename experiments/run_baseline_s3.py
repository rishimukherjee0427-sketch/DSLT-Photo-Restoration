"""Task 3 baselines under the EXACT Stage-3 protocol.

Runs experiments/run_ablation.py::main() unchanged (same per-epoch data order and degradation RNG, AdamW 2e-4,
betas (0.9, 0.99), wd 1e-4, warmup 500 + cosine to 0.5%, grad clip 1.0, EMA 0.999, 20 x 1000 steps, crop 256,
batch 4, validation on 250 images per severity with the same PSNR/SSIM code, validation-best selection), with the
DSLT network replaced by a baseline. Only difference in the objective: the baselines have no damage estimator,
so the damage-map L1 term (weight 0.15, which never affects the restored image of an ungated DSLT) is absent;
the restoration loss (CombinedLoss mode='psnr') is identical.

    python experiments/run_baseline_s3.py --config nafnet --seed 42 --epochs 20 --steps_per_epoch 1000 --crop 256 \
        --batch_size 4 --warmup 500 --ema_decay 0.999 --val_n 250 --log_every 200 --vgg_normalize 1 --resume \
        --out_dir results/v10_baselines/seed42

Checkpoints: {cfg}_best.pth stores the whole baseline network under the key "encoder" (container convention
of run_ablation.py); load with load_baseline().
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import experiments.run_ablation as ra  # noqa: E402
from models.baselines.nafnet import build_nafnet  # noqa: E402
from models.baselines.unet import build_unet  # noqa: E402
from models.baselines.restormer import build_restormer  # noqa: E402

ARCHS = {
    "unet": (lambda: build_unet("b64"), "plain U-Net (base 64, bilinear up, no norm)"),
    "nafnet": (lambda: build_nafnet("w32"), "NAFNet-width32 (ECCV 2022 config), re-implemented"),
    "restormer": (lambda: build_restormer("base"), "Restormer (official size, 26.1M), re-implemented"),
}


class _NoDSLT(nn.Module):
    """Stand-in so run_ablation's bookkeeping (record_gates / state_dict) works unchanged."""
    def __init__(self):
        super().__init__()
        self.record_gates = False
        self.last_gate_info = None


class BaselineNets(nn.Module):
    """Same interface as run_ablation.Nets. restore() returns (restored, severity, gates):
    severity is None in training (-> no damage loss) and a zero tensor in eval (val bookkeeping only)."""

    def __init__(self, arch):
        super().__init__()
        self.encoder = ARCHS[arch][0]()          # the whole baseline network
        self.decoder = nn.Identity()
        self.damage_estimator = nn.Identity()
        self.dslt = _NoDSLT()

    def restore(self, x, use_gating, gates_out=None, oracle_sev=None):
        assert not use_gating and oracle_sev is None
        out = self.encoder(x)
        g = x.new_zeros(x.shape[0], 4)
        sev = None if self.training else x.new_zeros(x.shape[0], 4, 1, 1)
        return out, sev, g


def load_baseline(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ARCHS[ck["config"]][0]()
    net.load_state_dict(ck["encoder"])
    return net.to(device).eval(), ck


def main():
    arch = sys.argv[sys.argv.index("--config") + 1]
    for k, (_, desc) in ARCHS.items():
        ra.CONFIGS[k] = dict(name=f"baseline_{k}", use_gating=False, w_gate_sup=0.0, w_gate_sparsity=0.0,
                             translators=(), gate_mode="none", desc=desc)
    ra.Nets = lambda *a, **k: BaselineNets(arch)
    ra.main()


if __name__ == "__main__":
    main()
