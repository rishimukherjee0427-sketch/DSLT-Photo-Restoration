"""
Benchmark-registry loaders for the Restormer / PromptIR baselines (experiments/train_baseline.py).
Loads results/baselines/{model}_tiny_best.pth (EMA weights). Trained only on data/splits/train.txt.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from models.baselines import build_baseline, pad_to_multiple

SIZE = "tiny"


def _make_loader(name):
    def loader(device):
        path = os.path.join(ROOT, "results", "baselines", f"{name}_{SIZE}_best.pth")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = build_baseline(name, SIZE)
        model.load_state_dict(ck["model"])
        model.to(device).eval()

        @torch.no_grad()
        def restore_fn(x):
            xp, (h, w) = pad_to_multiple(x, 16)
            return model(xp)[..., :h, :w].clamp(0, 1)

        params = sum(p.numel() for p in model.parameters())
        return restore_fn, {"params": params, "contaminated": False,
                            "note": f"reduced-size re-implementation ({SIZE}), 800 steps"}
    return loader


def register_all(register_model):
    for name in ("restormer", "promptir"):
        register_model(name, _make_loader(name))
