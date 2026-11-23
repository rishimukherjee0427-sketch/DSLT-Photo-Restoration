"""
Benchmark-registry loaders for the gated-vs-ungated ablation (experiments/run_ablation.py).

    from experiments.eval_benchmark import register_model
    from experiments.ablation_loaders import register_all
    register_all(register_model)
    # then run_model('dslt_A_ungated', split='test') etc.

Loader contract (experiments/eval_benchmark.py): loader(device) -> (restore_fn, info_dict).
Each loads results/ablation/{A,B,C}_best.pth (EMA weights of the best-val-epoch).  The models were
trained ONLY on data/splits/train.txt, so contaminated=False.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from utils.benchmark import pad_to_multiple, count_params

ABLATION_DIR = os.path.join(ROOT, "results", "ablation")

CONFIG_NAMES = {
    "dslt_A_ungated": "A",
    "dslt_B_gated": "B",
    "dslt_C_gated_sup": "C",
}


def load_ablation_model(cfg, device, ckpt_dir=ABLATION_DIR):
    """cfg in 'A'|'B'|'C' -> (restore_fn, info). restore_fn: (B,3,H,W)[0,1] -> same shape."""
    from models.autoencoder import Encoder, Decoder
    from models.latent_translator import MultiScaleDSLT
    from models.damage_estimator import DamageEstimator

    path = os.path.join(ckpt_dir, f"{cfg}_best.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run experiments/run_ablation.py --config {cfg} first")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    gated = bool(ck["use_gating"])

    enc, dec = Encoder(), Decoder()
    dslt = MultiScaleDSLT(ck.get("gate_mode", "film"))
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"]); dslt.load_state_dict(ck["dslt"])
    mods = [enc, dec, dslt]
    de = None
    if gated:
        de = DamageEstimator(3, 32, gate_head=ck.get("gate_head", "rich"))
        de.load_state_dict(ck["damage_estimator"])
        mods.append(de)
    for m in mods:
        m.to(device).eval()

    @torch.no_grad()
    def fn(x):
        b, skips = enc(x)
        if gated:
            sev, g = de(x)
            bn_t, skips_t = dslt(b, skips, scale_gates=g, use_gating=True, severity_map=sev)
        else:
            bn_t, skips_t = dslt(b, skips)   # ungated path (damage estimator only used as aux loss in training)
        return torch.clamp(x + dec(bn_t, skips_t), 0.0, 1.0)

    # parameters actually used at inference (A: gate modulators are unused -> excluded)
    n_params = count_params(enc, dec, dslt) + (count_params(de) if gated else 0)
    if not gated:
        n_params -= count_params(dslt.gate_mods)
    note = (f"ablation config {cfg} ({ck.get('name')}): EMA weights of best-val epoch {ck.get('epoch')}"
            f" (val PSNR mean {ck.get('val_psnr_mean', float('nan')):.2f}); trained on data/splits/train.txt only; "
            f"gating={'film' if gated else 'off'}, w_gate_sup={ck.get('w_gate_sup')}, "
            f"w_gate_sparsity={ck.get('w_gate_sparsity')}")
    return pad_to_multiple(fn, 16), {"params": int(n_params), "contaminated": False, "note": note}


def _make_loader(cfg):
    def loader(device):
        return load_ablation_model(cfg, device)
    loader.__name__ = f"load_ablation_{cfg}"
    return loader


def register_all(register_model):
    """register_model(name, loader) for dslt_A_ungated / dslt_B_gated / dslt_C_gated_sup."""
    for name, cfg in CONFIG_NAMES.items():
        register_model(name, _make_loader(cfg))
