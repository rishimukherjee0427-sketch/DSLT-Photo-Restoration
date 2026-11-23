"""
Evaluate a registered model on the held-out benchmark and append to results/benchmark_results.json.

LOADER CONTRACT
  A loader is a callable  loader(device) -> restore_fn   (or -> (restore_fn, info_dict)).
  restore_fn(x): x is a float tensor (B,3,H,W) in [0,1] on `device` (degraded input);
  it returns a tensor of the SAME shape (restored image, nominally [0,1]; clipped by the
  harness). It must run under torch.no_grad() or be called inside it (the harness does).
  Optional info_dict keys: 'params' (int), 'contaminated' (bool: model saw val/test images
  during training), 'note' (str). Wrap fixed-stride nets with
  utils.benchmark.pad_to_multiple(fn, 16).

ADD A MODEL (from any module, before calling main):
    from experiments.eval_benchmark import register_model
    register_model('restormer', lambda device: (my_fn, {'params': n, 'contaminated': False}))
  or add it in this file's REGISTRY section. Then:
    python experiments/eval_benchmark.py --model restormer --split val

Usage:
    python experiments/eval_benchmark.py --list
    python experiments/eval_benchmark.py --model identity --model dslt_v5 --split val
    python experiments/eval_benchmark.py --table            # rebuild results/benchmark_table.{md,csv}

Results JSON layout:  {model: {split: {severity: {psnr, ssim, lpips, ...}}, "_meta": {...}}}
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")   # memory-lean; never 0.0
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import argparse
import csv
import io
import json
import time
import datetime

import torch

from utils.benchmark import (evaluate_model, get_device, identity_fn, bicubic_fn,
                             pad_to_multiple, count_params, SEVERITIES)

RESULTS_JSON = os.path.join(ROOT, "results", "benchmark_results.json")
TABLE_MD = os.path.join(ROOT, "results", "benchmark_table.md")
TABLE_CSV = os.path.join(ROOT, "results", "benchmark_table.csv")

REGISTRY = {}


def register_model(name, loader):
    """Register loader(device) -> restore_fn | (restore_fn, info_dict). See module docstring."""
    REGISTRY[name] = loader


# ------------------------------------------------------------------- built-in models
def _load_identity(device):
    return identity_fn, {"params": 0, "contaminated": False, "note": "no restoration (input = output)"}


def _load_bicubic(device):
    return bicubic_fn, {"params": 0, "contaminated": False, "note": "bicubic down/up x2 (interpolation only)"}


def _load_dslt_v5(device, ckpt="results/best_model_v5.pth"):
    """DSLT v5 (ungated MultiScaleDSLT, no damage estimator in this checkpoint):
    out = clamp(x + decoder(dslt(encoder(x))), 0, 1) exactly as experiments/evaluate.py infer_full."""
    from models.autoencoder import Encoder, Decoder
    from models.latent_translator import MultiScaleDSLT
    enc, dec, dslt = Encoder().to(device), Decoder().to(device), MultiScaleDSLT().to(device)
    c = torch.load(os.path.join(ROOT, ckpt), map_location=device, weights_only=True)
    enc.load_state_dict(c["encoder"]); dec.load_state_dict(c["decoder"]); dslt.load_state_dict(c["dslt"])
    enc.eval(); dec.eval(); dslt.eval()

    @torch.no_grad()
    def fn(x):
        b, skips = enc(x)
        bn_t, skips_t = dslt(b, skips)           # scale_gates=None -> ungated (as trained)
        return torch.clamp(x + dec(bn_t, skips_t), 0, 1)

    return pad_to_multiple(fn, 16), {
        "params": count_params(enc, dec, dslt),
        "contaminated": True,
        "note": f"{ckpt}: trained on ALL of data/train_clean (incl. val/test) -> NOT a held-out result",
    }


register_model("identity", _load_identity)
register_model("bicubic", _load_bicubic)
register_model("dslt_v5", _load_dslt_v5)


# ------------------------------------------------------------------------- results
def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_model(name, split="val", severities=SEVERITIES, n_images=None, results_path=RESULTS_JSON,
              batch_size=4, device=None, lpips_net="alex", full_res=False):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    device = device or get_device()
    t0 = time.time()
    out = REGISTRY[name](device)
    fn, info = out if isinstance(out, tuple) else (out, {})
    t_load = time.time() - t0
    res = evaluate_model(fn, split, severities, n_images, device, batch_size, lpips_net, full_res)
    db = _load_json(results_path)
    entry = db.setdefault(name, {})
    key = split + ("_full" if full_res else "")
    entry.setdefault(key, {}).update(res)
    meta = entry.setdefault("_meta", {})
    meta.update(info)
    meta.update({"device": str(device), "load_s": round(t_load, 2), "n_images": n_images or res[severities[0]]["n"],
                 "updated": datetime.datetime.now().isoformat(timespec="seconds"), "lpips_net": lpips_net})
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(db, f, indent=2)
    return res


def build_table(results_json=RESULTS_JSON, split="val", md_path=TABLE_MD, csv_path=TABLE_CSV,
                severities=SEVERITIES):
    """Rows = models, columns = {severity} x {PSNR, SSIM, LPIPS} (+ params). Writes .md and .csv,
    returns (markdown, csv_text). Models flagged contaminated are suffixed ' (CONTAMINATED)'."""
    db = _load_json(results_json) if isinstance(results_json, str) else results_json
    metrics = (("psnr", "PSNR", "%.2f"), ("ssim", "SSIM", "%.4f"), ("lpips", "LPIPS", "%.4f"))
    header = ["Model", "Params (M)"] + [f"{s} {m[1]}" for s in severities for m in metrics]
    rows = []
    for name, entry in db.items():
        if split not in entry:
            continue
        meta = entry.get("_meta", {})
        label = name + (" (CONTAMINATED)" if meta.get("contaminated") else "")
        p = meta.get("params")
        row = [label, "-" if p is None else "%.2f" % (p / 1e6)]
        for s in severities:
            r = entry[split].get(s)
            for k, _, fmt in metrics:
                row.append(fmt % r[k] if r and k in r else "-")
        rows.append(row)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md_text = f"Split: **{split}** (PSNR dB / SSIM / LPIPS-alex; PSNR per-image capped at 50 dB)\n\n" + "\n".join(md) + "\n"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    if md_path:
        with open(md_path, "w") as f:
            f.write(md_text)
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            f.write(buf.getvalue())
    return md_text, buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="registered model name (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--severities", nargs="+", default=list(SEVERITIES))
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--full-res", action="store_true")
    ap.add_argument("--results", default=RESULTS_JSON)
    ap.add_argument("--table", action="store_true", help="(re)build results/benchmark_table.{md,csv}")
    a = ap.parse_args()
    if a.list:
        print("registered:", ", ".join(sorted(REGISTRY)))
    for m in a.model:
        t0 = time.time()
        res = run_model(m, a.split, tuple(a.severities), a.n_images, a.results, a.batch_size,
                        lpips_net=a.lpips_net, full_res=a.full_res)
        print(f"[{m}] {a.split} ({time.time()-t0:.1f}s total)")
        for s, r in res.items():
            print(f"  {s:7s} PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}  LPIPS {r['lpips']:.4f}  "
                  f"(n={r['n']}, eval {r['time_s']}s)")
    if a.table or a.model:
        md, _ = build_table(a.results, a.split)
        print(md)


if __name__ == "__main__":
    main()
