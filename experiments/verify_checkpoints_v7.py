"""
Integrity check for the v7 (corrected VGG) DSLT A/B/C checkpoints before evaluation.

    python experiments/verify_checkpoints_v7.py --seeds 123 2026

Per results/v7_corrected/seed{S}/{cfg}_best.pth: file exists, SHA256 equals the training manifest's
recorded hash, loads, has the expected sub-networks, every tensor is finite, and the stored metadata
(config, seed, vgg_normalize=True, steps, crop) matches the run. Exits non-zero on any failure.
Writes results/v7_corrected/checkpoint_integrity.json.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch

from utils.manifest import sha256_file

BASE = os.path.join(ROOT, "results", "v7_corrected")
KEYS = ("encoder", "decoder", "dslt", "damage_estimator")


def check(cfg, seed):
    d = os.path.join(BASE, f"seed{seed}")
    ck_path, man_path = os.path.join(d, f"{cfg}_best.pth"), os.path.join(d, f"{cfg}_manifest.json")
    r = dict(cfg=cfg, seed=seed, checkpoint=os.path.relpath(ck_path, ROOT), errors=[])
    if not os.path.exists(ck_path):
        r["errors"].append("checkpoint missing"); return r
    if not os.path.exists(man_path):
        r["errors"].append("training manifest missing")
    sha = sha256_file(ck_path); r["sha256"] = sha
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        if (man.get("checkpoint") or {}).get("sha256") != sha:
            r["errors"].append("sha256 differs from training manifest")
        r["best_epoch"] = man["metrics"].get("best_epoch")
        r["best_val_psnr_mean"] = man["metrics"].get("best_val_psnr_mean")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    for k in KEYS:
        if k not in ck:
            r["errors"].append(f"missing state dict '{k}'"); continue
        n_bad = sum(int((~torch.isfinite(t)).sum()) for t in ck[k].values() if torch.is_tensor(t) and t.is_floating_point())
        if n_bad:
            r["errors"].append(f"{n_bad} non-finite values in {k}")
    exp = dict(config=cfg, seed=seed, vgg_normalize=True, epochs=8, steps_per_epoch=100, crop=128)
    for k, v in exp.items():
        if ck.get(k) != v:
            r["errors"].append(f"metadata {k}={ck.get(k)!r}, expected {v!r}")
    r["n_params"] = ck.get("n_params"); r["ok"] = not r["errors"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 2026])
    a = ap.parse_args()
    rows = [check(c, s) for s in a.seeds for c in ("A", "B", "C")]
    json.dump(rows, open(os.path.join(BASE, "checkpoint_integrity.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['cfg']} seed {r['seed']}: {'OK' if r['ok'] else 'FAIL ' + '; '.join(r['errors'])} "
              f"| best epoch {r.get('best_epoch')} | val PSNR {r.get('best_val_psnr_mean')}")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
