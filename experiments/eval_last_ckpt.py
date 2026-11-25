"""Test-set evaluation of the FINAL-epoch (step 20k) EMA weights of a Stage-3 run, for comparison with
the validation-best checkpoint. Reporting only: model selection stays validation-best.

Writes results/v9_convergence_pilot/per_image/{cfg}_seed{seed}_last.json (+ eval manifest); refuses to
overwrite. Same metric code and alignment check as experiments/stage2.py::evaluate.
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402
from utils import benchmark as bm  # noqa: E402
from utils.manifest import write_manifest  # noqa: E402
import experiments.stage2 as s2  # noqa: E402

OUT = os.path.join(ROOT, "results", "v9_convergence_pilot")
SEVS, METRICS = s2.SEVS, s2.METRICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Tfull")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = os.path.join(OUT, "per_image", f"{a.config}_seed{a.seed}_last.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    lp = os.path.join(OUT, f"seed{a.seed}", f"{a.config}_last.pth")
    ck = torch.load(lp, map_location="cpu", weights_only=False)
    assert ck["config"] == a.config and ck["seed"] == a.seed and ck["step"] == ck["epochs"] * ck["steps_per_epoch"]
    ema = ck["ema"]
    enc, dec, dslt = s2.build(ck)
    for m, pre in ((enc, "encoder."), (dec, "decoder."), (dslt, "dslt.")):
        sd = {k[len(pre):]: v for k, v in ema.items() if k.startswith(pre)}
        m.load_state_dict(sd)
    device = bm.get_device()
    for m in (enc, dec, dslt):
        m.to(device).eval()

    @torch.no_grad()
    def fn(x):
        b, sk = enc(x)
        bn, skt = dslt(b, sk)
        return torch.clamp(x + dec(bn, skt), 0, 1)

    expected = [(s, n, p) for s in SEVS for n, p in zip(*bm.cached_paths("test", s)[:2])]
    ptr = dict(i=0)

    def wrapped(x):
        for j in range(x.shape[0]):
            _, _, pe = expected[ptr["i"] + j]
            ref = torch.from_numpy(bm._read_rgb(pe)).permute(2, 0, 1).float() / 255.0
            if not torch.equal(ref, x[j].float().cpu()):
                raise SystemExit(f"ALIGNMENT FAILURE input {ptr['i'] + j}")
        ptr["i"] += x.shape[0]
        return fn(x)

    t0 = time.time()
    res = bm.evaluate_model(wrapped, split="test", severities=SEVS, device=device, return_per_image=True)
    assert ptr["i"] == len(expected)
    n_params = sum(p.numel() for m in (enc, dec, dslt) for p in m.parameters())
    info = dict(cfg=a.config, seed=a.seed, params=n_params, translators=ck["translators"], weights="ema_last",
                epoch=ck["epoch"], step=ck["step"], alignment_checked_inputs=ptr["i"])
    json.dump(dict(info=info, results=res), open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    write_manifest(out.replace(".json", "_eval_manifest.json"), kind="eval",
                   config=dict(info=info, split="test", psnr_cap_db=50, lpips="alex"),
                   metrics={v: {m: res[v][m] for m in METRICS} for v in SEVS}, checkpoint=lp,
                   extra=dict(eval_s=time.time() - t0))
    print(f"[eval-last] {a.config} seed {a.seed} step {ck['step']}: " + " | ".join(
        f"{v} {res[v]['psnr']:.2f}/{res[v]['ssim']:.4f}/{res[v]['lpips']:.4f}" for v in SEVS), flush=True)


if __name__ == "__main__":
    main()
