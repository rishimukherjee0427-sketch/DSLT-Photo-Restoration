"""Task 3: test evaluation, profiling and paired statistics for the Stage-3-protocol baselines (seed 42).

    python experiments/eval_baselines.py eval      # per-image test metrics (same code + alignment check as Stage 2/3)
    python experiments/eval_baselines.py profile   # params, GMACs, latency, throughput (one process per model)
    python experiments/eval_baselines.py stats     # paired per-image tests vs Stage-3 Tfull/Tnone, seed 42 (CPU)

Only finished runs (manifest present, step 20000) are used. Outputs go to results/v10_baselines/ (never overwritten).
All statistics are image-level within one training run per model (seed 42 only): they show that these trained
models differ, not that the difference reproduces across seeds.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(ROOT, "results", "v10_baselines")
S3 = os.path.join(ROOT, "results", "v9_convergence_pilot")
CFGS = ("unet", "nafnet")
SEED = 42
SEVS = ("mild", "medium", "severe")
METRICS = ("psnr", "ssim", "lpips")
HIGHER = {"psnr": True, "ssim": True, "lpips": False}
MARGIN = {"psnr": 0.10, "ssim": 0.0020, "lpips": 0.0050}


def finished(c):
    d = os.path.join(OUT, f"seed{SEED}")
    m = os.path.join(d, f"{c}_manifest.json")
    if not os.path.exists(m):
        return False
    h = json.load(open(os.path.join(d, f"{c}_history.json")))["history"]
    return h[-1]["step"] == 20000


def evaluate():
    from utils import benchmark as bm
    from utils.manifest import write_manifest
    from experiments.run_baseline_s3 import load_baseline
    device = bm.get_device()
    pdir = os.path.join(OUT, "per_image"); os.makedirs(pdir, exist_ok=True)
    expected = [(s, n, p) for s in SEVS for n, p in zip(*bm.cached_paths("test", s)[:2])]
    for c in CFGS:
        out = os.path.join(pdir, f"{c}_seed{SEED}.json")
        if os.path.exists(out) or not finished(c):
            continue
        ckp = os.path.join(OUT, f"seed{SEED}", f"{c}_best.pth")
        net, ck = load_baseline(ckp, device)
        ptr = dict(i=0)

        @torch.no_grad()
        def wrapped(x):
            for j in range(x.shape[0]):
                _, _, pe = expected[ptr["i"] + j]
                ref = torch.from_numpy(bm._read_rgb(pe)).permute(2, 0, 1).float() / 255.0
                if not torch.equal(ref, x[j].float().cpu()):
                    raise SystemExit(f"ALIGNMENT FAILURE {c} input {ptr['i'] + j}")
            ptr["i"] += x.shape[0]
            return net(x)
        t0 = time.time()
        res = bm.evaluate_model(wrapped, split="test", severities=SEVS, device=device, return_per_image=True)
        assert ptr["i"] == len(expected)
        info = dict(cfg=c, seed=SEED, params=sum(p.numel() for p in net.parameters()), best_epoch=ck["epoch"],
                    alignment_checked_inputs=ptr["i"])
        json.dump(dict(info=info, results=res), open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
        write_manifest(out.replace(".json", "_eval_manifest.json"), kind="eval",
                       config=dict(info=info, split="test", psnr_cap_db=50, lpips="alex"),
                       metrics={v: {m: res[v][m] for m in METRICS} for v in SEVS}, checkpoint=ckp,
                       extra=dict(eval_s=time.time() - t0))
        print(f"[eval] {c}: " + " | ".join(f"{v} {res[v]['psnr']:.2f}/{res[v]['ssim']:.4f}/{res[v]['lpips']:.4f}"
                                           for v in SEVS), flush=True)
        del net
        torch.mps.empty_cache()


def profile_one(c):
    from torch.utils.flop_counter import FlopCounterMode
    from experiments.run_baseline_s3 import ARCHS
    m = ARCHS[c][0]().eval()
    r = dict(cfg=c, params_M=sum(p.numel() for p in m.parameters()) / 1e6)
    for res in (128, 256):
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            m(torch.rand(1, 3, res, res))
        r[f"GMACs_{res}"] = fc.get_total_flops() / 2e9
    dev = torch.device("mps"); m.to(dev)
    for res, bs in ((128, 1), (256, 1), (256, 4)):
        x = torch.rand(bs, 3, res, res, device=dev)
        torch.mps.empty_cache(); torch.mps.synchronize()
        with torch.no_grad():
            for _ in range(10):
                m(x)
            torch.mps.synchronize(); ts = []
            for _ in range(30):
                t = time.time(); m(x); torch.mps.synchronize(); ts.append(time.time() - t)
        r[f"latency_ms_{res}_b{bs}"] = 1000 * float(np.median(ts))
        r[f"throughput_img_s_{res}_b{bs}"] = bs / float(np.median(ts))
    print(json.dumps(r))


def profile():
    p = os.path.join(OUT, "profiling.json")
    if os.path.exists(p):
        raise SystemExit(f"{p} exists; refusing to overwrite")
    rows = []
    for c in CFGS:
        out = subprocess.check_output([sys.executable, __file__, "profile_one", c], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        rows.append(json.loads(out))
    json.dump(rows, open(p, "w"), indent=1)
    print(json.dumps(rows, indent=1))


def stats():
    from scipy.stats import wilcoxon
    from experiments.stats_v7 import holm
    load = lambda p: json.load(open(p))["results"]
    ref = {c: load(os.path.join(S3, "per_image", f"{c}_seed{SEED}.json")) for c in ("Tfull", "Tnone")}
    base = {c: load(os.path.join(OUT, "per_image", f"{c}_seed{SEED}.json")) for c in CFGS
            if os.path.exists(os.path.join(OUT, "per_image", f"{c}_seed{SEED}.json"))}
    rng = np.random.default_rng(0)
    tests = []
    for b, rb in base.items():
        for r, rr in ref.items():
            for v in SEVS:
                assert rb[v]["per_image"]["name"] == rr[v]["per_image"]["name"]
                for m in METRICS:
                    d = np.asarray(rb[v]["per_image"][m], float) - np.asarray(rr[v]["per_image"][m], float)
                    bt = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                    q = {k: float(np.percentile(bt, k)) for k in (2.5, 5, 95, 97.5)}
                    nz = d[np.abs(d) > 0]
                    better = (d > 0) if HIGHER[m] else (d < 0)
                    tests.append(dict(comparison=f"{b}-{r}", severity=v, metric=m, mean_diff=float(d.mean()),
                                      ci95=[q[2.5], q[97.5]], ci90=[q[5], q[95]],
                                      cohens_dz=float(d.mean() / d.std(ddof=1)) if d.std() > 0 else 0.0,
                                      pct_images_baseline_better=float(100 * better.mean()),
                                      p_wilcoxon=float(wilcoxon(nz).pvalue) if len(nz) else 1.0))
    for t, pa in zip(tests, holm([t["p_wilcoxon"] for t in tests])):
        t["p_holm"] = float(pa)
    p = os.path.join(OUT, "stats.json")
    json.dump(dict(seed=SEED, note="image-level, one training run per model; Holm over all tests in this file",
                   margins=MARGIN, tests=tests), open(p, "w"), indent=1)
    for t in tests:
        if t["metric"] == "psnr" or t["severity"] == "severe":
            print(f"{t['comparison']:14s} {t['severity']:6s} {t['metric']:5s} {t['mean_diff']:+.4f} "
                  f"[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.2g}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "profile_one":
        profile_one(sys.argv[2])
    else:
        dict(eval=evaluate, profile=profile, stats=stats)[cmd]()
