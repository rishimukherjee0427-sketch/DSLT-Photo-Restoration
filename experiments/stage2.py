"""
Stage 2 (v8): independently trained translator ablation -- verification, evaluation, statistics, profiling.

    python experiments/stage2.py all      # integrity -> eval -> profile -> stats -> visual   (results/v8_translator_training/)

Pre-registered analysis: results/v8_translator_training/PREREGISTRATION.md (margins PSNR 0.10 dB, SSIM 0.0020,
LPIPS 0.0050; equivalence = 90% seed x image bootstrap CI inside (-d, +d); Holm over 18 difference tests).
"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from utils import benchmark as bm
from utils.manifest import write_manifest, sha256_file

OUT = os.path.join(ROOT, "results", "v8_translator_training")
SEEDS = (42, 123, 2026)
CFGS = ("Tfull", "Te2", "Tnone")
NAMES = {"Tfull": "DSLT-A full (all translators)", "Te2": "DSLT-e2 (e2 translator only)",
         "Tnone": "DSLT-EncoderDecoder (no translators)"}
SEVS = ("mild", "medium", "severe")
METRICS = ("psnr", "ssim", "lpips")
HIGHER = {"psnr": True, "ssim": True, "lpips": False}
MARGIN = {"psnr": 0.10, "ssim": 0.0020, "lpips": 0.0050}
F = {"psnr": "{:.2f}", "ssim": "{:.4f}", "lpips": "{:.4f}"}


def ck_path(cfg, seed):
    return os.path.join(OUT, f"seed{seed}", f"{cfg}_best.pth")


def build(cfg_or_ck):
    from models.autoencoder import Encoder, Decoder
    from models.latent_translator import MultiScaleDSLT
    from experiments.run_ablation import CONFIGS
    tr = tuple(CONFIGS[cfg_or_ck]["translators"]) if isinstance(cfg_or_ck, str) else tuple(cfg_or_ck["translators"])
    return Encoder(), Decoder(), MultiScaleDSLT("none", translators=tr)


def load(cfg, seed, device):
    ck = torch.load(ck_path(cfg, seed), map_location="cpu", weights_only=False)
    enc, dec, dslt = build(ck)
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"]); dslt.load_state_dict(ck["dslt"])
    for m in (enc, dec, dslt):
        m.to(device).eval()

    @torch.no_grad()
    def fn(x):
        b, sk = enc(x)
        bn, skt = dslt(b, sk)
        return torch.clamp(x + dec(bn, skt), 0, 1)
    return fn, ck, sum(p.numel() for m in (enc, dec, dslt) for p in m.parameters())


# ------------------------------------------------------------------ integrity
def integrity():
    from experiments.run_ablation import CONFIGS
    rows = []
    for c in CFGS:
        for s in SEEDS:
            p = ck_path(c, s)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            man = json.load(open(os.path.join(OUT, f"seed{s}", f"{c}_manifest.json")))
            sha = sha256_file(p)
            bad = sum(int((~torch.isfinite(t)).sum()) for k in ("encoder", "decoder", "dslt", "damage_estimator")
                      for t in ck[k].values() if torch.is_tensor(t) and t.is_floating_point())
            meta_ok = (ck.get("config") == c and ck.get("seed") == s and ck.get("vgg_normalize") is True
                       and ck.get("epochs") == 8 and ck.get("steps_per_epoch") == 100 and ck.get("crop") == 128
                       and tuple(ck.get("translators")) == tuple(CONFIGS[c]["translators"]) and not ck.get("use_gating")
                       and ck.get("gate_mode") == "none")
            ok = bad == 0 and meta_ok and sha == man["checkpoint"]["sha256"]
            rows.append(dict(cfg=c, seed=s, checkpoint=os.path.relpath(p, ROOT), sha256=sha,
                             sha_matches_manifest=sha == man["checkpoint"]["sha256"], nonfinite=bad, metadata_ok=meta_ok,
                             best_epoch=ck.get("epoch"), train_minutes=man["extra"]["train_minutes"],
                             peak_train_mps_driver_GB=man["extra"]["peak_mps_driver_bytes"] / 1e9, ok=ok))
            if not ok:
                json.dump(rows, open(os.path.join(OUT, "checkpoint_integrity.json"), "w"), indent=1)
                raise SystemExit(f"INTEGRITY FAILURE: {rows[-1]}")
    json.dump(rows, open(os.path.join(OUT, "checkpoint_integrity.json"), "w"), indent=1)
    print(f"integrity OK for {len(rows)} checkpoints")
    return rows


# ------------------------------------------------------------------ evaluation
def evaluate():
    device = bm.get_device()
    pdir = os.path.join(OUT, "per_image"); os.makedirs(pdir, exist_ok=True)
    expected = [(s, n, p) for s in SEVS for n, p in zip(*bm.cached_paths("test", s)[:2])]
    for c in CFGS:
        for s in SEEDS:
            out = os.path.join(pdir, f"{c}_seed{s}.json")
            if os.path.exists(out):
                continue
            fn, ck, n_params = load(c, s, device)
            ptr, preds = dict(i=0), []

            def wrapped(x):
                for j in range(x.shape[0]):          # every input must be the expected cached test image
                    _, _, pe = expected[ptr["i"] + j]
                    ref = torch.from_numpy(bm._read_rgb(pe)).permute(2, 0, 1).float() / 255.0
                    if not torch.equal(ref, x[j].float().cpu()):
                        raise SystemExit(f"ALIGNMENT FAILURE {c} seed {s} input {ptr['i'] + j}")
                ptr["i"] += x.shape[0]
                y = fn(x)
                preds.append((y.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
                return y
            t0 = time.time()
            res = bm.evaluate_model(wrapped, split="test", severities=SEVS, device=device, return_per_image=True)
            assert ptr["i"] == len(expected), "not every test input was checked"
            info = dict(cfg=c, seed=s, params=n_params, translators=ck["translators"], alignment_checked_inputs=ptr["i"])
            json.dump(dict(info=info, results=res), open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
            os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
            np.savez_compressed(os.path.join(OUT, "predictions", f"{c}_seed{s}.npz"), images=np.concatenate(preds),
                                names=np.array([n for _, n, _ in expected]), severity=np.array([v for v, _, _ in expected]))
            write_manifest(os.path.join(pdir, f"{c}_seed{s}_eval_manifest.json"), kind="eval",
                           config=dict(info=info, split="test", psnr_cap_db=50, lpips="alex"),
                           metrics={v: {m: res[v][m] for m in METRICS} for v in SEVS}, checkpoint=ck_path(c, s),
                           extra=dict(eval_s=time.time() - t0, alignment_checked_inputs=ptr["i"]))
            print(f"[eval] {c} seed {s}: " + " | ".join(f"{v} {res[v]['psnr']:.2f}/{res[v]['ssim']:.4f}/{res[v]['lpips']:.4f}"
                                                        for v in SEVS), flush=True)
            torch.mps.empty_cache()


# ------------------------------------------------------------------ profiling (one model per process)
def profile_one(c):
    from torch.utils.flop_counter import FlopCounterMode
    enc, dec, dslt = build(c)

    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.enc, s.dec, s.dslt = enc, dec, dslt

        def forward(s, x):
            b, sk = s.enc(x); bn, skt = s.dslt(b, sk)
            return torch.clamp(x + s.dec(bn, skt), 0, 1)
    m = M().eval()
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
        peak = 0
        with torch.no_grad():
            for _ in range(10):
                m(x)
            torch.mps.synchronize(); ts = []
            for _ in range(30):
                t = time.time(); m(x); torch.mps.synchronize(); ts.append(time.time() - t)
                peak = max(peak, torch.mps.driver_allocated_memory())
        r[f"latency_ms_{res}_b{bs}"] = 1000 * float(np.median(ts))
        r[f"throughput_img_s_{res}_b{bs}"] = bs / float(np.median(ts))
        r[f"peak_mps_driver_GB_{res}_b{bs}"] = peak / 1e9
    print(json.dumps(r))


def profile():
    rows = []
    for c in CFGS:   # fresh process per model -> clean peak-memory measurement
        out = subprocess.check_output([sys.executable, __file__, "profile_one", c], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        rows.append(json.loads(out))
    json.dump(rows, open(os.path.join(OUT, "profiling.json"), "w"), indent=1)
    return rows


# ------------------------------------------------------------------ statistics
def boot(d, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    S, N = d.shape
    si = rng.integers(0, S, (reps, S)); ii = rng.integers(0, N, (reps, N))
    means = np.array([d[si[r]][:, ii[r]].mean() for r in range(reps)])
    return {q: float(np.percentile(means, q)) for q in (2.5, 5, 95, 97.5)}


def stats(integ, prof):
    from scipy.stats import wilcoxon
    from experiments.stats_v7 import holm
    data, names = {}, {}
    for c in CFGS:
        data[c] = {v: {m: [] for m in METRICS} for v in SEVS}
        for s in SEEDS:
            r = json.load(open(os.path.join(OUT, "per_image", f"{c}_seed{s}.json")))["results"]
            for v in SEVS:
                pi = r[v]["per_image"]
                names.setdefault(v, pi["name"]); assert pi["name"] == names[v], "image order mismatch"
                for m in METRICS:
                    data[c][v][m].append(np.asarray(pi[m], np.float64))
        data[c] = {v: {m: np.stack(data[c][v][m]) for m in METRICS} for v in SEVS}
    summ = {c: {f"{v}_{m}": dict(mean=float(data[c][v][m].mean(1).mean()), sd=float(data[c][v][m].mean(1).std(ddof=1)),
                                 per_seed=[float(x) for x in data[c][v][m].mean(1)]) for v in SEVS for m in METRICS} for c in CFGS}
    tests = []
    for c in ("Te2", "Tnone"):
        for v in SEVS:
            for m in METRICS:
                d = data[c][v][m] - data["Tfull"][v][m]
                di = d.mean(0); nz = di[np.abs(di) > 0]
                q = boot(d); dl = MARGIN[m]
                equiv = (q[5] > -dl) and (q[95] < dl)
                noninf = (q[5] > -dl) if HIGHER[m] else (q[95] < dl)
                worse_beyond = (q[95] < -dl) if HIGHER[m] else (q[5] > dl)
                verdict = "equivalent" if equiv else ("non-inferior (not equivalent)" if noninf else
                                                      ("worse beyond margin" if worse_beyond else "inconclusive"))
                ps = [float(x) for x in d.mean(1)]
                tests.append(dict(comparison=f"{c}-Tfull", severity=v, metric=m, mean_diff=float(di.mean()),
                                  ci95=[q[2.5], q[97.5]], ci90=[q[5], q[95]], margin=dl, verdict=verdict,
                                  cohens_dz=float(di.mean() / di.std(ddof=1)) if di.std(ddof=1) > 0 else 0.0,
                                  per_seed_mean_diff=ps, seeds_agree=bool(len(set(np.sign(ps))) == 1),
                                  p_wilcoxon=float(wilcoxon(nz).pvalue) if len(nz) else 1.0))
    for t, pa in zip(tests, holm([t["p_wilcoxon"] for t in tests])):
        t["p_holm"] = float(pa); t["sig_holm"] = bool(pa < 0.05)
    json.dump(dict(seeds=SEEDS, margins=MARGIN, summary=summ, tests=tests, integrity=integ, profiling=prof),
              open(os.path.join(OUT, "stats.json"), "w"), indent=1)

    L = ["## Test results (250 images; mean ± SD over seeds 42, 123, 2026; independently trained, 800 steps)", "",
         "| Model | " + " | ".join(f"{v.capitalize()} {m.upper()}" for v in SEVS for m in METRICS) + " |", "|---|" + "---:|" * 9]
    for c in CFGS:
        L.append(f"| {NAMES[c]} | " + " | ".join(f"{F[m].format(summ[c][f'{v}_{m}']['mean'])} ± {F[m].format(summ[c][f'{v}_{m}']['sd'])}"
                                                for v in SEVS for m in METRICS) + " |")
    L += ["", "Per seed (PSNR / SSIM / LPIPS):", "", "| Model | Seed | Mild | Medium | Severe |", "|---|---:|---|---|---|"]
    for c in CFGS:
        for i, s in enumerate(SEEDS):
            L.append(f"| {NAMES[c]} | {s} | " + " | ".join(" / ".join(F[m].format(summ[c][f'{v}_{m}']['per_seed'][i]) for m in METRICS)
                                                         for v in SEVS) + " |")
    L += ["", "## Pre-registered equivalence + difference tests (variant − full; margins PSNR ±0.10 dB, SSIM ±0.0020, LPIPS ±0.0050)", "",
          "| Comparison | Severity | Metric | Δ mean | 90% CI (TOST) | 95% CI | d_z | per-seed Δ | p | p Holm (18) | Verdict |",
          "|---|---|---|---:|---|---|---:|---|---:|---:|---|"]
    for t in tests:
        f = "{:+.3f}" if t["metric"] == "psnr" else "{:+.4f}"
        L.append(f"| {NAMES[t['comparison'].split('-')[0]]} | {t['severity']} | {t['metric'].upper()} | {f.format(t['mean_diff'])} | "
                 f"[{f.format(t['ci90'][0])}, {f.format(t['ci90'][1])}] | [{f.format(t['ci95'][0])}, {f.format(t['ci95'][1])}] | "
                 f"{t['cohens_dz']:+.2f} | {', '.join(f.format(x) for x in t['per_seed_mean_diff'])} | {t['p_wilcoxon']:.2g} | "
                 f"{t['p_holm']:.2g}{' ✓' if t['sig_holm'] else ''} | **{t['verdict']}** |")
    L += ["", "## Cost (inference: MPS FP32, median of 30 after 10 warm-up, one model per process; training: from run manifests)", "",
          "| Model | Params (M) | GMACs @128² | GMACs @256² | Latency ms @128² b1 | Latency ms @256² b1 | Throughput img/s @256² b4 | Peak MPS GB @256² b4 | Train min (mean) | Peak train MPS GB (mean) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, r in zip(CFGS, prof):
        tm = np.mean([x["train_minutes"] for x in integ if x["cfg"] == c])
        pm = np.mean([x["peak_train_mps_driver_GB"] for x in integ if x["cfg"] == c])
        L.append(f"| {NAMES[c]} | {r['params_M']:.2f} | {r['GMACs_128']:.1f} | {r['GMACs_256']:.1f} | {r['latency_ms_128_b1']:.1f} | "
                 f"{r['latency_ms_256_b1']:.1f} | {r['throughput_img_s_256_b4']:.1f} | {r['peak_mps_driver_GB_256_b4']:.2f} | {tm:.1f} | {pm:.2f} |")
    open(os.path.join(OUT, "stage2_tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return data


def visual():
    """Same test images as Stage 1 (rule: 50th/95th percentile of the Stage-1 per-image PSNR drop), seed 42."""
    import cv2
    s1 = [json.load(open(os.path.join(ROOT, "results", "v7_translator_ablation", f"per_image_seed{s}.json"))) for s in SEEDS]
    rows = []
    for v in SEVS:
        drop = np.mean([np.array(r["per_image"]["full"][v]["psnr"]) - np.array(r["per_image"]["off_all"][v]["psnr"]) for r in s1], 0)
        order = np.argsort(drop)
        names, dps, cps = bm.cached_paths("test", v)
        for q in (0.50, 0.95):
            idx = int(order[int(round(q * (len(order) - 1)))])
            tiles = [bm._read_rgb(cps[idx]), bm._read_rgb(dps[idx])]
            for c in CFGS:
                z = np.load(os.path.join(OUT, "predictions", f"{c}_seed42.npz"))
                k = [i for i, (sv, n) in enumerate(zip(z["severity"], z["names"])) if sv == v and n == names[idx]]
                assert len(k) == 1
                tiles.append(z["images"][k[0]])
            row = np.ascontiguousarray(np.concatenate(tiles, axis=1)[..., ::-1])
            cv2.putText(row, f"{v} p{int(q*100)} img{idx}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            rows.append(row)
    header = np.zeros((24, rows[0].shape[1], 3), np.uint8)
    for j, t in enumerate(("clean", "degraded", "full (44.4M)", "e2-only (23.0M)", "enc-dec (22.5M)")):
        cv2.putText(header, t, (6 + 256 * j, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, "visual_comparison_seed42.png"), np.concatenate([header] + rows, 0))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "profile_one":
        profile_one(sys.argv[2])
    elif cmd == "all":
        integ = integrity(); evaluate(); prof = profile(); stats(integ, prof); visual()
