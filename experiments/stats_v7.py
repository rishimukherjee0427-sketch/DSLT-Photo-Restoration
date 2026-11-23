"""
Canonical evaluation + paired statistics for the v7 (corrected VGG normalisation) DSLT A/B/C series.

    python experiments/stats_v7.py --seeds 123 2026

1. Evaluates results/v7_corrected/seed{S}/{A,B,C}_best.pth on the held-out test split (250 images ×
   mild/medium/severe, cached degradations, shared PSNR[cap 50 dB]/SSIM/LPIPS-alex) with per-image
   outputs -> results/v7_corrected/per_image/{cfg}_seed{S}.json (+ eval manifest). Cached: re-runs skip.
2. Statistics (no test-set model selection; checkpoints were chosen on val):
   - per config: mean ± SD across seeds of the per-seed test means (SD with ddof=1).
   - paired comparisons B−A, C−A, C−B per severity × metric. The unit of analysis is the TEST IMAGE:
     per image, the difference is averaged over seeds -> n=250 paired observations (seed-image pairs
     are NOT treated as independent). Wilcoxon signed-rank (two-sided) on those 250 differences.
   - Holm–Bonferroni correction over all comparisons × severities × metrics (27 tests), alpha = 0.05.
   - effect sizes: mean difference, Cohen's d_z (mean/SD of per-image differences), 95% CI from a
     two-level bootstrap (resample seeds, then images; 10k reps), and the per-seed mean differences.
3. Writes results/v7_corrected/{stats.json, results_table.md, stats_table.md} programmatically.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

CFGS = ("A", "B", "C")
CFG_NAMES = {"A": "DSLT-A (ungated)", "B": "DSLT-B (adaptive gating)",
             "C": "DSLT-C (gating + severity sup. + sparsity)"}
SEVS = ("mild", "medium", "severe")
METRICS = ("psnr", "ssim", "lpips")
HIGHER_BETTER = {"psnr": True, "ssim": True, "lpips": False}
COMPARISONS = (("B", "A"), ("C", "A"), ("C", "B"))
BASE = os.path.join(ROOT, "results", "v7_corrected")


def evaluate(seeds):
    import torch
    from experiments.ablation_loaders import load_ablation_model
    from utils.benchmark import evaluate_model, get_device
    from utils.manifest import write_manifest
    device = get_device()
    out_dir = os.path.join(BASE, "per_image"); os.makedirs(out_dir, exist_ok=True)
    for s in seeds:
        for c in CFGS:
            out = os.path.join(out_dir, f"{c}_seed{s}.json")
            if os.path.exists(out):
                continue
            ckpt_dir = os.path.join(BASE, f"seed{s}")
            fn, info = load_ablation_model(c, device, ckpt_dir=ckpt_dir)
            preds = []

            def capture(x, _fn=fn):  # keep raw predictions (uint8, call order = severity order, then image order)
                y = _fn(x)
                preds.append((y.detach().clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
                return y
            res = evaluate_model(capture, split="test", severities=SEVS, device=device, return_per_image=True)
            pred_dir = os.path.join(BASE, "predictions"); os.makedirs(pred_dir, exist_ok=True)
            arr = np.concatenate(preds)
            np.savez_compressed(os.path.join(pred_dir, f"{c}_seed{s}.npz"), images=arr,
                                names=np.array([n for sv in SEVS for n in res[sv]["per_image"]["name"]]),
                                severity=np.array([sv for sv in SEVS for _ in res[sv]["per_image"]["name"]]))
            assert len(arr) == sum(len(res[sv]["per_image"]["name"]) for sv in SEVS), "prediction count mismatch"
            with open(out + ".tmp", "w") as f:
                json.dump(dict(cfg=c, seed=s, info=info, results=res), f)
            os.replace(out + ".tmp", out)
            write_manifest(os.path.join(out_dir, f"{c}_seed{s}_eval_manifest.json"), kind="eval",
                           config=dict(cfg=c, seed=s, split="test", severities=SEVS, psnr_cap_db=50,
                                       ssim="skimage structural_similarity", lpips="alex"),
                           metrics={sv: {m: res[sv][m] for m in METRICS} for sv in SEVS},
                           checkpoint=os.path.join(ckpt_dir, f"{c}_best.pth"))
            print(f"[eval] {c} seed {s}: " + " | ".join(
                f"{sv} {res[sv]['psnr']:.2f}/{res[sv]['ssim']:.4f}/{res[sv]['lpips']:.4f}" for sv in SEVS), flush=True)
            del fn
            if device.type == "mps":
                torch.mps.empty_cache()


def load(seeds):
    data = {}  # data[cfg][sev][metric] -> array (n_seeds, n_images)
    ref_names = {}  # identical test images, in identical order, for every model and seed
    for c in CFGS:
        data[c] = {sv: {m: [] for m in METRICS} for sv in SEVS}
        for s in seeds:
            d = json.load(open(os.path.join(BASE, "per_image", f"{c}_seed{s}.json")))
            params = d["info"]["params"]
            for sv in SEVS:
                pi = d["results"][sv]["per_image"]
                ref_names.setdefault(sv, pi["name"])
                assert pi["name"] == ref_names[sv], f"image order mismatch ({c}, seed {s}, {sv})"
                for m in METRICS:
                    data[c][sv][m].append(np.asarray(pi[m], dtype=np.float64))
        for sv in SEVS:
            for m in METRICS:
                data[c][sv][m] = np.stack(data[c][sv][m])
        data[c]["_params"] = params
    return data


def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def two_level_bootstrap(diff, reps=10000, seed=0):
    """diff: (n_seeds, n_images). Resample seeds, then images, with replacement; return 95% CI of mean."""
    rng = np.random.default_rng(seed)
    S, N = diff.shape
    si = rng.integers(0, S, size=(reps, S))
    ii = rng.integers(0, N, size=(reps, N))
    means = np.empty(reps)
    for r in range(reps):
        means[r] = diff[si[r]][:, ii[r]].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(seeds):
    from scipy.stats import wilcoxon
    data = load(seeds)
    summary = {}
    for c in CFGS:
        summary[c] = {"params_M": data[c]["_params"] / 1e6}
        for sv in SEVS:
            for m in METRICS:
                per_seed = data[c][sv][m].mean(axis=1)
                summary[c][f"{sv}_{m}"] = dict(mean=float(per_seed.mean()),
                                               sd=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                                               per_seed={str(s): float(v) for s, v in zip(seeds, per_seed)})
    tests = []
    for a, b in COMPARISONS:
        for sv in SEVS:
            for m in METRICS:
                d = data[a][sv][m] - data[b][sv][m]          # (n_seeds, n_images)
                d_img = d.mean(axis=0)                        # seed-averaged per-image difference
                nz = d_img[np.abs(d_img) > 0]
                p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
                sd = d_img.std(ddof=1)
                lo, hi = two_level_bootstrap(d)
                tests.append(dict(comparison=f"{a}-{b}", severity=sv, metric=m, n_images=int(d.shape[1]),
                                  n_seeds=int(d.shape[0]), mean_diff=float(d_img.mean()),
                                  ci95=[lo, hi], cohens_dz=float(d_img.mean() / sd) if sd > 0 else 0.0,
                                  per_seed_mean_diff=[float(x) for x in d.mean(axis=1)],
                                  frac_images_improved=float(np.mean(d_img > 0) if HIGHER_BETTER[m] else np.mean(d_img < 0)),
                                  p_wilcoxon=p))
    adj = holm([t["p_wilcoxon"] for t in tests])
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)
        t["significant_holm_0.05"] = bool(pa < 0.05)
        better = (t["mean_diff"] > 0) == HIGHER_BETTER[t["metric"]]
        t["direction"] = "better" if better else "worse"
        t["seeds_agree_in_sign"] = bool(len(set(np.sign(t["per_seed_mean_diff"]))) == 1)
    out = dict(seeds=list(seeds), protocol=dict(
        training="800 steps (8x100), batch 4, crop 128, AdamW 2e-4, cosine, warmup 50, EMA 0.99, mixed severity, "
                 "FP32, vgg_normalize=True, train split only; best epoch selected on val (24 imgs x 3 sev)",
        evaluation="test split, 250 images x mild/medium/severe, cached degradations, PSNR capped at 50 dB, "
                   "SSIM skimage, LPIPS alex",
        statistics="per-image seed-averaged paired differences (n=250), two-sided Wilcoxon signed-rank, "
                   "Holm correction over 27 tests, two-level (seed, image) bootstrap 95% CI, Cohen's d_z"),
        summary=summary, tests=tests)
    with open(os.path.join(BASE, "stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_tables(out)
    return out


def write_tables(out):
    S, seeds = out["summary"], out["seeds"]
    fmt = {"psnr": "{:.2f}", "ssim": "{:.4f}", "lpips": "{:.4f}"}
    L = [f"**Test split (250 images), mean ± SD over {len(seeds)} seeds ({', '.join(map(str, seeds))}); "
         "corrected VGG normalisation; 800 training steps.** PSNR (dB) ↑ / SSIM ↑ / LPIPS ↓", "",
         "| Model | Params (M) | " + " | ".join(f"{sv.capitalize()} {m.upper()}" for sv in SEVS for m in METRICS) + " |",
         "|---|---:|" + "---:|" * 9]
    for c in CFGS:
        cells = []
        for sv in SEVS:
            for m in METRICS:
                r = S[c][f"{sv}_{m}"]
                cells.append(f"{fmt[m].format(r['mean'])} ± {fmt[m].format(r['sd'])}")
        L.append(f"| {CFG_NAMES[c]} | {S[c]['params_M']:.2f} | " + " | ".join(cells) + " |")
    open(os.path.join(BASE, "results_table.md"), "w").write("\n".join(L) + "\n")

    P = ["**Individual seeds (test split, 250 images).** PSNR (dB) / SSIM / LPIPS", "",
         "| Model | Seed | " + " | ".join(sv.capitalize() for sv in SEVS) + " |", "|---|---:|---|---|---|"]
    for c in CFGS:
        for s in seeds:
            cells = [" / ".join(fmt[m].format(S[c][f"{sv}_{m}"]["per_seed"][str(s)]) for m in METRICS) for sv in SEVS]
            P.append(f"| {CFG_NAMES[c]} | {s} | " + " | ".join(cells) + " |")
    open(os.path.join(BASE, "per_seed_table.md"), "w").write("\n".join(P) + "\n")

    T = ["**Paired comparisons on identical test images** (per-image differences averaged over seeds, n=250; "
         "two-sided Wilcoxon signed-rank; Holm correction over 27 tests; 95% CI from a seed×image bootstrap). "
         "Positive Δ = first model has higher metric value (for LPIPS, negative Δ is better).", "",
         "| Comparison | Severity | Metric | Δ mean | 95% CI | d_z | per-seed Δ | p (Wilcoxon) | p (Holm) | Sig. | Direction |",
         "|---|---|---|---:|---|---:|---|---:|---:|:---:|---|"]
    for t in out["tests"]:
        f = "{:+.3f}" if t["metric"] == "psnr" else "{:+.4f}"
        T.append(f"| {t['comparison']} | {t['severity']} | {t['metric'].upper()} | {f.format(t['mean_diff'])} | "
                 f"[{f.format(t['ci95'][0])}, {f.format(t['ci95'][1])}] | {t['cohens_dz']:+.2f} | "
                 f"{', '.join(f.format(x) for x in t['per_seed_mean_diff'])} | {t['p_wilcoxon']:.2g} | "
                 f"{t['p_holm']:.2g} | {'✓' if t['significant_holm_0.05'] else '–'} | {t['direction']} |")
    open(os.path.join(BASE, "stats_table.md"), "w").write("\n".join(T) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 2026])
    ap.add_argument("--skip_eval", action="store_true")
    a = ap.parse_args()
    if not a.skip_eval:
        evaluate(a.seeds)
    out = analyse(a.seeds)
    print(open(os.path.join(BASE, "results_table.md")).read())
    print(open(os.path.join(BASE, "stats_table.md")).read())


if __name__ == "__main__":
    main()
