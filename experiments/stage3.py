"""
Stage 3 (v9) convergence pilot analysis: Tfull vs Tnone, seed 42, 256x256, 20 x 1000 steps.

    python experiments/stage3.py      # integrity -> test eval (per-image, alignment-checked) -> profile -> stats -> curves

One seed: all inference is IMAGE-LEVEL only (conditional on this training run). Same pre-registered margins as
Stage 2 (PSNR 0.10 dB, SSIM 0.0020, LPIPS 0.0050); 90% image-bootstrap CI for equivalence, Wilcoxon + Holm (9 tests).
"""
import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch

import experiments.stage2 as s2
from utils.manifest import sha256_file

OUT = os.path.join(ROOT, "results", "v9_convergence_pilot")
CFGS = ("Tfull", "Tnone")
SEED = 42
s2.OUT, s2.SEEDS, s2.CFGS = OUT, (SEED,), CFGS           # reuse Stage-2 evaluation / profiling code
SEVS, METRICS, MARGIN, HIGHER, F = s2.SEVS, s2.METRICS, s2.MARGIN, s2.HIGHER, s2.F
NAMES = {"Tfull": "DSLT-A full (44.4M)", "Tnone": "Encoder-decoder, no translators (22.5M)"}


def integrity():
    from experiments.run_ablation import CONFIGS
    rows = []
    for c in CFGS:
        p = s2.ck_path(c, SEED)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        man = json.load(open(os.path.join(OUT, f"seed{SEED}", f"{c}_manifest.json")))
        last = torch.load(os.path.join(OUT, f"seed{SEED}", f"{c}_last.pth"), map_location="cpu", weights_only=False)
        sha = sha256_file(p)
        bad = sum(int((~torch.isfinite(t)).sum()) for k in ("encoder", "decoder", "dslt", "damage_estimator")
                  for t in ck[k].values() if torch.is_tensor(t) and t.is_floating_point())
        meta_ok = (ck.get("config") == c and ck.get("seed") == SEED and ck.get("epochs") == 20
                   and ck.get("steps_per_epoch") == 1000 and ck.get("crop") == 256 and ck.get("vgg_normalize") is True
                   and tuple(ck.get("translators")) == tuple(CONFIGS[c]["translators"]) and ck.get("gate_mode") == "none")
        complete = last["step"] == 20000 and last["epoch"] == 20
        ok = bad == 0 and meta_ok and complete and sha == man["checkpoint"]["sha256"]
        rows.append(dict(cfg=c, sha256=sha, sha_matches_manifest=sha == man["checkpoint"]["sha256"], nonfinite=bad,
                         metadata_ok=meta_ok, completed_steps=last["step"], best_epoch=ck.get("epoch"),
                         best_val_psnr_mean=ck.get("val_psnr_mean"), train_minutes_wallclock=man["extra"]["train_minutes"],
                         peak_train_mps_driver_GB=man["extra"]["peak_mps_driver_bytes"] / 1e9, ok=ok))
        if not ok:
            json.dump(rows, open(os.path.join(OUT, "checkpoint_integrity.json"), "w"), indent=1)
            raise SystemExit(f"INTEGRITY / COMPLETENESS FAILURE: {rows[-1]}")
    json.dump(rows, open(os.path.join(OUT, "checkpoint_integrity.json"), "w"), indent=1)
    print("integrity OK:", [(r["cfg"], r["completed_steps"], r["best_epoch"]) for r in rows])
    return rows


def img_boot(d, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (reps, len(d)))
    m = d[idx].mean(1)
    return {q: float(np.percentile(m, q)) for q in (2.5, 5, 95, 97.5)}


def stats(integ, prof):
    from scipy.stats import wilcoxon
    from experiments.stats_v7 import holm
    data = {}
    for c in CFGS:
        r = json.load(open(os.path.join(OUT, "per_image", f"{c}_seed{SEED}.json")))["results"]
        data[c] = {v: {m: np.asarray(r[v]["per_image"][m], np.float64) for m in METRICS} for v in SEVS}
        data[c]["_names"] = {v: r[v]["per_image"]["name"] for v in SEVS}
    assert data["Tfull"]["_names"] == data["Tnone"]["_names"], "image order mismatch"
    tests = []
    for v in SEVS:
        for m in METRICS:
            d = data["Tnone"][v][m] - data["Tfull"][v][m]
            q = img_boot(d); dl = MARGIN[m]
            equiv = q[5] > -dl and q[95] < dl
            noninf = (q[5] > -dl) if HIGHER[m] else (q[95] < dl)
            worse_beyond = (q[95] < -dl) if HIGHER[m] else (q[5] > dl)
            verdict = "equivalent" if equiv else ("non-inferior (not equivalent)" if noninf else
                                                  ("worse beyond margin" if worse_beyond else "inconclusive"))
            nz = d[np.abs(d) > 0]
            tests.append(dict(severity=v, metric=m, mean_diff=float(d.mean()), ci90=[q[5], q[95]], ci95=[q[2.5], q[97.5]],
                              cohens_dz=float(d.mean() / d.std(ddof=1)), frac_images_better=float(np.mean(d > 0) if HIGHER[m] else np.mean(d < 0)),
                              p_wilcoxon=float(wilcoxon(nz).pvalue), verdict=verdict))
    for t, pa in zip(tests, holm([t["p_wilcoxon"] for t in tests])):
        t["p_holm"] = float(pa); t["sig_holm"] = bool(pa < 0.05)

    hist = {c: json.load(open(os.path.join(OUT, f"seed{SEED}", f"{c}_history.json")))["history"] for c in CFGS}
    json.dump(dict(seed=SEED, margins=MARGIN, means={c: {f"{v}_{m}": float(data[c][v][m].mean()) for v in SEVS for m in METRICS}
                                                     for c in CFGS},
                   tests=tests, integrity=integ, profiling=prof,
                   curves={c: [dict(epoch=h["epoch"], step=h["step"], train_loss=h["train"]["loss"],
                                    train_s_per_step=h["train_s_per_step"], val=h["val"]) for h in hist[c]] for c in CFGS}),
              open(os.path.join(OUT, "stats.json"), "w"), indent=1)

    L = ["## Test results (250 images, seed 42, 20k steps at 256², best-val checkpoint)", "",
         "| Model | " + " | ".join(f"{v.capitalize()} {m.upper()}" for v in SEVS for m in METRICS) + " |", "|---|" + "---:|" * 9]
    for c in CFGS:
        L.append(f"| {NAMES[c]} | " + " | ".join(F[m].format(data[c][v][m].mean()) for v in SEVS for m in METRICS) + " |")
    L += ["", "## Encoder-decoder − full: image-level paired tests (n=250; ONE training seed; Holm over 9; pre-registered margins)", "",
          "| Severity | Metric | Δ mean | 90% CI | 95% CI | d_z | % images better | p | p Holm | Verdict |",
          "|---|---|---:|---|---|---:|---:|---:|---:|---|"]
    for t in tests:
        f = "{:+.3f}" if t["metric"] == "psnr" else "{:+.4f}"
        L.append(f"| {t['severity']} | {t['metric'].upper()} | {f.format(t['mean_diff'])} | [{f.format(t['ci90'][0])}, {f.format(t['ci90'][1])}] | "
                 f"[{f.format(t['ci95'][0])}, {f.format(t['ci95'][1])}] | {t['cohens_dz']:+.2f} | {100*t['frac_images_better']:.0f}% | "
                 f"{t['p_wilcoxon']:.2g} | {t['p_holm']:.2g}{' ✓' if t['sig_holm'] else ''} | **{t['verdict']}** |")
    L += ["", "## Training", "", "| Model | Best epoch (of 20) | Best val PSNR (mean of 3 sev.) | Final-epoch val PSNR | Train s/step (median epoch) | Wall-clock train+val (min) | Peak train MPS GB |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for c, r in zip(CFGS, integ):
        h = hist[c]
        L.append(f"| {NAMES[c]} | {r['best_epoch']} | {r['best_val_psnr_mean']:.3f} | {h[-1]['val']['psnr_mean']:.3f} | "
                 f"{np.median([x['train_s_per_step'] for x in h]):.3f} | {r['train_minutes_wallclock']:.0f} | {r['peak_train_mps_driver_GB']:.2f} |")
    L += ["", "## Inference cost (MPS FP32, median of 30 after 10 warm-up, one model per process)", "",
          "| Model | Params (M) | GMACs @128² | GMACs @256² | Latency ms @256² b1 | Throughput img/s @256² b4 |", "|---|---:|---:|---:|---:|---:|"]
    for c, r in zip(CFGS, prof):
        L.append(f"| {NAMES[c]} | {r['params_M']:.2f} | {r['GMACs_128']:.1f} | {r['GMACs_256']:.1f} | {r['latency_ms_256_b1']:.1f} | "
                 f"{r['throughput_img_s_256_b4']:.1f} |")
    open(os.path.join(OUT, "stage3_tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return hist


def curves(hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for c, col in zip(CFGS, ("#1f77b4", "#d62728")):
        h = hist[c]; st = [x["step"] for x in h]
        ax[0].plot(st, [x["train"]["loss"] for x in h], color=col, label=NAMES[c])
        for k, v in enumerate(SEVS):
            ax[k + 1].plot(st, [x["val"][v]["psnr"] for x in h], color=col, marker="o", ms=3, label=NAMES[c])
            be = max(h, key=lambda x: x["val"]["psnr_mean"])
            ax[k + 1].axvline(be["step"], color=col, ls=":", lw=1)
    ax[0].set_title("Train loss (epoch mean)"); ax[0].set_xlabel("step")
    for k, v in enumerate(SEVS):
        ax[k + 1].set_title(f"Val PSNR, {v} (250 imgs)"); ax[k + 1].set_xlabel("step"); ax[k + 1].set_ylabel("dB")
    ax[1].legend(fontsize=8)
    fig.suptitle("Stage 3 convergence pilot (seed 42, 256², 20k steps); dotted = best-val epoch")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=120)


if __name__ == "__main__":
    integ = integrity()
    s2.evaluate()
    prof = s2.profile()
    hist = stats(integ, prof)
    curves(hist)
