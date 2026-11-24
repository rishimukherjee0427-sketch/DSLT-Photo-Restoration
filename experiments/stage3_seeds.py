"""
Stage 3 three-seed confirmation analysis: Tfull vs Tnone, seeds 42 / 123 / 2026, 256x256, 20 x 1000 steps each.

    python experiments/stage3_seeds.py      # integrity -> test eval (new seeds only) -> stats -> curves

Reuses the Stage-2/3 implementation unchanged:
  * evaluation  : experiments.stage2.evaluate (alignment-checked, per-image PSNR/SSIM/LPIPS, cached test degradations);
                  existing per-image files (seed 42) are skipped, never overwritten.
  * statistics  : Stage-2 pre-registered procedure -> two-level seed x image bootstrap (stage2.boot), 90% CI vs margins
                  (PSNR 0.10 dB, SSIM 0.0020, LPIPS 0.0050), Wilcoxon on seed-averaged per-image differences, Holm over
                  the 9 tests, d_z; plus the Stage-3 per-seed image-level tests (image bootstrap, Wilcoxon) per seed.
  * efficiency  : the existing Stage-3 profiling.json (architecture-only, seed-independent) is reused, not re-measured;
                  training time / peak memory come from each run's manifest.
Outputs go to results/v9_convergence_pilot/three_seed/ (seed-42 Stage-3 files are left untouched).
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
DST = os.path.join(OUT, "three_seed")
CFGS = ("Tfull", "Tnone")
SEEDS = (42, 123, 2026)
s2.OUT, s2.SEEDS, s2.CFGS = OUT, SEEDS, CFGS
SEVS, METRICS, MARGIN, HIGHER, F = s2.SEVS, s2.METRICS, s2.MARGIN, s2.HIGHER, s2.F
NAMES = {"Tfull": "DSLT-A full (44.4M)", "Tnone": "Encoder-decoder (22.5M)"}
# metadata keys that must be identical across every run (everything except the seed)
META_KEYS = ("use_gating", "gate_mode", "gate_head", "w_gate_sup", "w_gate_sparsity", "crop", "steps_per_epoch",
             "epochs", "uses_ema", "ema_decay", "vgg_normalize", "gate_mode_built")


def _finite(sd):
    bad, n = 0, 0
    for v in sd.values():
        if torch.is_tensor(v) and v.is_floating_point():
            bad += int((~torch.isfinite(v)).sum()); n += 1
        elif isinstance(v, dict):
            b, k = _finite(v); bad += b; n += k
    return bad, n


def _opt_finite(opt):
    bad, n = 0, 0
    for st in opt["state"].values():
        for v in st.values():
            if torch.is_tensor(v) and v.is_floating_point():
                bad += int((~torch.isfinite(v)).sum()); n += 1
    return bad, n


def integrity():
    from experiments.run_ablation import CONFIGS
    os.makedirs(DST, exist_ok=True)
    rows, ref = [], {}
    for c in CFGS:
        for s in SEEDS:
            d = os.path.join(OUT, f"seed{s}")
            bp, lp = os.path.join(d, f"{c}_best.pth"), os.path.join(d, f"{c}_last.pth")
            ck = torch.load(bp, map_location="cpu", weights_only=False)
            last = torch.load(lp, map_location="cpu", weights_only=False)
            man = json.load(open(os.path.join(d, f"{c}_manifest.json")))
            hist = json.load(open(os.path.join(d, f"{c}_history.json")))["history"]
            sha = sha256_file(bp)
            bad_best = sum(_finite(ck[k])[0] for k in ("encoder", "decoder", "dslt", "damage_estimator"))
            bad_raw, n_raw = _finite(last["nets"]); bad_ema, n_ema = _finite(last["ema"])
            bad_opt, n_opt = _opt_finite(last["opt"])
            opt_ok = bad_opt == 0 and n_opt > 0 and len(last["opt"]["state"]) > 0 \
                and all(int(st.get("step", 0)) == 20000 for st in last["opt"]["state"].values())
            meta = {k: ck.get(k) for k in META_KEYS}
            ref.setdefault(c, meta)
            meta_ok = (ck.get("config") == c and ck.get("seed") == s and last.get("seed") == s
                       and meta == ref[c] and meta["epochs"] == 20 and meta["steps_per_epoch"] == 1000
                       and meta["crop"] == 256 and meta["vgg_normalize"] is True and meta["gate_mode"] == "none"
                       and tuple(ck.get("translators")) == tuple(CONFIGS[c]["translators"]))
            best_hist = max(hist, key=lambda x: x["val"]["psnr_mean"])
            sel_ok = (best_hist["epoch"] == ck["epoch"] and abs(best_hist["val"]["psnr_mean"] - ck["val_psnr_mean"]) < 1e-9
                      and len(hist) == 20 and [h["epoch"] for h in hist] == list(range(1, 21)))
            complete = last["step"] == 20000 and last["epoch"] == 20 and hist[-1]["step"] == 20000
            ok = (bad_best == 0 and bad_raw == 0 and bad_ema == 0 and opt_ok and meta_ok and sel_ok and complete
                  and sha == man["checkpoint"]["sha256"])
            rows.append(dict(cfg=c, seed=s, sha256=sha, sha_matches_manifest=sha == man["checkpoint"]["sha256"],
                             nonfinite_best=bad_best, nonfinite_raw=bad_raw, raw_tensors=n_raw, nonfinite_ema=bad_ema,
                             ema_tensors=n_ema, nonfinite_opt=bad_opt, opt_tensors=n_opt, opt_state_ok=opt_ok,
                             metadata_ok=meta_ok, completed_steps=last["step"], completed_epochs=last["epoch"],
                             best_epoch=ck["epoch"], best_selected_by_val_only=sel_ok,
                             best_val_psnr_mean=ck["val_psnr_mean"], final_val_psnr_mean=hist[-1]["val"]["psnr_mean"],
                             train_minutes_wallclock=man["extra"]["train_minutes"],
                             peak_train_mps_driver_GB=man["extra"]["peak_mps_driver_bytes"] / 1e9,
                             git_commit=man.get("git", {}).get("commit") if isinstance(man.get("git"), dict) else man.get("git"),
                             ok=ok))
            json.dump(rows, open(os.path.join(DST, "checkpoint_integrity.json"), "w"), indent=1)
            if not ok:
                raise SystemExit(f"INTEGRITY / COMPLETENESS FAILURE: {rows[-1]}")
    print("integrity OK:", [(r["cfg"], r["seed"], r["completed_steps"], r["best_epoch"]) for r in rows])
    return rows


def eval_consistency():
    """Every per-image file: 250 images per severity, same image order, 750 alignment-checked inputs, right cfg/seed."""
    names, rows = None, []
    for c in CFGS:
        for s in SEEDS:
            d = json.load(open(os.path.join(OUT, "per_image", f"{c}_seed{s}.json")))
            info, r = d["info"], d["results"]
            nm = {v: r[v]["per_image"]["name"] for v in SEVS}
            names = names or nm
            ok = (info["cfg"] == c and info["seed"] == s and info["alignment_checked_inputs"] == 750 and nm == names
                  and all(len(r[v]["per_image"][m]) == 250 and np.all(np.isfinite(r[v]["per_image"][m]))
                          for v in SEVS for m in METRICS))
            rows.append(dict(cfg=c, seed=s, alignment_checked_inputs=info["alignment_checked_inputs"], ok=bool(ok)))
            if not ok:
                raise SystemExit(f"EVAL CONSISTENCY FAILURE: {c} seed {s}")
    json.dump(rows, open(os.path.join(DST, "eval_consistency.json"), "w"), indent=1)
    print("eval consistency OK")
    return rows


def img_boot(d, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return {q: float(np.percentile(m, q)) for q in (2.5, 5, 95, 97.5)}


def verdict(q, m):
    dl = MARGIN[m]
    if q[5] > -dl and q[95] < dl:
        return "equivalent"
    if (q[5] > -dl) if HIGHER[m] else (q[95] < dl):
        return "non-inferior (not equivalent)"
    if (q[95] < -dl) if HIGHER[m] else (q[5] > dl):
        return "worse beyond margin"
    return "inconclusive"


def crossover(hist_full, hist_none, key, sev):
    """Per epoch Tfull - Tnone val metric; 'sustained lead step' = first step from which Tfull stays ahead to step 20k."""
    d = [hf["val"][sev][key] - hn["val"][sev][key] for hf, hn in zip(hist_full, hist_none)]
    steps = [h["step"] for h in hist_full]
    lead = None
    for i in range(len(d)):
        if all(x > 0 for x in d[i:]):
            lead = steps[i]; break
    return dict(diff_per_epoch=[float(x) for x in d], steps=steps, sustained_tfull_lead_from_step=lead,
                tfull_ahead_at_start=bool(d[0] > 0), final_diff=float(d[-1]))


def stats(integ):
    from scipy.stats import wilcoxon, t as tdist
    from experiments.stats_v7 import holm
    data = {c: {v: {m: [] for m in METRICS} for v in SEVS} for c in CFGS}
    for c in CFGS:
        for s in SEEDS:
            r = json.load(open(os.path.join(OUT, "per_image", f"{c}_seed{s}.json")))["results"]
            for v in SEVS:
                for m in METRICS:
                    data[c][v][m].append(np.asarray(r[v]["per_image"][m], np.float64))
    data = {c: {v: {m: np.stack(data[c][v][m]) for m in METRICS} for v in SEVS} for c in CFGS}   # [seed, image]

    summ = {c: {f"{v}_{m}": dict(mean=float(data[c][v][m].mean(1).mean()), sd=float(data[c][v][m].mean(1).std(ddof=1)),
                                 per_seed=[float(x) for x in data[c][v][m].mean(1)]) for v in SEVS for m in METRICS}
            for c in CFGS}
    tests = []
    for v in SEVS:
        for m in METRICS:
            d = data["Tnone"][v][m] - data["Tfull"][v][m]        # [3, 250]  (Tnone - Tfull, as in Stage 2/3)
            di = d.mean(0); nz = di[np.abs(di) > 0]
            q = s2.boot(d)
            ps = d.mean(1); sd = float(ps.std(ddof=1))
            hw = float(tdist.ppf(0.975, len(ps) - 1) * sd / np.sqrt(len(ps)))
            per_seed = []
            for k, s in enumerate(SEEDS):
                qi = img_boot(d[k]); nzk = d[k][np.abs(d[k]) > 0]
                per_seed.append(dict(seed=s, mean_diff=float(d[k].mean()), ci95_img=[qi[2.5], qi[97.5]],
                                     ci90_img=[qi[5], qi[95]], verdict_img=verdict(qi, m),
                                     cohens_dz=float(d[k].mean() / d[k].std(ddof=1)),
                                     pct_images_tfull_better=float(100 * (np.mean(d[k] < 0) if HIGHER[m] else np.mean(d[k] > 0))),
                                     p_wilcoxon=float(wilcoxon(nzk).pvalue)))
            tests.append(dict(severity=v, metric=m, mean_diff=float(di.mean()), seed_sd=sd,
                              ci95_seed_t=[float(ps.mean() - hw), float(ps.mean() + hw)],
                              ci90_2level=[q[5], q[95]], ci95_2level=[q[2.5], q[97.5]], verdict=verdict(q, m),
                              cohens_dz=float(di.mean() / di.std(ddof=1)),
                              pct_images_tfull_better_seedavg=float(100 * (np.mean(di < 0) if HIGHER[m] else np.mean(di > 0))),
                              per_seed=per_seed, seeds_agree=bool(len(set(np.sign(ps))) == 1),
                              n_seeds_tfull_better=int(np.sum(ps < 0) if HIGHER[m] else np.sum(ps > 0)),
                              p_wilcoxon=float(wilcoxon(nz).pvalue)))
    for t, pa in zip(tests, holm([t["p_wilcoxon"] for t in tests])):
        t["p_holm"] = float(pa); t["sig_holm"] = bool(pa < 0.05)
    for k in range(len(SEEDS)):     # per-seed Holm over that seed's 9 image-level tests
        for t, pa in zip(tests, holm([t["per_seed"][k]["p_wilcoxon"] for t in tests])):
            t["per_seed"][k]["p_holm"] = float(pa)

    hist = {(c, s): json.load(open(os.path.join(OUT, f"seed{s}", f"{c}_history.json")))["history"] for c in CFGS for s in SEEDS}
    cross = {s: {f"{sev}_{k}": crossover(hist[("Tfull", s)], hist[("Tnone", s)], k, sev) for sev in SEVS for k in ("psnr", "ssim")}
             for s in SEEDS}
    prof = json.load(open(os.path.join(OUT, "profiling.json")))
    json.dump(dict(seeds=SEEDS, margins=MARGIN, direction="Tnone - Tfull", summary=summ, tests=tests, crossover=cross,
                   integrity=integ, profiling_reused_from="results/v9_convergence_pilot/profiling.json", profiling=prof),
              open(os.path.join(DST, "stats.json"), "w"), indent=1)

    # ---------------------------------------------------------------- tables
    fm = lambda m: "{:+.3f}" if m == "psnr" else "{:+.4f}"
    L = ["## Test results (250 images; mean ± SD over seeds 42, 123, 2026; 20k steps at 256², best-val checkpoint)", "",
         "| Model | " + " | ".join(f"{v.capitalize()} {m.upper()}" for v in SEVS for m in METRICS) + " |", "|---|" + "---:|" * 9]
    for c in CFGS:
        L.append(f"| {NAMES[c]} | " + " | ".join(F[m].format(summ[c][f'{v}_{m}']['mean']) + " ± " + F[m].format(summ[c][f'{v}_{m}']['sd'])
                                                 for v in SEVS for m in METRICS) + " |")
    L += ["", "Per seed (PSNR / SSIM / LPIPS):", "", "| Model | Seed | Mild | Medium | Severe |", "|---|---:|---|---|---|"]
    for c in CFGS:
        for i, s in enumerate(SEEDS):
            L.append(f"| {NAMES[c]} | {s} | " + " | ".join(" / ".join(F[m].format(summ[c][f'{v}_{m}']['per_seed'][i]) for m in METRICS)
                                                           for v in SEVS) + " |")
    L += ["", "## Encoder-decoder − full, three seeds (pre-registered Stage-2 procedure; Holm over 9)", "",
          "| Severity | Metric | Δ mean | per-seed Δ (42, 123, 2026) | seed SD | 95% CI (t, seeds) | 90% CI (2-level) | 95% CI (2-level) | d_z | % img Tfull better | p Holm | Verdict |",
          "|---|---|---:|---|---:|---|---|---|---:|---:|---:|---|"]
    for t in tests:
        f = fm(t["metric"])
        L.append(f"| {t['severity']} | {t['metric'].upper()} | {f.format(t['mean_diff'])} | "
                 f"{', '.join(f.format(p['mean_diff']) for p in t['per_seed'])}{' (agree)' if t['seeds_agree'] else ''} | "
                 f"{t['seed_sd']:.4f} | [{f.format(t['ci95_seed_t'][0])}, {f.format(t['ci95_seed_t'][1])}] | "
                 f"[{f.format(t['ci90_2level'][0])}, {f.format(t['ci90_2level'][1])}] | [{f.format(t['ci95_2level'][0])}, {f.format(t['ci95_2level'][1])}] | "
                 f"{t['cohens_dz']:+.2f} | {t['pct_images_tfull_better_seedavg']:.0f}% | {t['p_holm']:.2g}{' ✓' if t['sig_holm'] else ''} | **{t['verdict']}** |")
    L += ["", "## Per-seed image-level tests (each conditional on one training run; Holm over 9 within seed)", "",
          "| Severity | Metric | Seed | Δ | 95% CI (img) | d_z | % img Tfull better | p Holm | Verdict (img-level) |",
          "|---|---|---:|---:|---|---:|---:|---:|---|"]
    for t in tests:
        f = fm(t["metric"])
        for p in t["per_seed"]:
            L.append(f"| {t['severity']} | {t['metric'].upper()} | {p['seed']} | {f.format(p['mean_diff'])} | "
                     f"[{f.format(p['ci95_img'][0])}, {f.format(p['ci95_img'][1])}] | {p['cohens_dz']:+.2f} | "
                     f"{p['pct_images_tfull_better']:.0f}% | {p['p_holm']:.2g} | {p['verdict_img']} |")
    L += ["", "## Training (validation-only selection)", "",
          "| Model | Seed | Best epoch | Best val PSNR (mean 3 sev.) | Final val mild / medium / severe | Train s/step (median) | Wall-clock (min) | Peak train GB |",
          "|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in integ:
        h = hist[(r["cfg"], r["seed"])]
        L.append(f"| {NAMES[r['cfg']]} | {r['seed']} | {r['best_epoch']} | {r['best_val_psnr_mean']:.3f} | "
                 + " / ".join(f"{h[-1]['val'][v]['psnr']:.2f}" for v in SEVS)
                 + f" | {np.median([x['train_s_per_step'] for x in h]):.3f} | {r['train_minutes_wallclock']:.0f} | {r['peak_train_mps_driver_GB']:.2f} |")
    L += ["", "## Learning-curve crossover (validation, Tfull − Tnone)", "",
          "| Seed | Severe PSNR: Tfull ahead at step 1k? | Severe PSNR: sustained Tfull lead from step | Final Δ severe PSNR | Severe SSIM: sustained lead from | Final Δ severe SSIM | Final Δ mild PSNR | Final Δ medium PSNR |",
          "|---:|---|---|---:|---|---:|---:|---:|"]
    for s in SEEDS:
        x = cross[s]
        L.append(f"| {s} | {'yes' if x['severe_psnr']['tfull_ahead_at_start'] else 'no'} | {x['severe_psnr']['sustained_tfull_lead_from_step'] or 'never'} | "
                 f"{x['severe_psnr']['final_diff']:+.3f} | {x['severe_ssim']['sustained_tfull_lead_from_step'] or 'never'} | {x['severe_ssim']['final_diff']:+.4f} | "
                 f"{x['mild_psnr']['final_diff']:+.3f} | {x['medium_psnr']['final_diff']:+.3f} |")
    L += ["", "## Efficiency (inference: existing Stage-3 profiling.json; training: per-run manifests, mean over seeds)", "",
          "| Model | Params (M) | GMACs @256² | Latency ms @256² b1 | Throughput img/s @256² b4 | Train s/step | Train wall-clock (min) | Peak train GB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, r in zip(CFGS, prof):
        rs = [x for x in integ if x["cfg"] == c]
        sps = np.mean([np.median([h["train_s_per_step"] for h in hist[(c, x["seed"])]]) for x in rs])
        L.append(f"| {NAMES[c]} | {r['params_M']:.2f} | {r['GMACs_256']:.1f} | {r['latency_ms_256_b1']:.1f} | {r['throughput_img_s_256_b4']:.1f} | "
                 f"{sps:.3f} | {np.mean([x['train_minutes_wallclock'] for x in rs]):.0f} | {np.mean([x['peak_train_mps_driver_GB'] for x in rs]):.2f} |")
    open(os.path.join(DST, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return hist, cross


def curves(hist, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(SEEDS), 5, figsize=(24, 4 * len(SEEDS)))
    for i, s in enumerate(SEEDS):
        for c, col in zip(CFGS, ("#1f77b4", "#d62728")):
            h = hist[(c, s)]; st = [x["step"] for x in h]
            ax[i, 0].plot(st, [x["train"]["loss"] for x in h], color=col, label=NAMES[c])
            for k, v in enumerate(SEVS):
                ax[i, k + 1].plot(st, [x["val"][v]["psnr"] for x in h], color=col, marker="o", ms=3, label=NAMES[c])
        x = cross[s]["severe_psnr"]
        ax[i, 4].bar(x["steps"], x["diff_per_epoch"], width=700, color=["#1f77b4" if d > 0 else "#d62728" for d in x["diff_per_epoch"]])
        ax[i, 4].axhline(0, color="k", lw=0.8)
        if x["sustained_tfull_lead_from_step"]:
            ax[i, 4].axvline(x["sustained_tfull_lead_from_step"], color="k", ls=":", lw=1)
        ax[i, 0].set_ylabel(f"seed {s}")
        ax[i, 0].set_title("Train loss (epoch mean)")
        for k, v in enumerate(SEVS):
            ax[i, k + 1].set_title(f"Val PSNR {v} (dB)")
        ax[i, 4].set_title("Val severe PSNR: Tfull − Tnone (dB)")
        for a in ax[i]:
            a.set_xlabel("step")
    ax[0, 1].legend(fontsize=8)
    fig.suptitle("Stage 3 three-seed convergence (256², 20k steps); blue bar = Tfull ahead; dotted = sustained Tfull lead")
    fig.tight_layout(); fig.savefig(os.path.join(DST, "learning_curves_3seed.png"), dpi=110)


if __name__ == "__main__":
    integ = integrity()
    s2.evaluate()           # skips existing per-image files (seed 42); writes seeds 123 / 2026
    eval_consistency()
    hist, cross = stats(integ)
    curves(hist, cross)
