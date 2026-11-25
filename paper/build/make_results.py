"""
Single source of truth for the paper: raw result files -> CSVs -> LaTeX tables + figures.

    python paper/build/make_results.py            # CPU only; never touches checkpoints or the GPU

Inputs (read-only; produced by the experiment scripts, never edited by hand):
  Stage 0  gating      results/v7_stage0/per_image/*.json (+ v7_corrected/per_image for A/B/C seeds 123, 2026),
                       results/v7_stage0/stats.json, results/v7_stage0/profiling.json
  Stage 1  inference   results/v7_translator_ablation/per_image_seed*.json, stats.json
  Stage 2  translators results/v8_translator_training/per_image/*.json, stats.json, profiling.json
  Stage 3  convergence results/v9_convergence_pilot/per_image/*.json, stats.json (seed 42) or
                       three_seed/stats.json (when seeds 123/2026 exist), profiling.json, seed*/ histories+manifests
Outputs (paper/build/out/):
  results.csv          one row per (stage, model, seed, severity, metric): mean over the 250 test images
  per_image.csv.gz     every per-image value (stage, model, seed, severity, image, psnr, ssim, lpips)
  stats.csv            every paired test exactly as stored in the stage stats.json files
  profiling.csv, training.csv, curves.csv
  tables/*.tex         booktabs tables for IEEEtran
  figures/*.pdf|png    300 dpi
Every number in the tables is formatted from these CSVs; nothing is typed by hand.
"""
import csv
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = lambda *p: os.path.join(ROOT, "results", *p)
OUT = os.path.join(ROOT, "paper", "build", "out")
SEVS = ("mild", "medium", "severe")
METRICS = ("psnr", "ssim", "lpips")
SEEDS3 = (42, 123, 2026)
FMT = {"psnr": "{:.2f}", "ssim": "{:.4f}", "lpips": "{:.4f}"}
DFMT = {"psnr": "{:+.3f}", "ssim": "{:+.4f}", "lpips": "{:+.4f}"}

STAGE_INFO = {  # descriptive metadata, mirrored from the run configs (verified in RESULTS_AUDIT.md)
    "S0": dict(name="gating", steps=800, crop=128, val_images=24),
    "S1": dict(name="translator intervention (inference only)", steps=800, crop=128, val_images=24),
    "S2": dict(name="translator ablation (trained)", steps=800, crop=128, val_images=24),
    "S3": dict(name="convergence", steps=20000, crop=256, val_images=250),
}
LABEL = {"A": "DSLT-A (ungated)", "B": "DSLT-B (gated)", "C": "DSLT-C (gated + sev. sup.)",
         "O": "DSLT-O (oracle, diagnostic)", "B_g1": "DSLT-B, gates forced to 1", "C_g1": "DSLT-C, gates forced to 1",
         "Tfull": "Full (all translators)", "Te2": "e2 translator only", "Tnone": "Encoder--decoder (no translators)"}


# ------------------------------------------------------------------ collection
def _load(p):
    with open(p) as f:
        return json.load(f)


def per_image_sources():
    """Yield (stage, model, seed, path, getter) for every per-image result file."""
    for m in ("A", "B", "C", "O", "B_g1", "C_g1"):
        for s in SEEDS3:
            p = R("v7_stage0", "per_image", f"{m}_seed{s}.json")
            if not os.path.exists(p) and m in ("A", "B", "C"):
                p = R("v7_corrected", "per_image", f"{m}_seed{s}.json")   # reused by Stage 0 (same pipeline)
            if os.path.exists(p):
                yield "S0", m, s, p, lambda d: d["results"]
    for s in SEEDS3:
        p = R("v7_translator_ablation", f"per_image_seed{s}.json")
        if os.path.exists(p):
            for v in ("full", "off_all", "off_e2", "off_e3", "off_e4", "off_bn", "resid_zero", "content_off"):
                yield "S1", v, s, p, (lambda d, v=v: {sv: dict(per_image=dict(name=d["names"][sv], **d["per_image"][v][sv]))
                                                       for sv in SEVS})
    for m in ("Tfull", "Te2", "Tnone"):
        for s in SEEDS3:
            p = R("v8_translator_training", "per_image", f"{m}_seed{s}.json")
            if os.path.exists(p):
                yield "S2", m, s, p, lambda d: d["results"]
            p = R("v9_convergence_pilot", "per_image", f"{m}_seed{s}.json")
            if os.path.exists(p):
                yield "S3", m, s, p, lambda d: d["results"]


def collect():
    rows, pis = [], []
    for stage, model, seed, path, get in per_image_sources():
        res = get(_load(path))
        names0 = None
        for sv in SEVS:
            pi = res[sv]["per_image"]
            assert len(pi["name"]) == 250, (path, sv)
            names0 = names0 or {}
            for m in METRICS:
                a = np.asarray(pi[m], np.float64)
                assert np.all(np.isfinite(a)), (path, sv, m)
                rows.append(dict(stage=stage, model=model, seed=seed, severity=sv, metric=m, mean=float(a.mean()),
                                 n_images=len(a), source=os.path.relpath(path, ROOT)))
            for i, n in enumerate(pi["name"]):
                pis.append((stage, model, seed, sv, n, pi["psnr"][i], pi["ssim"][i], pi["lpips"][i]))
    return rows, pis


def collect_stats():
    out = []
    s0 = _load(R("v7_stage0", "stats.json"))
    for fam in ("oracle_family", "counterfactual_family", "abc_3seed_family"):
        for t in s0[fam]:
            out.append(dict(stage="S0", family=fam, comparison=t["comparison"], severity=t["severity"], metric=t["metric"],
                            mean_diff=t["mean_diff"], ci95_lo=t["ci95"][0], ci95_hi=t["ci95"][1], ci90_lo="", ci90_hi="",
                            cohens_dz=t["cohens_dz"], per_seed=";".join(f"{x:.6g}" for x in t["per_seed_mean_diff"]),
                            p_wilcoxon=t["p_wilcoxon"], p_holm=t["p_holm"], verdict="", n_seeds=3,
                            source="results/v7_stage0/stats.json"))
    s1 = _load(R("v7_translator_ablation", "stats.json"))
    for t in s1["tests"]:
        out.append(dict(stage="S1", family="translator_intervention", comparison=t.get("comparison", t.get("variant", "")),
                        severity=t["severity"], metric=t["metric"], mean_diff=t["mean_diff"],
                        ci95_lo=t["ci95"][0], ci95_hi=t["ci95"][1], ci90_lo="", ci90_hi="", cohens_dz=t.get("cohens_dz", ""),
                        per_seed=";".join(f"{x:.6g}" for x in t.get("per_seed_mean_diff", [])), p_wilcoxon=t["p_wilcoxon"],
                        p_holm=t["p_holm"], verdict="", n_seeds=3, source="results/v7_translator_ablation/stats.json"))
    s2 = _load(R("v8_translator_training", "stats.json"))
    for t in s2["tests"]:
        out.append(dict(stage="S2", family="preregistered_equivalence", comparison=t["comparison"], severity=t["severity"],
                        metric=t["metric"], mean_diff=t["mean_diff"], ci95_lo=t["ci95"][0], ci95_hi=t["ci95"][1],
                        ci90_lo=t["ci90"][0], ci90_hi=t["ci90"][1], cohens_dz=t["cohens_dz"],
                        per_seed=";".join(f"{x:.6g}" for x in t["per_seed_mean_diff"]), p_wilcoxon=t["p_wilcoxon"],
                        p_holm=t["p_holm"], verdict=t["verdict"], n_seeds=3, source="results/v8_translator_training/stats.json"))
    p3 = R("v9_convergence_pilot", "three_seed", "stats.json")
    if os.path.exists(p3):
        for t in _load(p3)["tests"]:
            out.append(dict(stage="S3", family="three_seed", comparison="Tnone-Tfull", severity=t["severity"], metric=t["metric"],
                            mean_diff=t["mean_diff"], ci95_lo=t["ci95_2level"][0], ci95_hi=t["ci95_2level"][1],
                            ci90_lo=t["ci90_2level"][0], ci90_hi=t["ci90_2level"][1], cohens_dz=t["cohens_dz"],
                            per_seed=";".join(f"{p['mean_diff']:.6g}" for p in t["per_seed"]), p_wilcoxon=t["p_wilcoxon"],
                            p_holm=t["p_holm"], verdict=t["verdict"], n_seeds=3,
                            source="results/v9_convergence_pilot/three_seed/stats.json"))
    for t in _load(R("v9_convergence_pilot", "stats.json"))["tests"]:
        out.append(dict(stage="S3", family="seed42_image_level", comparison="Tnone-Tfull", severity=t["severity"],
                        metric=t["metric"], mean_diff=t["mean_diff"], ci95_lo=t["ci95"][0], ci95_hi=t["ci95"][1],
                        ci90_lo=t["ci90"][0], ci90_hi=t["ci90"][1], cohens_dz=t["cohens_dz"], per_seed=f"{t['mean_diff']:.6g}",
                        p_wilcoxon=t["p_wilcoxon"], p_holm=t["p_holm"], verdict=t["verdict"], n_seeds=1,
                        source="results/v9_convergence_pilot/stats.json"))
    return out


def collect_profiling():
    out = []
    for stage, p, key in (("S0", R("v7_stage0", "profiling.json"), "model"), ("S2", R("v8_translator_training", "profiling.json"), "cfg"),
                          ("S3", R("v9_convergence_pilot", "profiling.json"), "cfg")):
        for r in _load(p):
            out.append(dict(stage=stage, model=r[key], params_M=r["params_M"], GMACs_128=r["GMACs_128"], GMACs_256=r["GMACs_256"],
                            latency_ms_256_b1=r["latency_ms_256_b1"], throughput_img_s_256_b4=r["throughput_img_s_256_b4"],
                            source=os.path.relpath(p, ROOT)))
    return out


def collect_training():
    tr, cv = [], []
    for d in sorted(glob.glob(R("v9_convergence_pilot", "seed*"))):
        seed = int(os.path.basename(d)[4:])
        for m in ("Tfull", "Tnone"):
            hp, mp = os.path.join(d, f"{m}_history.json"), os.path.join(d, f"{m}_manifest.json")
            if not (os.path.exists(hp) and os.path.exists(mp)):
                continue                                   # run not finished -> not reported
            h, man = _load(hp)["history"], _load(mp)
            if h[-1]["step"] != 20000:
                continue
            best = max(h, key=lambda x: x["val"]["psnr_mean"])
            tr.append(dict(stage="S3", model=m, seed=seed, steps=h[-1]["step"], best_epoch=best["epoch"],
                           best_val_psnr_mean=best["val"]["psnr_mean"],
                           train_s_per_step_median=float(np.median([x["train_s_per_step"] for x in h])),
                           train_minutes=man["extra"]["train_minutes"], peak_train_GB=man["extra"]["peak_mps_driver_bytes"] / 1e9,
                           chip=man["environment"].get("chip"), git=man["git"]["commit"][:7], source=os.path.relpath(mp, ROOT)))
            for x in h:
                cv.append(dict(model=m, seed=seed, epoch=x["epoch"], step=x["step"], train_loss=x["train"]["loss"],
                               **{f"val_{sv}_{k}": x["val"][sv][k] for sv in SEVS for k in ("psnr", "ssim")}))
    return tr, cv


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ tables
def agg(rows, stage, model, sv, m, seeds=None):
    v = [r["mean"] for r in rows if r["stage"] == stage and r["model"] == model and r["severity"] == sv and r["metric"] == m
         and (seeds is None or r["seed"] in seeds)]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None, len(v))


def cell(rows, stage, model, sv, m, seeds=None, sd=True):
    mu, s, n = agg(rows, stage, model, sv, m, seeds)
    txt = FMT[m].format(mu)
    if sd and s is not None and m == "psnr":
        txt += r"$\pm$" + "{:.2f}".format(s)
    return txt


def results_table(rows, stage, models, caption, label, seeds=None, path=None):
    n = agg(rows, stage, models[0], "mild", "psnr", seeds)[2]
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + f" ({n} seed{'s' if n > 1 else ''}; PSNR mean$\\pm$SD over seeds)" + "}",
         r"\label{" + label + "}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
         r" & \multicolumn{3}{c}{Mild} & \multicolumn{3}{c}{Medium} & \multicolumn{3}{c}{Severe} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         "Model & " + " & ".join([r"PSNR$\uparrow$", r"SSIM$\uparrow$", r"LPIPS$\downarrow$"] * 3) + r" \\", r"\midrule"]
    for mo in models:
        L.append(LABEL.get(mo, mo) + " & " + " & ".join(cell(rows, stage, mo, sv, m, seeds) for sv in SEVS for m in METRICS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(L) + "\n")


def stats_table(stats, stage, family, comparisons, caption, label, path):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrc" + ("c" if stage in ("S2", "S3") else "") + "rr" + ("l" if stage in ("S2", "S3") else "") + "}", r"\toprule",
         r"Comparison & Sev. & $\Delta$ & 95\% CI" + (r" & 90\% CI (TOST)" if stage in ("S2", "S3") else "") + r" & $d_z$ & $p_\mathrm{Holm}$" + (r" & Verdict" if stage in ("S2", "S3") else "") + r" \\",
         r"\midrule"]
    for t in stats:
        if t["stage"] != stage or t["family"] != family or t["comparison"] not in comparisons:
            continue
        f = DFMT[t["metric"]]
        L.append(f"{t['comparison']} ({t['metric'].upper()}) & {t['severity']} & {f.format(t['mean_diff'])} & "
                 f"[{f.format(t['ci95_lo'])}, {f.format(t['ci95_hi'])}]"
                 + (f" & [{f.format(t['ci90_lo'])}, {f.format(t['ci90_hi'])}]" if stage in ("S2", "S3") else "")
                 + f" & {t['cohens_dz']:+.2f} & {t['p_holm']:.2g}"
                 + (f" & {t['verdict']}" if stage in ("S2", "S3") else "") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def efficiency_table(prof, training, path):
    p = {r["model"]: r for r in prof if r["stage"] == "S3"}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Cost at $256^2$ (Stage~3 profiling, Apple M4 Pro, MPS, FP32; training figures are means over completed Stage~3 runs).}",
         r"\label{tab:efficiency}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Model & Params (M) & GMACs & Lat. (ms) & Thr. (img/s) & s/step & Train (min) & Peak (GB) \\", r"\midrule"]
    for m in ("Tfull", "Tnone"):
        t = [x for x in training if x["model"] == m]
        L.append(f"{LABEL[m]} & {p[m]['params_M']:.2f} & {p[m]['GMACs_256']:.1f} & {p[m]['latency_ms_256_b1']:.1f} & "
                 f"{p[m]['throughput_img_s_256_b4']:.1f} & {np.mean([x['train_s_per_step_median'] for x in t]):.3f} & "
                 f"{np.mean([x['train_minutes'] for x in t]):.0f} & {np.mean([x['peak_train_GB'] for x in t]):.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ figures
def fig_curves(curves, path_noext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = sorted({c["seed"] for c in curves})
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))          # IEEE double-column width
    for k, sv in enumerate(SEVS):
        for m, col in (("Tfull", "#1f77b4"), ("Tnone", "#d62728")):
            for s in seeds:
                c = [x for x in curves if x["model"] == m and x["seed"] == s]
                ax[k].plot([x["step"] / 1000 for x in c], [x[f"val_{sv}_psnr"] for x in c], color=col, lw=1,
                           alpha=1 if len(seeds) == 1 else 0.6, label=LABEL[m] if s == seeds[0] else None)
        ax[k].set_title(f"{sv.capitalize()} (val, 250 img)", fontsize=8); ax[k].set_xlabel("step (k)", fontsize=7)
        ax[k].tick_params(labelsize=6)
    ax[0].set_ylabel("PSNR (dB)", fontsize=7); ax[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", dpi=300)


def fig_severe_gap(curves, path_noext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for s in sorted({c["seed"] for c in curves}):
        f = [x for x in curves if x["model"] == "Tfull" and x["seed"] == s]
        n = [x for x in curves if x["model"] == "Tnone" and x["seed"] == s]
        if len(f) == len(n) == 20:
            ax.plot([x["step"] / 1000 for x in f], [a["val_severe_psnr"] - b["val_severe_psnr"] for a, b in zip(f, n)],
                    marker="o", ms=2, lw=1, label=f"seed {s}")
    ax.axhline(0, color="k", lw=0.6); ax.axhspan(-0.1, 0.1, color="0.9", zorder=0, label=r"$\pm$0.10 dB margin")
    ax.set_xlabel("step (k)", fontsize=7); ax.set_ylabel(r"$\Delta$ val PSNR, severe (dB)" + "\nfull $-$ enc-dec", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", dpi=300)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    rows, pis = collect()
    stats = collect_stats(); prof = collect_profiling(); training, curves = collect_training()
    write_csv(os.path.join(OUT, "results.csv"), rows); write_csv(os.path.join(OUT, "stats.csv"), stats)
    write_csv(os.path.join(OUT, "profiling.csv"), prof); write_csv(os.path.join(OUT, "training.csv"), training)
    write_csv(os.path.join(OUT, "curves.csv"), curves)
    with gzip.open(os.path.join(OUT, "per_image.csv.gz"), "wt", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "model", "seed", "severity", "image", "psnr", "ssim", "lpips"]); w.writerows(pis)

    T = lambda n: os.path.join(OUT, "tables", n)
    results_table(rows, "S0", ["A", "B", "C", "O"], "Stage~0: damage-conditioned gating, test set, 800 steps", "tab:gating",
                  SEEDS3, T("tab_gating.tex"))
    results_table(rows, "S2", ["Tfull", "Te2", "Tnone"], "Stage~2: independently trained translator ablation, test set, 800 steps",
                  "tab:translators", SEEDS3, T("tab_translators.tex"))
    s3_seeds = sorted({r["seed"] for r in rows if r["stage"] == "S3" and r["model"] == "Tfull"})
    results_table(rows, "S3", ["Tfull", "Tnone"], "Stage~3: convergence, test set, 20k steps at $256^2$" +
                  ("" if len(s3_seeds) > 1 else " (exploratory)"), "tab:convergence", None, T("tab_convergence.tex"))
    stats_table(stats, "S0", "oracle_family", ("O-A", "C-A", "O-C"), r"Stage~0 paired differences (Holm over 27 tests; 3 seeds)",
                "tab:gating_stats", T("tab_gating_stats.tex"))
    stats_table(stats, "S2", "preregistered_equivalence", ("Tnone-Tfull",),
                r"Stage~2: encoder--decoder $-$ full (pre-registered margins; Holm over 18; 3 seeds)", "tab:translator_stats",
                T("tab_translator_stats.tex"))
    fam3 = "three_seed" if any(t["family"] == "three_seed" for t in stats) else "seed42_image_level"
    stats_table(stats, "S3", fam3, ("Tnone-Tfull",),
                r"Stage~3: encoder--decoder $-$ full (" + ("3 seeds" if fam3 == "three_seed" else "one seed, image-level only") + ")",
                "tab:convergence_stats", T("tab_convergence_stats.tex"))
    efficiency_table(prof, training, T("tab_efficiency.tex"))
    if curves:
        fig_curves(curves, os.path.join(OUT, "figures", "fig_convergence_curves"))
        fig_severe_gap(curves, os.path.join(OUT, "figures", "fig_severe_gap"))
    print(f"results.csv: {len(rows)} rows | per_image: {len(pis)} | stats: {len(stats)} | profiling: {len(prof)} | "
          f"training runs: {len(training)} | Stage-3 seeds with test results: {s3_seeds} | stats family S3: {fam3}")


if __name__ == "__main__":
    sys.exit(main())
