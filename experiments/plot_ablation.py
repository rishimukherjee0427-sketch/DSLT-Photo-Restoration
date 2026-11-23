"""
CPU-only plotting / summary for the gated-vs-ungated ablation.
Reads results/ablation/{A,B,C}_history.json (written by experiments/run_ablation.py) and writes
  results/ablation/ablation_curves.png   val PSNR per severity (+ mean) vs epoch, A/B/C
  results/ablation/gate_stats.png        gate statistics of B and C vs epoch
  results/ablation/ablation_summary.json best epoch / final / delta-vs-A per config
Usage: python experiments/plot_ablation.py [--dir results/ablation]
"""
import os
import sys
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEVS = ["mild", "medium", "severe"]
GATE_NAMES = ["e2", "e3", "e4", "bn"]
COL = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868"}
LABEL = {"A": "A ungated", "B": "B gated", "C": "C gated + sup"}


def load(d):
    H = {}
    for c in "ABC":
        p = os.path.join(d, f"{c}_history.json")
        if os.path.exists(p):
            with open(p) as f:
                H[c] = json.load(f)["history"]
    return H


def curves(d, H):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharex=True)
    for ax, key, title in zip(axs, SEVS + ["mean"], ["mild", "medium", "severe", "mean of 3"]):
        for c, h in H.items():
            ep = [r["epoch"] for r in h]
            y = [r["val"]["psnr_mean"] if key == "mean" else r["val"][key]["psnr"] for r in h]
            ax.plot(ep, y, "-o", ms=3, color=COL[c], label=LABEL[c])
        ax.set_title(f"val PSNR ({title})"); ax.set_xlabel("epoch (200 steps each)"); ax.grid(alpha=.3)
        ax.set_ylabel("PSNR (dB)")
    axs[0].legend()
    fig.suptitle("Gated vs ungated DSLT: validation PSNR (EMA weights, 48 val images/severity)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "ablation_curves.png"), dpi=140); plt.close(fig)


def gate_stats(d, H):
    cs = [c for c in "BC" if c in H]
    if not cs:
        return
    fig, axs = plt.subplots(len(cs), 4, figsize=(18, 3.6 * len(cs)), squeeze=False)
    cols = plt.cm.tab10(np.arange(4))
    for r, c in enumerate(cs):
        h = [x for x in H[c] if x["gates"] != "n/a"]
        ep = [x["epoch"] for x in h]
        a = axs[r, 0]
        for k in range(4):
            m = np.array([x["gates"]["scalar_mean"][k] for x in h]); s = np.array([x["gates"]["scalar_std"][k] for x in h])
            a.plot(ep, m, color=cols[k], label=GATE_NAMES[k]); a.fill_between(ep, m - s, m + s, color=cols[k], alpha=.15)
        a.set_title(f"{c}: scalar gate mean +- std (val)"); a.set_ylim(0, 1)
        a = axs[r, 1]
        for k in range(4):
            a.plot(ep, [x["gates"]["eff_mean"][k] for x in h], color=cols[k], label=GATE_NAMES[k])
        a.set_title(f"{c}: effective (FiLM) gate mean"); a.set_ylim(0, 1)
        a = axs[r, 2]
        for k in range(4):
            a.plot(ep, [np.nan if x["gates"]["corr_scalar_vs_target"][k] is None else x["gates"]["corr_scalar_vs_target"][k]
                        for x in h], color=cols[k], label=GATE_NAMES[k])
        a.axhline(0, color="k", lw=.5); a.set_ylim(-1, 1); a.set_title(f"{c}: corr(scalar gate, GT-derived target)")
        a = axs[r, 3]
        last = h[-1]["gates"]["by_severity"]
        w = 0.25
        for i, s in enumerate(SEVS):
            a.bar(np.arange(4) + (i - 1) * w, last[s]["eff_mean"], w, label=s)
        a.set_xticks(range(4)); a.set_xticklabels(GATE_NAMES); a.set_ylim(0, 1)
        a.set_title(f"{c}: effective gate by severity (epoch {ep[-1]})"); a.legend(fontsize=8)
        for j in range(3):
            axs[r, j].set_xlabel("epoch"); axs[r, j].grid(alpha=.3)
        axs[r, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(d, "gate_stats.png"), dpi=140); plt.close(fig)


def summary(d, H):
    S = {}
    for c, h in H.items():
        b = max(h, key=lambda r: r["val"]["psnr_mean"])
        S[c] = dict(epochs_done=len(h), best_epoch=b["epoch"], best_val=b["val"],
                    last_epoch=h[-1]["epoch"], last_val=h[-1]["val"],
                    s_per_step=float(np.mean([r["train_s_per_step"] for r in h])),
                    last_gates=h[-1]["gates"])
    if "A" in S:
        for c in "BC":
            if c in S:
                S[c]["delta_vs_A_best_psnr_mean"] = S[c]["best_val"]["psnr_mean"] - S["A"]["best_val"]["psnr_mean"]
                S[c]["delta_vs_A_last_psnr_mean"] = S[c]["last_val"]["psnr_mean"] - S["A"]["last_val"]["psnr_mean"]
                S[c]["delta_vs_A_best_per_sev"] = {s: S[c]["best_val"][s]["psnr"] - S["A"]["best_val"][s]["psnr"] for s in SEVS}
    with open(os.path.join(d, "ablation_summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    for c, s in S.items():
        print(f"{c}: {s['epochs_done']} ep | best ep {s['best_epoch']} PSNR mean {s['best_val']['psnr_mean']:.3f} "
              f"| last {s['last_val']['psnr_mean']:.3f}" + (f" | d_vs_A(best) {s.get('delta_vs_A_best_psnr_mean', 0):+.3f}" if c != "A" else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "ablation"))
    a = ap.parse_args()
    H = load(a.dir)
    if not H:
        sys.exit(f"no *_history.json in {a.dir}")
    curves(a.dir, H); gate_stats(a.dir, H); summary(a.dir, H)
