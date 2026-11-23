"""
Stage 1: translator-contribution ablation on trained DSLT-A checkpoints (inference-time interventions only).

    python experiments/translator_ablation.py          # evaluate + stats + figures -> results/v7_translator_ablation/

Translator (models/latent_translator.py, ScaleTranslator): T_k(z) = z + s_k * Blocks_k(z), with s_k a learned
scalar and Blocks_k a stack of DilatedResidualBlocks (each lrelu(x + CBAM(conv(x)))). DSLT-A uses the
ungated path: bottleneck -> T_bn, skips (e1, e2, e3, e4) -> (e1, T_e2, T_e3, T_e4).

Variants (same checkpoint, same encoder/decoder, ONLY the translator outputs are replaced):
  full        : original DSLT-A                                    (must reproduce stored per-image results)
  off_all     : T_k(z) = z at all four scales                      ("translators disabled")
  off_e2/e3/e4/bn : T_k(z) = z at that single scale only
  resid_zero  : s_k = 0 at all scales                              (identical to off_all by construction; verified)
  content_off : T_k(z) = (1 + s_k) z   (learned transformation removed, learned rescaling kept)
These are interventions on networks TRAINED WITH translators; they are not independently trained
architectural ablations (a network trained without translators could compensate).
Metrics: canonical benchmark functions (PSNR capped at 50 dB, skimage SSIM, LPIPS-alex) on the cached
250-image test split x mild/medium/severe. Seeds 42, 123, 2026 (corrected-v7 A checkpoints).
"""
import json
import os
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
from experiments.stage0 import ckpt_path, load_nets, paired_family, SEVS, METRICS, F

OUT = os.path.join(ROOT, "results", "v7_translator_ablation")
SEEDS = (42, 123, 2026)
SCALES = ("e2", "e3", "e4", "bn")
VARIANTS = ("full", "off_all", "off_e2", "off_e3", "off_e4", "off_bn", "resid_zero", "content_off")
NAMES = {"full": "DSLT-A (original)", "off_all": "All translators off (T=I)", "off_e2": "e2 translator off",
         "off_e3": "e3 translator off", "off_e4": "e4 translator off", "off_bn": "bottleneck translator off",
         "resid_zero": "Residual scale s=0 (all)", "content_off": "Learned content off, rescaling kept"}


def translate(dslt, b, sk, variant):
    e1, e2, e3, e4 = sk
    feats = {"e2": e2, "e3": e3, "e4": e4, "bn": b}
    mods = {"e2": dslt.translator_e2, "e3": dslt.translator_e3, "e4": dslt.translator_e4, "bn": dslt.translator_bn}
    out = {}
    for k in SCALES:
        z, T = feats[k], mods[k]
        if variant == "full":
            out[k] = T(z)
        elif variant in ("off_all", f"off_{k}"):
            out[k] = z
        elif variant == "resid_zero":
            out[k] = z + T.blocks(z) * torch.zeros_like(T.scale)
        elif variant == "content_off":
            out[k] = z * (1 + T.scale)
        else:
            out[k] = T(z)
    return out["bn"], [e1, out["e2"], out["e3"], out["e4"]]


@torch.no_grad()
def evaluate_seed(seed, device):
    path = ckpt_path("A", seed)
    enc, dec, dslt, _, ck = load_nets(path, device)
    assert ck["config"] == "A" and ck["seed"] == seed and not ck["use_gating"], "unexpected checkpoint"
    scales = {k: float(getattr(dslt, f"translator_{k}").scale) for k in SCALES}
    res = {v: {s: {m: [] for m in METRICS} for s in SEVS} for v in VARIANTS}
    names_all, rel, eq_resid = {}, {k: [] for k in SCALES}, 0.0
    vis = {}
    for sv in SEVS:
        names, dpaths, cpaths = bm.cached_paths("test", sv)
        names_all[sv] = names
        for i in range(0, len(dpaths), 4):
            x = torch.from_numpy(np.stack([bm._read_rgb(p) for p in dpaths[i:i + 4]])).permute(0, 3, 1, 2).float().div(255).to(device)
            c = torch.from_numpy(np.stack([bm._read_rgb(p) for p in cpaths[i:i + 4]])).permute(0, 3, 1, 2).float().div(255)
            b, sk = enc(x)
            for k, z in zip(SCALES, (sk[1], sk[2], sk[3], b)):        # relative size of the translator change
                T = getattr(dslt, f"translator_{k}")
                rel[k].extend(((T(z) - z).flatten(1).norm(dim=1) / z.flatten(1).norm(dim=1)).cpu().tolist())
            ys = {}
            for v in VARIANTS:
                bn_t, sk_t = translate(dslt, b, sk, v)
                ys[v] = torch.clamp(x + dec(bn_t, sk_t), 0, 1).float().cpu()
            eq_resid = max(eq_resid, float((ys["resid_zero"] - ys["off_all"]).abs().max()))
            cn = c.permute(0, 2, 3, 1).numpy()
            for v in VARIANTS:
                yn = ys[v].permute(0, 2, 3, 1).numpy()
                for k in range(len(yn)):
                    res[v][sv]["psnr"].append(bm.psnr_np(yn[k], cn[k]))
                    res[v][sv]["ssim"].append(bm.ssim_np(yn[k], cn[k]))
                res[v][sv]["lpips"].extend(bm.lpips_batch(ys[v], c, "alex", device).tolist())
            if seed == 42:
                for k in range(x.shape[0]):
                    vis[(sv, i + k)] = (x[k].cpu(), c[k], ys["full"][k], ys["off_all"][k], ys["content_off"][k])
            del x, b, sk, ys
        torch.mps.empty_cache()
    # consistency: 'full' must reproduce the stored canonical per-image results of this checkpoint
    ref_p = os.path.join(ROOT, "results", "v7_stage0" if seed == 42 else "v7_corrected", "per_image", f"A_seed{seed}.json")
    ref = json.load(open(ref_p))["results"]
    repro = {sv: {m: float(np.max(np.abs(np.array(res["full"][sv][m]) - np.array(ref[sv]["per_image"][m]))))
                  for m in METRICS} for sv in SEVS}
    for sv in SEVS:
        assert ref[sv]["per_image"]["name"] == names_all[sv], "image order mismatch vs canonical results"
    out = dict(seed=seed, checkpoint=os.path.relpath(path, ROOT), checkpoint_sha256=sha256_file(path),
               learned_scale_s=scales, rel_translator_change_mean={k: float(np.mean(v)) for k, v in rel.items()},
               rel_translator_change_by_sev=None,
               max_abs_diff_resid_zero_vs_off_all=eq_resid, repro_max_abs_diff_vs_canonical=repro,
               names=names_all, per_image=res)
    return out, vis


def main():
    device = bm.get_device()
    os.makedirs(OUT, exist_ok=True)
    data = {v: {sv: {m: [] for m in METRICS} for sv in SEVS} for v in VARIANTS}
    diag, vis42 = {}, None
    t0 = time.time()
    for s in SEEDS:
        p = os.path.join(OUT, f"per_image_seed{s}.json")
        if os.path.exists(p) and s != 42:
            r = json.load(open(p))
        else:
            r, vis = evaluate_seed(s, device)
            json.dump(r, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
            if s == 42:
                vis42 = vis
            write_manifest(os.path.join(OUT, f"eval_manifest_seed{s}.json"), kind="eval",
                           config=dict(experiment="translator ablation (inference-time)", variants=VARIANTS, seed=s,
                                       split="test", psnr_cap_db=50, lpips="alex"),
                           metrics={v: {sv: {m: float(np.mean(r["per_image"][v][sv][m])) for m in METRICS} for sv in SEVS}
                                    for v in VARIANTS},
                           checkpoint=os.path.join(ROOT, r["checkpoint"]))
        for k in ("learned_scale_s", "rel_translator_change_mean", "max_abs_diff_resid_zero_vs_off_all",
                  "repro_max_abs_diff_vs_canonical"):
            diag.setdefault(str(s), {})[k] = r[k]
        repro_max = max(v for sv in r["repro_max_abs_diff_vs_canonical"].values() for v in sv.values())
        if repro_max > 1e-4:
            raise SystemExit(f"REPRODUCTION FAILURE seed {s}: 'full' differs from canonical results by {repro_max}")
        if r["max_abs_diff_resid_zero_vs_off_all"] > 1e-6:
            raise SystemExit(f"resid_zero != off_all for seed {s}: {r['max_abs_diff_resid_zero_vs_off_all']}")
        for v in VARIANTS:
            for sv in SEVS:
                for m in METRICS:
                    data[v][sv][m].append(np.asarray(r["per_image"][v][sv][m], np.float64))
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    data = {v: {sv: {m: np.stack(data[v][sv][m]) for m in METRICS} for sv in SEVS} for v in VARIANTS}

    tested = [v for v in VARIANTS if v not in ("full", "resid_zero")]   # resid_zero == off_all (verified)
    fam = paired_family([(v, "full") for v in tested], data, list(SEEDS))
    summ = {}
    for v in VARIANTS:
        summ[v] = {f"{sv}_{m}": dict(mean=float(data[v][sv][m].mean(1).mean()), sd=float(data[v][sv][m].mean(1).std(ddof=1)),
                                     per_seed=[float(x) for x in data[v][sv][m].mean(1)]) for sv in SEVS for m in METRICS}
    json.dump(dict(seeds=SEEDS, summary=summ, tests=fam, diagnostics=diag,
                   notes="inference-time interventions on DSLT-A trained with translators; Holm over all tests; "
                         "per-image seed-averaged differences (n=250), Wilcoxon, seed x image bootstrap CI"),
              open(os.path.join(OUT, "stats.json"), "w"), indent=1)

    L = ["## Translator contribution (inference-time interventions on DSLT-A; test split; mean ± SD over seeds 42, 123, 2026)", "",
         "| Variant | " + " | ".join(f"{sv.capitalize()} {m.upper()}" for sv in SEVS for m in METRICS) + " |", "|---|" + "---:|" * 9]
    for v in VARIANTS:
        L.append(f"| {NAMES[v]} | " + " | ".join(f"{F[m].format(summ[v][f'{sv}_{m}']['mean'])} ± {F[m].format(summ[v][f'{sv}_{m}']['sd'])}"
                                                for sv in SEVS for m in METRICS) + " |")
    L += ["", f"### Paired differences vs. original DSLT-A (variant − original; Holm over {len(fam)} tests)", "",
          "| Variant | Severity | Metric | Δ mean | 95% CI (seed×image bootstrap) | d_z | per-seed Δ | p | p Holm | Sig. | Seeds agree | CI excl. 0 | Direction |",
          "|---|---|---|---:|---|---:|---|---:|---:|:---:|:---:|:---:|---|"]
    for t in fam:
        f = "{:+.3f}" if t["metric"] == "psnr" else "{:+.4f}"
        L.append(f"| {NAMES[t['comparison'].split('-')[0]]} | {t['severity']} | {t['metric'].upper()} | {f.format(t['mean_diff'])} | "
                 f"[{f.format(t['ci95'][0])}, {f.format(t['ci95'][1])}] | {t['cohens_dz']:+.2f} | "
                 f"{', '.join(f.format(x) for x in t['per_seed_mean_diff'])} | {t['p_wilcoxon']:.2g} | {t['p_holm']:.2g} | "
                 f"{'✓' if t['sig_holm'] else '–'} | {'✓' if t['seeds_agree'] else '–'} | {'✓' if t['ci_excludes_0'] else '–'} | {t['direction']} |")
    L += ["", "### Diagnostics", "", "| Seed | learned s (e2, e3, e4, bn) | mean ‖T(z)−z‖/‖z‖ (e2, e3, e4, bn) | max \\|full − canonical\\| | max \\|s=0 − off_all\\| |",
          "|---:|---|---|---:|---:|"]
    for s in SEEDS:
        d = diag[str(s)]
        L.append(f"| {s} | {[round(d['learned_scale_s'][k], 3) for k in SCALES]} | "
                 f"{[round(d['rel_translator_change_mean'][k], 3) for k in SCALES]} | "
                 f"{max(v for sv in d['repro_max_abs_diff_vs_canonical'].values() for v in sv.values()):.2g} | "
                 f"{d['max_abs_diff_resid_zero_vs_off_all']:.2g} |")
    open(os.path.join(OUT, "translator_tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # visual comparisons (seed 42). Pre-specified rule: per severity, the test images at the 50th and 95th
    # percentile of the per-image PSNR drop (full − off_all, seed-averaged) -- not chosen by appearance.
    if vis42:
        import cv2
        rows = []
        for sv in SEVS:
            drop = (data["full"][sv]["psnr"] - data["off_all"][sv]["psnr"]).mean(0)
            order = np.argsort(drop)
            for q in (0.50, 0.95):
                idx = int(order[int(round(q * (len(order) - 1)))])
                x, c, yf, yo, yc = vis42[(sv, idx)]
                diff = ((yf - yo).abs() * 10).clamp(0, 1)
                tiles = [c, x, yf, yo, yc, diff]
                row = np.concatenate([(t.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for t in tiles], axis=1)
                lab = f"{sv} p{int(q*100)} img{idx} dPSNR(full-off)={drop[idx]:+.2f}dB"
                row = cv2.putText(np.ascontiguousarray(row[..., ::-1]), lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 255), 1, cv2.LINE_AA)
                rows.append(row)
        header = np.zeros((24, rows[0].shape[1], 3), np.uint8)
        for j, t in enumerate(("clean", "degraded", "DSLT-A", "translators off", "content off", "|A - off| x10")):
            cv2.putText(header, t, (6 + 256 * j, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, "visual_comparison_seed42.png"), np.concatenate([header] + rows, axis=0))


if __name__ == "__main__":
    main()
