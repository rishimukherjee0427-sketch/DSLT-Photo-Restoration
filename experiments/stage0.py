"""
Stage 0 go/no-go diagnostics for damage-conditioned gating (outputs: results/v7_stage0/).

    python experiments/stage0.py profile        # params, MACs, latency, throughput, peak memory
    python experiments/stage0.py eval           # test-set evaluation (per-image) for all Stage-0 jobs
    python experiments/stage0.py stats          # paired statistics + generated tables

Evaluation jobs (all on the cached 250-image test split x mild/medium/severe, same metrics as the
canonical benchmark: PSNR capped 50 dB, skimage SSIM, LPIPS-alex):
  A/B/C  seed 42       : new corrected-v7 runs (results/v7_stage0/seed42), normal inference
  A/B/C  seeds 123/2026: REUSED per-image results of results/v7_corrected/per_image (same pipeline)
  O      seeds 42/123/2026: oracle-conditioned model, ground-truth severity map as gate conditioning at
                        test time too (diagnostic upper bound on the value of damage information;
                        NOT a deployable result)
  B_g1 / C_g1 (all seeds): counterfactual intervention on trained B/C checkpoints, gates forced to 1
                        (use_gating=False path: output = T(e) at every scale). Diagnostic only; not a
                        substitute for independently trained ungated models.
Ground-truth test severity maps are regenerated from the same deterministic dataset the cache was
rendered from; correspondence with the cached degraded PNGs is verified (max |diff| <= 1/255).
"""
import argparse
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

BASE = os.path.join(ROOT, "results", "v7_stage0")
V7 = os.path.join(ROOT, "results", "v7_corrected")
SEEDS = (42, 123, 2026)
SEVS = ("mild", "medium", "severe")
METRICS = ("psnr", "ssim", "lpips")
HIGHER = {"psnr": True, "ssim": True, "lpips": False}


def ckpt_path(cfg, seed):
    if cfg in ("A", "B", "C") and seed in (123, 2026):
        return os.path.join(V7, f"seed{seed}", f"{cfg}_best.pth")
    return os.path.join(BASE, f"seed{seed}", f"{cfg}_best.pth")


def load_nets(path, device):
    from models.autoencoder import Encoder, Decoder
    from models.latent_translator import MultiScaleDSLT
    from models.damage_estimator import DamageEstimator
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc, dec, dslt = Encoder(), Decoder(), MultiScaleDSLT(ck.get("gate_mode", "film"))
    de = DamageEstimator(3, 32, gate_head=ck.get("gate_head", "rich"))
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
    dslt.load_state_dict(ck["dslt"]); de.load_state_dict(ck["damage_estimator"])
    for m in (enc, dec, dslt, de):
        m.to(device).eval()
    return enc, dec, dslt, de, ck


# ------------------------------------------------------------------ ground-truth test maps
_GT = {}


def gt_maps(sev):
    """(N,4,256,256) GT severity maps for the cached test images of `sev`, verified against the cache."""
    if sev in _GT:
        return _GT[sev]
    names, dpaths, _ = bm.cached_paths("test", sev)
    ds = bm.build_dataset("test", sev, 256, 0)
    maps, maxdiff = [], 0.0
    for i, p in enumerate(dpaths):
        deg, _, m = ds[i]
        cached = bm._read_rgb(p).astype(np.float32) / 255.0
        maxdiff = max(maxdiff, float(np.abs(deg.permute(1, 2, 0).numpy() - cached).max()))
        maps.append(m)
    assert maxdiff <= 1.0 / 255 + 1e-6, f"cache/dataset mismatch for {sev}: {maxdiff}"
    _GT[sev] = (torch.stack(maps), maxdiff, names)
    return _GT[sev]


# ------------------------------------------------------------------ evaluation
def run_job(tag, cfg, seed, mode, device):
    """mode: normal | gates1 | oracle. Writes per-image JSON, eval manifest, raw predictions."""
    from experiments.run_ablation import oracle_scale_gates
    out_dir = os.path.join(BASE, "per_image"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{tag}_seed{seed}.json")
    if os.path.exists(out):
        return
    path = ckpt_path(cfg, seed)
    enc, dec, dslt, de, ck = load_nets(path, device)
    gated = bool(ck["use_gating"])
    state = dict(sev_i=0, pos=0, checked=0)
    preds, gate_rec, diff_rec = [], {s: [] for s in SEVS}, {s: [] for s in SEVS}
    order = [(s, gt_maps(s)[0]) for s in SEVS] if mode == "oracle" else None
    # expected input sequence: cached test images, severity order then image order (alignment check)
    expected = [(s, n, p) for s in SEVS for n, p in zip(*bm.cached_paths("test", s)[:2])]
    k_ptr = dict(i=0)

    @torch.no_grad()
    def fn(x):
        i0 = k_ptr["i"]; k_ptr["i"] += x.shape[0]
        for j in range(x.shape[0]):                                  # every input == expected cached image
            sv_e, _, p_e = expected[i0 + j]
            ref = torch.from_numpy(bm._read_rgb(p_e)).permute(2, 0, 1).float() / 255.0
            if not torch.equal(ref, x[j].float().cpu()):
                raise SystemExit(f"ALIGNMENT FAILURE in {tag} seed {seed}: input {i0 + j} != {p_e}")
            if mode == "oracle" and sv_e != order[state["sev_i"]][0]:
                raise SystemExit(f"SEVERITY-MAP ALIGNMENT FAILURE in {tag} seed {seed} at input {i0 + j}")
            state["checked"] += 1
        b, sk = enc(x)
        if mode == "oracle":
            sev_name, maps = order[state["sev_i"]]
            m = maps[state["pos"]:state["pos"] + x.shape[0]].to(device)
            state["pos"] += x.shape[0]
            if state["pos"] >= len(maps):
                state["sev_i"] += 1; state["pos"] = 0
            g = oracle_scale_gates(de, x, m)
            bn, skt = dslt(b, sk, scale_gates=g, use_gating=True, severity_map=m)
        elif mode == "gates1" or not gated:
            bn, skt = dslt(b, sk)                                   # G == 1 (or ungated model)
        else:
            sm, g = de(x)
            bn, skt = dslt(b, sk, scale_gates=g, use_gating=True, severity_map=sm)
        y = torch.clamp(x + dec(bn, skt), 0, 1)
        if mode == "gates1":                                        # also run normal path: gate stats + output diff
            sev_name = SEVS[min(state["sev_i"], 2)]
            sm, g = de(x)
            dslt.record_gates = True
            bn2, sk2 = dslt(b, sk, scale_gates=g, use_gating=True, severity_map=sm)
            dslt.record_gates = False
            y2 = torch.clamp(x + dec(bn2, sk2), 0, 1)
            gate_rec[sev_name].append(dslt.last_gate_info["eff_gates"].float().cpu().numpy())
            diff_rec[sev_name].extend((y - y2).abs().mean(dim=(1, 2, 3)).float().cpu().tolist())
            state["pos"] += x.shape[0]
            if state["pos"] >= 250:
                state["sev_i"] += 1; state["pos"] = 0
        preds.append((y.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        return y

    t0 = time.time()
    res = bm.evaluate_model(fn, split="test", severities=SEVS, device=device, return_per_image=True)
    info = dict(cfg=cfg, mode=mode, seed=seed, checkpoint=os.path.relpath(path, ROOT),
                params=sum(p.numel() for m in ((enc, dec, dslt, de) if gated else (enc, dec, dslt))
                           for p in m.parameters()))
    extra = {}
    if mode == "gates1":
        extra["gate_eff_by_severity"] = {s: dict(mean=np.concatenate(gate_rec[s]).mean(0).tolist(),
                                                 std=np.concatenate(gate_rec[s]).std(0).tolist())
                                         for s in SEVS}
        extra["mean_abs_output_diff_vs_normal"] = {s: float(np.mean(diff_rec[s])) for s in SEVS}
    if mode == "oracle":
        extra["gt_cache_max_abs_diff"] = {s: gt_maps(s)[1] for s in SEVS}
    with open(out + ".tmp", "w") as f:
        json.dump(dict(cfg=cfg, tag=tag, seed=seed, info=info, extra=extra, results=res), f)
    os.replace(out + ".tmp", out)
    pd = os.path.join(BASE, "predictions"); os.makedirs(pd, exist_ok=True)
    if state["checked"] != len(expected):
        raise SystemExit(f"ALIGNMENT FAILURE in {tag} seed {seed}: {state['checked']} inputs checked, {len(expected)} expected")
    extra["alignment_checked_inputs"] = state["checked"]
    np.savez_compressed(os.path.join(pd, f"{tag}_seed{seed}.npz"), images=np.concatenate(preds),
                        names=np.array([n for _, n, _ in expected]), severity=np.array([s for s, _, _ in expected]))
    write_manifest(os.path.join(out_dir, f"{tag}_seed{seed}_eval_manifest.json"), kind="eval",
                   config=dict(info=info, split="test", severities=SEVS, psnr_cap_db=50, lpips="alex"),
                   metrics={s: {m: res[s][m] for m in METRICS} for s in SEVS}, checkpoint=path,
                   extra=dict(eval_s=time.time() - t0, **extra))
    print(f"[eval] {tag} seed {seed}: " + " | ".join(
        f"{s} {res[s]['psnr']:.2f}/{res[s]['ssim']:.4f}/{res[s]['lpips']:.4f}" for s in SEVS), flush=True)
    del enc, dec, dslt, de
    torch.mps.empty_cache()


def cmd_integrity():
    # integrity of every checkpoint used: SHA256 vs training manifest, finite weights, metadata
    integ = []
    for cfg in ("A", "B", "C", "O"):
        for s in SEEDS:
            p = ckpt_path(cfg, s)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            bad = sum(int((~torch.isfinite(t)).sum()) for k in ("encoder", "decoder", "dslt", "damage_estimator")
                      for t in ck[k].values() if torch.is_tensor(t) and t.is_floating_point())
            meta_ok = ck.get("vgg_normalize") is True and ck.get("seed") == s and ck.get("steps_per_epoch") == 100 \
                and ck.get("epochs") == 8 and ck.get("crop") == 128 and ck.get("config") == cfg
            sha = sha256_file(p)
            man_p = os.path.join(os.path.dirname(p), f"{cfg}_manifest.json")
            man_sha = (json.load(open(man_p)).get("checkpoint") or {}).get("sha256") if os.path.exists(man_p) else None
            ok = bad == 0 and meta_ok and sha == man_sha
            integ.append(dict(cfg=cfg, seed=s, checkpoint=os.path.relpath(p, ROOT), sha256=sha,
                              manifest_sha256=man_sha, sha_matches_manifest=sha == man_sha,
                              nonfinite=bad, metadata_ok=meta_ok, best_epoch=ck.get("epoch"), ok=ok))
            if not ok:
                json.dump(integ, open(os.path.join(BASE, "checkpoint_integrity.json"), "w"), indent=1)
                raise SystemExit(f"INTEGRITY FAILURE, evaluation stopped: {integ[-1]}")
    os.makedirs(BASE, exist_ok=True)
    json.dump(integ, open(os.path.join(BASE, "checkpoint_integrity.json"), "w"), indent=1)
    print("checkpoint integrity OK for", len(integ), "checkpoints")


def cmd_eval():
    device = bm.get_device()
    cmd_integrity()
    for c in ("A", "B", "C"):
        run_job(c, c, 42, "normal", device)
    for s in SEEDS:
        run_job("O", "O", s, "oracle", device)
    for s in SEEDS:
        for c in ("B", "C"):
            run_job(f"{c}_g1", c, s, "gates1", device)


# ------------------------------------------------------------------ statistics
def load_arr(tag, seeds):
    arr = {s: {m: [] for m in METRICS} for s in SEVS}
    names = None
    for seed in seeds:
        p = os.path.join(BASE, "per_image", f"{tag}_seed{seed}.json")
        if not os.path.exists(p):                          # reuse v7_corrected per-image results
            p = os.path.join(V7, "per_image", f"{tag}_seed{seed}.json")
        d = json.load(open(p))
        for sv in SEVS:
            pi = d["results"][sv]["per_image"]
            if names is None:
                names = {}
            names.setdefault(sv, pi["name"])
            assert pi["name"] == names[sv], "image order mismatch"
            for m in METRICS:
                arr[sv][m].append(np.asarray(pi[m], np.float64))
    return {sv: {m: np.stack(arr[sv][m]) for m in METRICS} for sv in SEVS}


def paired_family(pairs, data, seeds):
    from scipy.stats import wilcoxon
    from experiments.stats_v7 import holm, two_level_bootstrap
    tests = []
    for a, b in pairs:
        for sv in SEVS:
            for m in METRICS:
                d = data[a][sv][m] - data[b][sv][m]
                di = d.mean(0)
                nz = di[np.abs(di) > 0]
                p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
                sd = di.std(ddof=1)
                lo, hi = two_level_bootstrap(d)
                ps = [float(x) for x in d.mean(1)]
                better = (di.mean() > 0) == HIGHER[m]
                tests.append(dict(comparison=f"{a}-{b}", severity=sv, metric=m, mean_diff=float(di.mean()),
                                  ci95=[lo, hi], cohens_dz=float(di.mean() / sd) if sd > 0 else 0.0,
                                  per_seed_mean_diff=ps, p_wilcoxon=p,
                                  direction="better" if better else "worse",
                                  seeds_agree=bool(len(set(np.sign(ps))) == 1),
                                  ci_excludes_0=bool(lo > 0 or hi < 0)))
    for t, pa in zip(tests, holm([t["p_wilcoxon"] for t in tests])):
        t["p_holm"] = float(pa); t["sig_holm"] = bool(pa < 0.05)
    return tests


def summary(data, tags, seeds):
    S = {}
    for t in tags:
        S[t] = {}
        for sv in SEVS:
            for m in METRICS:
                ps = data[t][sv][m].mean(1)
                S[t][f"{sv}_{m}"] = dict(mean=float(ps.mean()), sd=float(ps.std(ddof=1)),
                                         per_seed={str(s): float(v) for s, v in zip(seeds, ps)})
    return S


NAMES = {"A": "DSLT-A (ungated)", "B": "DSLT-B (learned gating)", "C": "DSLT-C (learned gating + sev. sup.)",
         "O": "DSLT-O (ORACLE GT-severity conditioning; diagnostic)", "B_g1": "DSLT-B, gates forced to 1 (intervention)",
         "C_g1": "DSLT-C, gates forced to 1 (intervention)"}
F = {"psnr": "{:.2f}", "ssim": "{:.4f}", "lpips": "{:.4f}"}


def md_summary(S, tags, seeds, title):
    L = [title, "", "| Model | " + " | ".join(f"{sv.capitalize()} {m.upper()}" for sv in SEVS for m in METRICS) + " |",
         "|---|" + "---:|" * 9]
    for t in tags:
        L.append(f"| {NAMES[t]} | " + " | ".join(
            f"{F[m].format(S[t][f'{sv}_{m}']['mean'])} ± {F[m].format(S[t][f'{sv}_{m}']['sd'])}"
            for sv in SEVS for m in METRICS) + " |")
    L += ["", "Per seed (PSNR / SSIM / LPIPS):", "", "| Model | Seed | Mild | Medium | Severe |", "|---|---:|---|---|---|"]
    for t in tags:
        for s in seeds:
            L.append(f"| {NAMES[t]} | {s} | " + " | ".join(" / ".join(
                F[m].format(S[t][f"{sv}_{m}"]["per_seed"][str(s)]) for m in METRICS) for sv in SEVS) + " |")
    return "\n".join(L)


def md_tests(tests, title):
    L = [title, "", "| Comparison | Severity | Metric | Δ mean | 95% CI (seed×image bootstrap) | d_z | per-seed Δ | p | p Holm | Sig. | Seeds agree | CI excl. 0 | Direction |",
         "|---|---|---|---:|---|---:|---|---:|---:|:---:|:---:|:---:|---|"]
    for t in tests:
        f = "{:+.3f}" if t["metric"] == "psnr" else "{:+.4f}"
        L.append(f"| {t['comparison']} | {t['severity']} | {t['metric'].upper()} | {f.format(t['mean_diff'])} | "
                 f"[{f.format(t['ci95'][0])}, {f.format(t['ci95'][1])}] | {t['cohens_dz']:+.2f} | "
                 f"{', '.join(f.format(x) for x in t['per_seed_mean_diff'])} | {t['p_wilcoxon']:.2g} | {t['p_holm']:.2g} | "
                 f"{'✓' if t['sig_holm'] else '–'} | {'✓' if t['seeds_agree'] else '–'} | {'✓' if t['ci_excludes_0'] else '–'} | {t['direction']} |")
    return "\n".join(L)


def cmd_stats():
    seeds = list(SEEDS)
    tags = ("A", "B", "C", "O", "B_g1", "C_g1")
    data = {t: load_arr(t, seeds) for t in tags}
    S = summary(data, tags, seeds)
    fam_oracle = paired_family([("O", "A"), ("O", "C"), ("C", "A")], data, seeds)
    fam_cf = paired_family([("B", "B_g1"), ("C", "C_g1")], data, seeds)
    fam_abc = paired_family([("B", "A"), ("C", "A"), ("C", "B")], data, seeds)
    gates = {}
    for t in ("B_g1", "C_g1"):
        gates[t] = {}
        for s in seeds:
            d = json.load(open(os.path.join(BASE, "per_image", f"{t}_seed{s}.json")))
            gates[t][str(s)] = d["extra"]
    out = dict(seeds=seeds, summary=S, oracle_family=fam_oracle, counterfactual_family=fam_cf,
               abc_3seed_family=fam_abc, gate_stats=gates,
               notes=dict(statistics="per-image seed-averaged paired differences (n=250), two-sided Wilcoxon; "
                                     "Holm within each family (27 / 18 / 27 tests); two-level seed x image bootstrap",
                          oracle="ground-truth severity used as conditioning in training, validation AND test: "
                                 "upper bound on value of damage information, not deployable",
                          counterfactual="intervention on trained B/C weights (G forced to 1); not an independently "
                                         "trained ungated model"))
    json.dump(out, open(os.path.join(BASE, "stats.json"), "w"), indent=1)
    md = [md_summary(S, ("A", "C", "O"), seeds, "## Oracle conditioning: test split, mean ± SD over 3 seeds (42, 123, 2026)"),
          "", md_tests(fam_oracle, "### Oracle family (Holm over 27 tests)"), "",
          md_summary(S, ("B", "B_g1", "C", "C_g1"), seeds, "## Counterfactual gate intervention (diagnostic)"), "",
          md_tests(fam_cf, "### Counterfactual family (Holm over 18 tests)"), "",
          "### Effective gates (mean over images, scales e2/e3/e4/bn) and mean |Δoutput| when forcing G=1", "",
          "| Model | Seed | Severity | eff. gate mean | eff. gate SD | mean abs Δoutput |", "|---|---:|---|---|---|---:|"]
    for t in ("B_g1", "C_g1"):
        for s in seeds:
            e = gates[t][str(s)]
            for sv in SEVS:
                g = e["gate_eff_by_severity"][sv]
                md.append(f"| {NAMES[t[0]]} | {s} | {sv} | {np.round(g['mean'], 3).tolist()} | "
                          f"{np.round(g['std'], 3).tolist()} | {e['mean_abs_output_diff_vs_normal'][sv]:.5f} |")
    md += ["", md_summary(S, ("A", "B", "C"), seeds, "## DSLT-A/B/C, corrected v7 protocol, 3 seeds (42, 123, 2026)"), "",
           md_tests(fam_abc, "### A/B/C family (Holm over 27 tests)")]
    open(os.path.join(BASE, "stage0_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ profiling
def cmd_profile():
    from torch.utils.flop_counter import FlopCounterMode
    from models.autoencoder import Encoder, Decoder
    from models.latent_translator import MultiScaleDSLT
    from models.damage_estimator import DamageEstimator
    from models.baselines import build_baseline
    from experiments.run_ablation import oracle_scale_gates
    dev = torch.device("mps")

    def dslt_model(kind):
        enc, dec, dslt, de = Encoder(), Decoder(), MultiScaleDSLT(), DamageEstimator(3, 32)

        class M(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.enc, s.dec, s.dslt = enc, dec, dslt
                if kind != "A":
                    s.de = de

            def forward(s, x):
                b, sk = s.enc(x)
                if kind == "A":
                    bn, skt = s.dslt(b, sk)
                elif kind == "O":                     # oracle: GT map is an input (here: a dummy map)
                    m = torch.zeros(x.shape[0], 4, *x.shape[2:], device=x.device)
                    _ = s.de(x)                       # severity head still runs (as in training)
                    g = oracle_scale_gates(s.de, x, m)
                    bn, skt = s.dslt(b, sk, scale_gates=g, use_gating=True, severity_map=m)
                else:
                    sm, g = s.de(x)
                    bn, skt = s.dslt(b, sk, scale_gates=g, use_gating=True, severity_map=sm)
                return torch.clamp(x + s.dec(bn, skt), 0, 1)
        m = M()
        if kind == "A":
            n = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in dslt.gate_mods.parameters())
        else:
            n = sum(p.numel() for p in m.parameters())
        return m, n

    models = {}
    for k in ("A", "B", "C", "O"):
        models[f"DSLT-{k}"] = dslt_model(k)
    for name in ("restormer", "promptir"):
        m = build_baseline(name, "tiny")
        models[f"{name.capitalize()}-tiny"] = (m, sum(p.numel() for p in m.parameters()))
    rows = []
    for name, (m, n) in models.items():
        m.eval()
        r = dict(model=name, params_M=n / 1e6)
        for res in (128, 256):                      # MACs on CPU (FlopCounterMode counts matmul/conv; MACs = FLOPs/2)
            fc = FlopCounterMode(display=False)
            with torch.no_grad(), fc:
                m(torch.rand(1, 3, res, res))
            r[f"GMACs_{res}"] = fc.get_total_flops() / 2 / 1e9
        m.to(dev)
        for res, bs in ((128, 1), (256, 1), (256, 4)):
            x = torch.rand(bs, 3, res, res, device=dev)
            torch.mps.empty_cache(); torch.mps.synchronize()
            base = torch.mps.driver_allocated_memory(); peak = base
            with torch.no_grad():
                for _ in range(10):
                    m(x)
                torch.mps.synchronize()
                ts = []
                for _ in range(30):
                    t = time.time(); m(x); torch.mps.synchronize(); ts.append(time.time() - t)
                    peak = max(peak, torch.mps.driver_allocated_memory())
            key = f"{res}_b{bs}"
            r[f"latency_ms_{key}"] = 1000 * float(np.median(ts))
            r[f"throughput_img_s_{key}"] = bs / float(np.median(ts))
            r[f"peak_mps_driver_GB_{key}"] = peak / 1e9
            del x
        m.to("cpu"); torch.mps.empty_cache()
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    os.makedirs(BASE, exist_ok=True)
    json.dump(rows, open(os.path.join(BASE, "profiling.json"), "w"), indent=1)
    L = ["## Computational profile (Apple M4 GPU via MPS, FP32, inference, torch.no_grad; median of 30 runs after 10 warm-up)", "",
         "| Model | Params (M) | GMACs @128² | GMACs @256² | Latency ms @128² b1 | Latency ms @256² b1 | Throughput img/s @256² b4 | Peak MPS driver mem GB @256² b4 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['params_M']:.2f} | {r['GMACs_128']:.1f} | {r['GMACs_256']:.1f} | "
                 f"{r['latency_ms_128_b1']:.1f} | {r['latency_ms_256_b1']:.1f} | {r['throughput_img_s_256_b4']:.1f} | "
                 f"{r['peak_mps_driver_GB_256_b4']:.2f} |")
    L += ["", "Notes: MACs from torch.utils.flop_counter (convolutions/matmuls; element-wise ops, pooling and "
          "normalisation not counted). DSLT-A excludes the damage estimator and unused gate modulators (not executed "
          "at inference). DSLT-O = DSLT-C compute plus a second pass of the damage-estimator encoder for the oracle "
          "gate head (a dummy map stands in for the GT map). Peak memory = MPS driver allocation (includes the "
          "allocator cache; measured process-wide, models profiled sequentially)."]
    open(os.path.join(BASE, "profiling_table.md"), "w").write("\n".join(L) + "\n")
    write_manifest(os.path.join(BASE, "profiling_manifest.json"), kind="profile", metrics=dict(rows=rows))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["integrity", "profile", "eval", "stats"])
    a = ap.parse_args()
    {"integrity": cmd_integrity, "profile": cmd_profile, "eval": cmd_eval, "stats": cmd_stats}[a.cmd]()
