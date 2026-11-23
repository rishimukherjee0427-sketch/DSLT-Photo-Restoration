"""
experiments/diagnose_gates.py
Empirical diagnosis of why adaptive damage-conditioned gating "does nothing"
(ablation: Config A ungated 25.71 dB vs Config B gated 25.77 dB, identical losses),
plus verification of the fixed design.

Usage (run from project root, PYTORCH_ENABLE_MPS_FALLBACK=1):
  python experiments/diagnose_gates.py diagnose --variant legacy   # unfixed Config B
  python experiments/diagnose_gates.py diagnose --variant fixed    # fixed Config B
  python experiments/diagnose_gates.py sanity   --variant legacy   # A vs B same seed
  python experiments/diagnose_gates.py sanity   --variant fixed
  python experiments/diagnose_gates.py compare                     # before/after table + plot

variant=legacy : DamageEstimator(gate_head='legacy'), MultiScaleDSLT('scalar'), no gate loss
variant=fixed  : DamageEstimator('rich'), MultiScaleDSLT('film'), CombinedLoss gate supervision
Outputs go to results/gate_diagnosis/.

Hypotheses tested (see summarize_hypotheses):
 H1 residual redundancy: ||raw-orig||/||orig|| per scale small; replacing predicted gates by the
    batch-constant mean (or removing gating) barely changes probe PSNR; gate std across images ~0.
 H2 nothing ties gates to damage: gate-vs-severity correlation ~0; severity (damage) loss gives
    exactly zero gradient to the gate head; gates stay near init (~0.5).
 H3 weak gate input: a ridge probe on the legacy gate input (avg-pooled e4) predicts GT severity
    poorly / 'mild' severity has tiny variance, vs richer input.
 H4 e1 skip never gated: structural (e1 has no translator, is passed through in both configs).
"""

import os
import sys
import json
import time
import random
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # memory cap (shared machine)
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.damage_estimator import DamageEstimator
from models.losses import CombinedLoss, severity_scalars, gate_targets_from_severity, GATE_MIX
from utils.dataloader import OnlineDegradationDataset

OUT_DIR = "results/gate_diagnosis"
CLEAN_DIR = "data/train_clean"
GATE_NAMES = ["e2", "e3", "e4", "bn"]
SEV_NAMES = ["blur", "noise", "scratch", "fade"]
N_HELDOUT = 100  # last N images are never used for training; probe comes from them


def _empty_cache():
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ───────────────────────── model / data setup ─────────────────────────

class Bundle:
    def __init__(self, variant, device, gating=True, use_gate_loss=None):
        self.variant = variant
        self.gating = gating
        fixed = variant == "fixed"
        self.de = DamageEstimator(3, 32, gate_head="rich" if fixed else "legacy").to(device)
        self.enc = Encoder().to(device)
        self.dslt = MultiScaleDSLT("film" if fixed else "scalar").to(device)
        self.dec = Decoder().to(device)
        use_gate_loss = fixed if use_gate_loss is None else use_gate_loss
        self.crit = CombinedLoss(
            device, mode="psnr",
            w_gate_sup=0.5 if use_gate_loss else 0.0,
            w_gate_sparsity=0.0, w_gate_bimodal=0.0)
        self.params = [p for m in (self.de, self.enc, self.dslt, self.dec) for p in m.parameters()]
        self.spatial = fixed  # pass predicted severity map to the film gate

    def train(self):
        for m in (self.de, self.enc, self.dslt, self.dec):
            m.train()

    def eval(self):
        for m in (self.de, self.enc, self.dslt, self.dec):
            m.eval()

    def forward(self, degraded, gate_override=None, gating=None):
        gating = self.gating if gating is None else gating
        sev, gates = self.de(degraded)
        if gate_override is not None:
            gates = gate_override(gates)
        b, sk = self.enc(degraded)
        bn_t, sk_t = self.dslt(b, sk, scale_gates=gates, use_gating=gating,
                               severity_map=sev.detach() if self.spatial else None)
        out = torch.clamp(degraded + self.dec(bn_t, sk_t), 0.0, 1.0)
        return out, sev, gates


def make_probe(device, n_per_mode, crop):
    """Deterministic held-out probe: n_per_mode mild + n_per_mode heavy samples."""
    xs, ys, ss, modes = [], [], [], []
    n_files = len(os.listdir(CLEAN_DIR))
    for mi, mode in enumerate(["mild", "heavy"]):
        ds = OnlineDegradationDataset(CLEAN_DIR, crop_size=crop, augment=False, severity=mode)
        for k in range(n_per_mode):
            idx = n_files - 1 - k  # last N_HELDOUT files
            d, c, s = ds[idx]
            xs.append(d); ys.append(c); ss.append(s); modes.append(mi)
    return (torch.stack(xs), torch.stack(ys), torch.stack(ss), np.array(modes))


def make_train_loader(batch, crop, seed):
    mild = OnlineDegradationDataset(CLEAN_DIR, crop_size=crop, augment=True, severity="mild")
    heavy = OnlineDegradationDataset(CLEAN_DIR, crop_size=crop, augment=True, severity="heavy")
    n_train = len(mild) - N_HELDOUT
    ds = ConcatDataset([Subset(mild, range(n_train)), Subset(heavy, range(n_train))])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0, generator=g)


def psnr_per_image(out, tgt):
    mse = ((out - tgt) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / (mse + 1e-8))


# ───────────────────────── measurements ─────────────────────────

def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sev_features(sev_gt):
    """(N,4,H,W) -> dict of per-image GT scalars: mean_c (4), max_c (4), overall, per-scale s."""
    mean_c = sev_gt.mean(dim=(2, 3)).numpy()
    max_c = sev_gt.flatten(2).amax(dim=2).numpy()
    sc = severity_scalars(sev_gt)
    mix = torch.tensor(GATE_MIX)
    s_scale = (sc @ mix.t()).clamp(0, 1).numpy()
    return {"mean_c": mean_c, "max_c": max_c, "overall": sc.mean(dim=1).numpy(), "s_scale": s_scale}


@torch.no_grad()
def probe_eval(B, probe, device, bs=4, counterfactual=False):
    """Run the probe set; return gate stats, correlations, rel-delta, PSNR (+counterfactuals)."""
    B.eval()
    x, y, sev_gt, modes = probe
    N = x.size(0)
    B.dslt.record_gates = True
    gates, effs, rels, psnr_pred = [], [], [], []
    feats = {"legacy_in": [], "rich_in": []}
    for i in range(0, N, bs):
        xb = x[i:i + bs].to(device); yb = y[i:i + bs].to(device)
        out, sev, g = B.forward(xb)
        gates.append(g.float().cpu())
        info = B.dslt.last_gate_info if B.gating else None
        if info is None:  # ungated config: no gate info is recorded
            info = {"eff_gates": torch.ones(len(xb), 4), "rel_delta": torch.zeros(len(xb), 4)}
        effs.append(info["eff_gates"].cpu()); rels.append(info["rel_delta"].cpu())
        psnr_pred.append(psnr_per_image(out, yb).cpu())
        # gate-head inputs for the H3 linear probe
        e1 = B.de.enc1(xb); e2 = B.de.enc2(e1); e3 = B.de.enc3(e2); e4 = B.de.enc4(e3)
        feats["legacy_in"].append(e4.mean(dim=(2, 3)).cpu())
        feats["rich_in"].append(torch.cat([
            e3.mean(dim=(2, 3)), e3.amax(dim=(2, 3)), e4.mean(dim=(2, 3)), e4.amax(dim=(2, 3)),
            sev.mean(dim=(2, 3)), sev.amax(dim=(2, 3))], dim=1).cpu())
    B.dslt.record_gates = False
    gates = torch.cat(gates); effs = torch.cat(effs); rels = torch.cat(rels)
    psnr_pred = torch.cat(psnr_pred)
    G = gates.numpy(); E = effs.numpy()
    F_ = sev_features(sev_gt)
    tgt = gate_targets_from_severity(sev_gt).numpy()

    res = {
        "gate_mean": G.mean(0).tolist(), "gate_std": G.std(0).tolist(),
        "gate_mean_mild": G[modes == 0].mean(0).tolist(), "gate_mean_heavy": G[modes == 1].mean(0).tolist(),
        "gate_std_mild": G[modes == 0].std(0).tolist(), "gate_std_heavy": G[modes == 1].std(0).tolist(),
        "eff_gate_mean": E.mean(0).tolist(), "eff_gate_std": E.std(0).tolist(),
        "rel_delta_mean": rels.numpy().mean(0).tolist(),
        "psnr_pred": float(psnr_pred.mean()),
        "scale_params": [float(t.scale) for t in (B.dslt.translator_e2, B.dslt.translator_e3,
                                                   B.dslt.translator_e4, B.dslt.translator_bn)],
        "gt_sev_std_mild": F_["overall"][modes == 0].std().item(),
        "gt_sev_std_heavy": F_["overall"][modes == 1].std().item(),
        "gt_sev_mean_mild": F_["overall"][modes == 0].mean().item(),
        "gt_sev_mean_heavy": F_["overall"][modes == 1].mean().item(),
    }
    # correlations of each gate with GT severity descriptors
    corr = {}
    for gi, gn in enumerate(GATE_NAMES):
        row = {}
        for ci, cn in enumerate(SEV_NAMES):
            row["mean_" + cn] = pearson(G[:, gi], F_["mean_c"][:, ci])
            row["max_" + cn] = pearson(G[:, gi], F_["max_c"][:, ci])
        row["overall"] = pearson(G[:, gi], F_["overall"])
        row["own_scale_sev"] = pearson(G[:, gi], F_["s_scale"][:, gi])
        row["target_gate"] = pearson(G[:, gi], tgt[:, gi])
        row["eff_vs_overall"] = pearson(E[:, gi], F_["overall"])
        corr[gn] = row
    res["corr"] = corr
    res["gate_mae_to_target"] = np.abs(G - tgt).mean(0).tolist()
    res["max_abs_corr_overall"] = float(max(abs(corr[g]["overall"]) for g in GATE_NAMES))
    res["mean_corr_target"] = float(np.mean([corr[g]["target_gate"] for g in GATE_NAMES]))
    res["_G"] = G; res["_feats"] = {k: torch.cat(v).numpy() for k, v in feats.items()}
    res["_overall"] = F_["overall"]; res["_modes"] = modes

    if counterfactual:
        def run(override=None, gating=None):
            ps = []
            for i in range(0, N, bs):
                xb = x[i:i + bs].to(device); yb = y[i:i + bs].to(device)
                out, _, _ = B.forward(xb, gate_override=override, gating=gating)
                ps.append(psnr_per_image(out, yb).cpu())
            return float(torch.cat(ps).mean())
        mean_g = gates.mean(0).to(device)
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
        cf = {
            "pred": float(psnr_pred.mean()),
            "ungated": run(gating=False),
            "const_mean": run(override=lambda g: mean_g.view(1, 4).expand_as(g)),
            "zeros": run(override=lambda g: torch.zeros_like(g)),
            "ones": run(override=lambda g: torch.ones_like(g)),
        }
        # shuffled gates (right gate distribution, wrong image): per-batch roll
        cf["shuffled"] = run(override=lambda g: g.roll(1, dims=0))
        res["counterfactual_psnr"] = cf
    return res


def ridge_cv_r2(X, y, alpha=10.0, folds=5):
    """5-fold CV R^2 of ridge regression on standardised X."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n = len(y); idx = np.arange(n); rng = np.random.RandomState(0); rng.shuffle(idx)
    pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        Xt, yt = X[tr], y[tr]
        w = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(X.shape[1]), Xt.T @ (yt - yt.mean()))
        pred[te] = X[te] @ w + yt.mean()
    ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))


def h3_probe(res):
    y = res["_overall"]; m = res["_modes"]
    out = {}
    for name, X in res["_feats"].items():
        out[name] = {
            "all": ridge_cv_r2(X, y),
            "mild_only": ridge_cv_r2(X[m == 0], y[m == 0]),
            "heavy_only": ridge_cv_r2(X[m == 1], y[m == 1]),
        }
    return out


def grad_routing_test(B, batch, device):
    """H2: gradient reaching the gate head from the severity (damage) loss vs the pixel loss."""
    B.train()
    x, y, sev = [t.to(device) for t in batch]
    def gnorm(loss_fn):
        for p in B.params:
            p.grad = None
        sv, g = B.de(x); b, sk = B.enc(x)
        bn_t, sk_t = B.dslt(b, sk, scale_gates=g, use_gating=True,
                            severity_map=sv.detach() if B.spatial else None)
        out = torch.clamp(x + B.dec(bn_t, sk_t), 0, 1)
        loss = loss_fn(out, sv, g)
        loss.backward(retain_graph=False)
        gm = [p.grad for p in B.de.gate_mlp.parameters() if p.grad is not None]
        return float(torch.sqrt(sum((t ** 2).sum() for t in gm))) if gm else 0.0
    import torch.nn.functional as F
    r = {
        "gate_mlp_grad_from_severity_loss": gnorm(lambda o, sv, g: F.l1_loss(sv, sev) + 0 * g.sum() + 0 * o.sum()),
        "gate_mlp_grad_from_pixel_loss": gnorm(lambda o, sv, g: F.l1_loss(o, y) + 0 * sv.sum()),
    }
    if B.variant == "fixed":
        r["gate_mlp_grad_from_gate_sup_loss"] = gnorm(
            lambda o, sv, g: B.crit.compute_gate_loss(g, sev)[0] + 0 * o.sum() + 0 * sv.sum())
    for p in B.params:
        p.grad = None
    return r


# ───────────────────────── training / diagnose ─────────────────────────

def train_steps(B, loader, iters, device, lr=2e-4, log_cb=None, log_every=50, seed=0):
    opt = torch.optim.AdamW(B.params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))
    hist = []
    it = 0
    t0 = time.time()
    while it < iters:
        for degraded, clean, gt_sev in loader:
            B.train()
            degraded, clean, gt_sev = degraded.to(device), clean.to(device), gt_sev.to(device)
            opt.zero_grad(set_to_none=True)
            out, sev, gates = B.forward(degraded)
            total, l_char, _, _, _, l_dmg = B.crit(out, clean, pred_severity=sev, target_severity=gt_sev,
                                                   pred_gates=gates if B.gating else None)
            total.backward()
            torch.nn.utils.clip_grad_norm_(B.params, max_norm=1.0)
            opt.step()
            it += 1
            with torch.no_grad():
                p = float(psnr_per_image(out, clean).mean())
            hist.append({"iter": it, "loss": float(total.detach()), "char": float(l_char.detach()),
                         "psnr": p, "gate_loss": float(B.crit.last_gate_loss),
                         "train_gate_mean": gates.detach().float().mean(0).cpu().tolist(),
                         "train_gate_std": gates.detach().float().std(0).cpu().tolist()})
            if log_cb is not None and (it % log_every == 0 or it == 1 or it == iters):
                log_cb(it, time.time() - t0)
            if it >= iters:
                break
    return hist


def strip(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def cmd_diagnose(args):
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={device} variant={args.variant} iters={args.iters}")
    probe = make_probe(device, args.probe_n, args.crop)
    set_seed(args.seed)
    B = Bundle(args.variant, device)
    loader = make_train_loader(args.batch, args.crop, args.seed)
    log = []

    def cb(it, el):
        _empty_cache()
        r = probe_eval(B, probe, device)
        _empty_cache()
        entry = {"iter": it, **strip(r)}
        log.append(entry)
        print(f"[{it:4d}] {el:5.0f}s gate_mean={np.round(r['gate_mean'], 3)} gate_std={np.round(r['gate_std'], 3)} "
              f"corr_overall={[round(r['corr'][g]['overall'], 2) for g in GATE_NAMES]} "
              f"rel_delta={np.round(r['rel_delta_mean'], 3)} psnr={r['psnr_pred']:.2f}", flush=True)

    hist = train_steps(B, loader, args.iters, device, log_cb=cb, log_every=args.log_every)
    _empty_cache()
    final = probe_eval(B, probe, device, counterfactual=True)
    _empty_cache()
    h3 = h3_probe(final)
    b0 = next(iter(DataLoader(Subset(OnlineDegradationDataset(CLEAN_DIR, args.crop, True, "mild"), range(8)),
                              batch_size=4)))
    h2 = grad_routing_test(B, b0, device)
    summary = {"variant": args.variant, "iters": args.iters, "final": strip(final), "h3_ridge_cv_r2": h3,
               "h2_grad_routing": h2, "structural_h4": "e1 skip is returned unchanged in gated and ungated paths "
               "(no translator on e1) -> never gated"}
    payload = {"summary": summary, "log": log, "train_hist": hist}
    with open(os.path.join(OUT_DIR, f"diagnosis_{args.variant}.json"), "w") as f:
        json.dump(payload, f, indent=1)
    np.save(os.path.join(OUT_DIR, f"probe_gates_{args.variant}.npy"),
            np.concatenate([final["_G"], final["_overall"][:, None], final["_modes"][:, None]], axis=1))
    plot_variant(payload, os.path.join(OUT_DIR, f"diagnosis_{args.variant}.png"))
    print(json.dumps({k: summary[k] for k in ("h2_grad_routing", "h3_ridge_cv_r2")}, indent=1))
    print("counterfactual PSNR:", final["counterfactual_psnr"])
    print("learned translator scales:", final["scale_params"])
    print("rel_delta:", final["rel_delta_mean"])


def plot_variant(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = payload["log"]
    its = [e["iter"] for e in log]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for gi, gn in enumerate(GATE_NAMES):
        m = np.array([e["gate_mean"][gi] for e in log]); s = np.array([e["gate_std"][gi] for e in log])
        ax[0, 0].plot(its, m, color=cols[gi], label=gn)
        ax[0, 0].fill_between(its, m - s, m + s, color=cols[gi], alpha=0.15)
        ax[0, 1].plot(its, [e["gate_std"][gi] for e in log], color=cols[gi], label=gn)
        ax[0, 2].plot(its, [e["corr"][gn]["overall"] for e in log], color=cols[gi], label=gn)
        ax[1, 0].plot(its, [e["corr"][gn]["target_gate"] for e in log], color=cols[gi], label=gn)
        ax[1, 1].plot(its, [e["rel_delta_mean"][gi] for e in log], color=cols[gi], label=gn)
    ax[0, 0].set_title("gate mean +/- std over probe images"); ax[0, 0].set_ylim(0, 1)
    ax[0, 1].set_title("gate std across images")
    ax[0, 2].set_title("corr(gate, GT overall severity)"); ax[0, 2].set_ylim(-1, 1)
    ax[1, 0].set_title("corr(gate, target gate)"); ax[1, 0].set_ylim(-1, 1)
    ax[1, 1].set_title("||raw-orig|| / ||orig||")
    th = payload["train_hist"]
    ax[1, 2].plot([h["iter"] for h in th], [h["psnr"] for h in th], alpha=0.5, label="train batch")
    ax[1, 2].plot(its, [e["psnr_pred"] for e in log], "k-o", label="probe")
    ax[1, 2].set_title("PSNR (dB)")
    for a in ax.flat:
        a.set_xlabel("iter"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle(f"Gate diagnosis - variant={payload['summary']['variant']}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ───────────────────────── A vs B sanity ─────────────────────────

def cmd_sanity(args):
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    probe = make_probe(device, args.probe_n, args.crop)
    runs = {}
    # A, A_repeat (noise-floor control), B  -- identical seed / init / data order
    for name, gating in [("A_ungated", False), ("A_ungated_repeat", False), ("B_gated", True)]:
        set_seed(args.seed)
        B = Bundle(args.variant, device, gating=gating)
        loader = make_train_loader(args.batch, args.crop, args.seed)
        hist = train_steps(B, loader, args.iters, device, log_every=10 ** 9)
        _empty_cache()
        r = probe_eval(B, probe, device, counterfactual=False)
        del B
        _empty_cache()
        runs[name] = {"hist": hist, "probe_psnr": r["psnr_pred"], "gate_mean": r["gate_mean"],
                      "gate_std": r["gate_std"], "scale_params": r["scale_params"],
                      "corr_target": r["mean_corr_target"]}
        print(f"{args.variant}/{name}: final train loss(last20)={np.mean([h['loss'] for h in hist[-20:]]):.4f} "
              f"probe psnr={r['psnr_pred']:.3f} scales={np.round(r['scale_params'], 3)}", flush=True)
    la = np.array([h["char"] for h in runs["A_ungated"]["hist"]])
    lr_ = np.array([h["char"] for h in runs["A_ungated_repeat"]["hist"]])
    lb = np.array([h["char"] for h in runs["B_gated"]["hist"]])
    summ = {
        "variant": args.variant, "iters": args.iters,
        "mean_abs_char_diff_A_vs_Arepeat": float(np.abs(la - lr_).mean()),
        "mean_abs_char_diff_A_vs_B": float(np.abs(la - lb).mean()),
        "mean_abs_char_diff_A_vs_B_last50": float(np.abs(la[-50:] - lb[-50:]).mean()),
        "mean_abs_char_diff_A_vs_Arepeat_last50": float(np.abs(la[-50:] - lr_[-50:]).mean()),
        "probe_psnr": {k: v["probe_psnr"] for k, v in runs.items()},
        "scale_params": {k: v["scale_params"] for k, v in runs.items()},
        "B_gate_mean": runs["B_gated"]["gate_mean"], "B_gate_std": runs["B_gated"]["gate_std"],
    }
    with open(os.path.join(OUT_DIR, f"sanity_{args.variant}.json"), "w") as f:
        json.dump({"summary": summ, "runs": runs}, f, indent=1)
    print(json.dumps(summ, indent=1))


# ───────────────────────── compare before/after ─────────────────────────

def cmd_compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = {v: json.load(open(os.path.join(OUT_DIR, f"diagnosis_{v}.json"))) for v in ("legacy", "fixed")}
    rows = {}
    for v, p in P.items():
        f = p["summary"]["final"]
        rows[v] = {
            "gate_mean": np.round(f["gate_mean"], 3).tolist(), "gate_std": np.round(f["gate_std"], 3).tolist(),
            "corr_gate_vs_overall_severity": [round(f["corr"][g]["overall"], 3) for g in GATE_NAMES],
            "corr_gate_vs_target_gate": [round(f["corr"][g]["target_gate"], 3) for g in GATE_NAMES],
            "eff_gate_std": np.round(f["eff_gate_std"], 3).tolist(),
            "rel_delta": np.round(f["rel_delta_mean"], 3).tolist(),
            "counterfactual_psnr": {k: round(x, 3) for k, x in f["counterfactual_psnr"].items()},
        }
    with open(os.path.join(OUT_DIR, "before_after.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(json.dumps(rows, indent=1))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for v, c in (("legacy", "#7f7f7f"), ("fixed", "#1f77b4")):
        G = np.load(os.path.join(OUT_DIR, f"probe_gates_{v}.npy"))
        modes = G[:, 5]; sev = G[:, 4]
        ax[0].scatter(sev, G[:, 3], s=12, alpha=0.6, c=c, label=f"{v}: bottleneck gate")
        ax[1].scatter(sev, G[:, 0], s=12, alpha=0.6, c=c, label=f"{v}: e2 gate")
    ax[0].set_xlabel("GT overall severity"); ax[0].set_ylabel("gate"); ax[0].set_ylim(0, 1)
    ax[1].set_xlabel("GT overall severity"); ax[1].set_ylim(0, 1)
    for a in ax[:2]:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    x = np.arange(4); w = 0.38
    ax[2].bar(x - w / 2, rows["legacy"]["gate_std"], w, label="legacy", color="#7f7f7f")
    ax[2].bar(x + w / 2, rows["fixed"]["gate_std"], w, label="fixed", color="#1f77b4")
    ax[2].set_xticks(x); ax[2].set_xticklabels(GATE_NAMES); ax[2].set_title("gate std across probe images")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "before_after.png"), dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "sanity", "compare"])
    ap.add_argument("--variant", choices=["legacy", "fixed"], default="legacy")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--probe_n", type=int, default=48, help="probe images per severity mode")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"diagnose": cmd_diagnose, "sanity": cmd_sanity, "compare": cmd_compare}[a.cmd](a)
