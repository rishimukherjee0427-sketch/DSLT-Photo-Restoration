"""
Definitive gated-vs-ungated ablation for DSLT v5 (held-out protocol, matches
experiments/train_baseline.py wherever noted).

Configs (everything identical except the listed differences; same seed before building
each, same per-epoch data order and degradations, same loss incl. damage-estimator
severity loss):
  A  ungated (use_gating=False). ALSO the official "DSLT v5 retrained on clean train split".
  B  adaptive gated (gate_mode='film'), no gate losses.
  C  adaptive gated (film) + gate severity supervision (w_gate_sup=0.5) + sparsity (0.01).

Protocol
  train data : data/train_clean restricted to data/splits/train.txt, severity='mixed'
               (uniform mild/medium/severe per sample), crop 128, flips/rot90 augment, batch 4.
  budget     : an "epoch" here = --steps_per_epoch (200) OPTIMIZER STEPS, NOT a pass over the data;
               30 epochs = 6000 steps (24000 samples ~ 5.6 passes over the 4279 train images).
  optimiser  : AdamW lr 2e-4 wd 1e-4 betas (0.9, 0.99), 500-step linear warm-up then cosine to
               0.5 % of lr over the whole budget, grad-clip 1.0 (all trainable params), EMA 0.999
               (with the same (1+t)/(10+t) warm-up as train_baseline.py).
  validation : after EVERY epoch, EMA weights, eval(), no_grad, on the first --val_n (48) images
               of data/benchmark_cache/val x {mild, medium, severe} at 256 px (the exact inputs the
               final benchmark uses; held-out from train).  Metrics from utils.benchmark:
               psnr_np (clip to [0,1], float64, per-image cap 50 dB) and ssim_np (skimage, 7x7
               uniform window); LPIPS is not computed.  Model selection = mean of the three
               per-severity mean PSNRs ("val_psnr_mean").
  data order : the per-epoch RNG (python `random`, numpy, torch sampler generator) is re-seeded
               from (seed, epoch), so A/B/C see identical batches/degradations and --resume is exact
               at epoch boundaries.

Outputs (results/ablation/, or results/ablation/smoke/ with --smoke):
  {cfg}_best.pth     EMA weights {encoder, decoder, dslt, damage_estimator} + config metadata
  {cfg}_last.pth     full resumable state (raw + EMA nets, optimiser, scheduler, RNG, epoch, history)
  {cfg}_history.json written after every epoch

Usage
  python experiments/run_ablation.py --config A --resume
  python experiments/run_ablation.py --config C --smoke
"""
import os
import sys
import json
import math
import time
import copy
import random
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.autoencoder import Encoder, Decoder
from models.latent_translator import MultiScaleDSLT
from models.damage_estimator import DamageEstimator
from models.losses import CombinedLoss, gate_targets_from_severity
from utils.dataloader import OnlineDegradationDataset
from utils import benchmark as bm

OUT_DIR = os.path.join(ROOT, "results", "ablation")
SEVERITIES = bm.SEVERITIES  # ('mild', 'medium', 'severe')
GATE_NAMES = ["e2", "e3", "e4", "bn"]

CONFIGS = {
    "A": dict(name="dslt_A_ungated", use_gating=False, w_gate_sup=0.0, w_gate_sparsity=0.0,
              desc="ungated (use_gating=False)"),
    "B": dict(name="dslt_B_gated", use_gating=True, w_gate_sup=0.0, w_gate_sparsity=0.0,
              desc="adaptive film gating, no gate losses"),
    "C": dict(name="dslt_C_gated_sup", use_gating=True, w_gate_sup=0.5, w_gate_sparsity=0.01,
              desc="adaptive film gating + gate supervision (0.5) + sparsity (0.01)"),
}


# ------------------------------------------------------------------ helpers
def get_device():
    return bm.get_device()


def empty_cache(device):
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


class Nets(nn.Module):
    """Container so optimiser / EMA / state_dict handle all four sub-networks at once.
    Construction order (and hence RNG consumption) is identical for every config."""

    def __init__(self):
        super().__init__()
        self.damage_estimator = DamageEstimator(3, 32)          # gate_head='rich'
        self.encoder = Encoder()
        self.dslt = MultiScaleDSLT()                            # gate_mode='film'
        self.decoder = Decoder()

    def restore(self, x, use_gating, gates_out=None):
        """Returns (restored, severity_map, scale_gates). Same code path for train and eval."""
        sev, g = self.damage_estimator(x)
        b, sk = self.encoder(x)
        bn_t, sk_t = self.dslt(b, sk, scale_gates=g, use_gating=use_gating, severity_map=sev.detach())
        out = torch.clamp(x + self.decoder(bn_t, sk_t), 0.0, 1.0)
        return out, sev, g


class EMA:
    """Same update rule as train_baseline.py (decay 0.999 with (1+t)/(10+t) warm-up)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for e, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def make_lr_lambda(total, warmup, min_ratio):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def seed_all(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


def _f(x):
    """float -> JSON-safe (nan/inf -> None)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return _f(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------ validation
class ValSet:
    """First n val images x 3 severities from the benchmark cache (uint8 in RAM, ~small) plus the
    GT severity maps of the same deterministic samples (for gate-vs-severity correlation)."""

    def __init__(self, n):
        self.n = n
        self.deg, self.clean, self.gt_scal, self.gt_gate, self.gt_sevmap_mean = {}, {}, {}, {}, {}
        self.sev_maps = {}
        for sev in SEVERITIES:
            names, dp, cp = bm.cached_paths("val", sev, n)
            self.deg[sev] = np.stack([bm._read_rgb(p) for p in dp])
            self.clean[sev] = np.stack([bm._read_rgb(p) for p in cp])
            ds = bm.build_dataset("val", sev, 256, 0)          # same samples the cache was rendered from
            maps = torch.stack([ds[i][2] for i in range(len(names))])      # (n,4,H,W)
            self.sev_maps[sev] = maps.half()                              # keep RAM small
            self.gt_gate[sev] = gate_targets_from_severity(maps).numpy()  # (n,4) target gates

    def batches(self, sev, bs):
        for i in range(0, self.n, bs):
            sl = slice(i, i + bs)
            x = torch.from_numpy(self.deg[sev][sl]).permute(0, 3, 1, 2).float() / 255.0
            c = torch.from_numpy(self.clean[sev][sl]).permute(0, 3, 1, 2).float() / 255.0
            yield x, c, self.sev_maps[sev][sl].float()


@torch.no_grad()
def validate(ema_nets, vset, cfg, device, bs=4):
    ema_nets.eval()
    ema_nets.dslt.record_gates = bool(cfg["use_gating"])
    out = {}
    sc_all, eff_all, rel_all, tgt_all, lvl_all = [], [], [], [], []
    by_sev = {}
    sev_l1 = []
    for li, sev in enumerate(SEVERITIES):
        P, S = [], []
        sc, eff, rel = [], [], []
        for x, c, gt_sev in vset.batches(sev, bs):
            x = x.to(device)
            y, sm, g = ema_nets.restore(x, cfg["use_gating"])
            yn = y.float().cpu().permute(0, 2, 3, 1).numpy()
            cn = c.permute(0, 2, 3, 1).numpy()
            for k in range(len(yn)):
                P.append(bm.psnr_np(yn[k], cn[k]))
                S.append(bm.ssim_np(yn[k], cn[k]))
            sev_l1.append(float((sm.float().cpu() - gt_sev).abs().mean()))
            sc.append(g.float().cpu().numpy())
            if cfg["use_gating"]:
                gi = ema_nets.dslt.last_gate_info
                eff.append(gi["eff_gates"].float().cpu().numpy())
                rel.append(gi["rel_delta"].float().cpu().numpy())
            del x, y, sm, g
        out[sev] = dict(psnr=_f(np.mean(P)), ssim=_f(np.mean(S)))
        sc = np.concatenate(sc)
        by_sev[sev] = dict(scalar_mean=sc.mean(0).tolist())
        sc_all.append(sc)
        tgt_all.append(vset.gt_gate[sev])
        lvl_all.append(np.full(len(sc), li, np.float64))
        if cfg["use_gating"]:
            eff = np.concatenate(eff); rel = np.concatenate(rel)
            by_sev[sev].update(eff_mean=eff.mean(0).tolist(), rel_delta_mean=rel.mean(0).tolist())
            eff_all.append(eff); rel_all.append(rel)
    ema_nets.dslt.record_gates = False
    out["psnr_mean"] = _f(np.mean([out[s]["psnr"] for s in SEVERITIES]))
    out["ssim_mean"] = _f(np.mean([out[s]["ssim"] for s in SEVERITIES]))
    out["sev_map_l1"] = _f(np.mean(sev_l1))

    if not cfg["use_gating"]:
        return out, "n/a"
    sc, eff, rel = np.concatenate(sc_all), np.concatenate(eff_all), np.concatenate(rel_all)
    tgt, lvl = np.concatenate(tgt_all), np.concatenate(lvl_all)
    gates = dict(
        names=GATE_NAMES,
        scalar_mean=sc.mean(0).tolist(), scalar_std=sc.std(0).tolist(),
        eff_mean=eff.mean(0).tolist(), eff_std=eff.std(0).tolist(),
        rel_delta_mean=rel.mean(0).tolist(),
        by_severity=by_sev,
        # correlation (over the 3*n val samples) of the predicted scalar / effective gate with the
        # GT-derived target gate of the same scale, and with the ordinal severity level (0,1,2)
        corr_scalar_vs_target=[_corr(sc[:, k], tgt[:, k]) for k in range(4)],
        corr_eff_vs_target=[_corr(eff[:, k], tgt[:, k]) for k in range(4)],
        corr_scalar_vs_level=[_corr(sc[:, k], lvl) for k in range(4)],
        corr_eff_vs_level=[_corr(eff[:, k], lvl) for k in range(4)],
        target_mean=tgt.mean(0).tolist(),
    )
    return out, gates


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--epochs", type=int, default=None, help="default 30 (smoke: 2)")
    ap.add_argument("--steps_per_epoch", type=int, default=None, help="default 200 (smoke: 5)")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.005)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_n", type=int, default=48, help="val images per severity (default 48; smoke: 8)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--clean_dir", default=os.path.join(ROOT, "data", "train_clean"))
    ap.add_argument("--file_list", default=os.path.join(ROOT, "data", "splits", "train.txt"))
    ap.add_argument("--resume", action="store_true", help="resume from {cfg}_last.pth if present")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (2 epochs x 5 steps, val on 8 imgs), outputs to results/ablation/smoke/")
    ap.add_argument("--stop_after_epoch", type=int, default=None,
                    help="(testing) exit after this epoch without finishing, to test --resume")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = args.epochs or 2
        args.steps_per_epoch = args.steps_per_epoch or 5
        args.val_n = min(args.val_n, 8) if args.val_n == 48 else args.val_n
        args.warmup = 2 if args.warmup == 500 else args.warmup
        args.log_every = min(args.log_every, 5)
    args.epochs = args.epochs or 30
    args.steps_per_epoch = args.steps_per_epoch or 200
    out_dir = args.out_dir or (os.path.join(OUT_DIR, "smoke") if args.smoke else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    cfg = CONFIGS[args.config]
    tag = args.config
    best_path, last_path, hist_path = [os.path.join(out_dir, f"{tag}_{k}") for k in
                                       ("best.pth", "last.pth", "history.json")]
    device = get_device()
    total = args.epochs * args.steps_per_epoch

    # ---- data / validation set ----
    train_set = OnlineDegradationDataset(args.clean_dir, crop_size=args.crop, augment=True,
                                         severity="mixed", file_list=args.file_list)
    vset = ValSet(args.val_n)

    # ---- model (identical seed + construction order for every config) ----
    seed_all(args.seed)
    nets = Nets().to(device)
    ema = EMA(nets, args.ema_decay)
    n_params = sum(p.numel() for p in nets.parameters())
    params = [p for p in nets.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(total, args.warmup, args.min_lr_ratio))
    crit = CombinedLoss(device, mode="psnr", w_gate_sup=cfg["w_gate_sup"],
                        w_gate_sparsity=cfg["w_gate_sparsity"], w_gate_bimodal=0.0)

    step, epoch0, best_psnr, history = 0, 0, -1.0, []
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        nets.load_state_dict(ck["nets"]); ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, epoch0, best_psnr, history = ck["step"], ck["epoch"], ck["best_psnr"], ck["history"]
        print(f"[{tag}] resumed from {last_path}: epoch {epoch0}/{args.epochs}, step {step}, "
              f"best val PSNR {best_psnr:.3f}", flush=True)
        del ck
    elif args.resume:
        print(f"[{tag}] --resume given but no {last_path}; starting fresh", flush=True)

    print(f"[{tag}] {cfg['desc']} | {n_params/1e6:.2f}M params (incl. damage estimator"
          f"{'' if cfg['use_gating'] else ' + unused gate modulators'}) | device={device} | "
          f"{args.epochs} epochs x {args.steps_per_epoch} steps = {total} steps | bs={args.batch_size} "
          f"crop={args.crop} | val {args.val_n} imgs x 3 sev", flush=True)

    def save_history():
        with open(hist_path, "w") as f:
            json.dump(dict(config=tag, name=cfg["name"], cfg=cfg, args=vars(args), n_params=n_params,
                           history=history), f, indent=1)

    def atomic_save(obj, path):
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)

    meta = dict(config=tag, name=cfg["name"], use_gating=cfg["use_gating"], gate_mode="film",
                gate_head="rich", w_gate_sup=cfg["w_gate_sup"], w_gate_sparsity=cfg["w_gate_sparsity"],
                crop=args.crop, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
                seed=args.seed, uses_ema=True, ema_decay=args.ema_decay)

    t_start = time.time()
    t_resume_epochs = 0
    keys = ("loss", "charb", "ssim_loss", "damage", "gate_sup", "gate_sparsity", "gate_loss")
    for epoch in range(epoch0 + 1, args.epochs + 1):
        # per-epoch RNG: identical across configs, makes resume exact at epoch boundaries
        es = args.seed * 100003 + epoch
        seed_all(es)
        g = torch.Generator(); g.manual_seed(es)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=0, generator=g)
        nets.train()
        run = {k: 0.0 for k in keys}; n_run = 0
        gate_run = np.zeros(4); n_gate = 0
        ep_run = {k: 0.0 for k in keys}; n_ep = 0
        t_ep = time.time(); t_blk = time.time(); skipped = 0
        for it, (deg, clean, gt_sev) in enumerate(loader):
            if it >= args.steps_per_epoch:
                break
            deg, clean, gt_sev = deg.to(device), clean.to(device), gt_sev.to(device)
            out, sev, gates = nets.restore(deg, cfg["use_gating"])
            total_loss, l_char, _, l_ssim, _, l_dmg = crit(
                out, clean, pred_severity=sev, target_severity=gt_sev, pred_gates=gates)
            if not torch.isfinite(total_loss):
                print(f"!! non-finite loss at step {step}; skipping batch", flush=True)
                opt.zero_grad(set_to_none=True); step += 1; sched.step(); skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step(); sched.step(); step += 1
            ema.update(nets, step)

            gt = crit.last_gate_terms
            vals = dict(loss=total_loss.item(), charb=l_char.item(), ssim_loss=l_ssim.item(),
                        damage=l_dmg.item(), gate_sup=gt["sup"], gate_sparsity=gt["sparsity"],
                        gate_loss=float(crit.last_gate_loss))
            for k in keys:
                run[k] += vals[k]; ep_run[k] += vals[k]
            n_run += 1; n_ep += 1
            gate_run += gates.detach().float().mean(0).cpu().numpy(); n_gate += 1
            if step % args.log_every == 0:
                sps = (time.time() - t_blk) / n_run
                gtxt = ("" if not cfg["use_gating"] else
                        f" | gates {np.array2string(gate_run / n_gate, precision=2, separator=',')}")
                print(f"[{tag}] ep {epoch} step {step}/{total} loss {run['loss']/n_run:.4f} "
                      f"(charb {run['charb']/n_run:.4f} 1-ssim {run['ssim_loss']/n_run:.3f} "
                      f"dmg {run['damage']/n_run:.4f} gsup {run['gate_sup']/n_run:.3f}) "
                      f"lr {sched.get_last_lr()[0]:.2e} | {sps:.2f} s/step{gtxt}", flush=True)
                run = {k: 0.0 for k in keys}; n_run = 0; gate_run = np.zeros(4); n_gate = 0
                t_blk = time.time()
        del loader
        t_train = time.time() - t_ep
        if device.type == "mps":
            torch.mps.synchronize()
        empty_cache(device)

        # ---- validation (EMA weights) ----
        t_v = time.time()
        val, gates = validate(ema.shadow, vset, cfg, device)
        t_val = time.time() - t_v
        empty_cache(device)

        is_best = val["psnr_mean"] > best_psnr
        if is_best:
            best_psnr = val["psnr_mean"]
        rec = dict(epoch=epoch, step=step, lr=sched.get_last_lr()[0],
                   train={k: ep_run[k] / max(1, n_ep) for k in keys},
                   train_s_per_step=t_train / max(1, n_ep), skipped=skipped,
                   val=val, gates=gates, best=is_best, time_train_s=t_train, time_val_s=t_val)
        history.append(rec)

        if is_best:
            e = ema.shadow
            atomic_save(dict(encoder=e.encoder.state_dict(), decoder=e.decoder.state_dict(),
                             dslt=e.dslt.state_dict(), damage_estimator=e.damage_estimator.state_dict(),
                             epoch=epoch, step=step, val_psnr_mean=best_psnr, val=val,
                             n_params=n_params, **meta), best_path)
        atomic_save(dict(nets=nets.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(),
                         sched=sched.state_dict(), epoch=epoch, step=step, best_psnr=best_psnr,
                         history=history, rng=dict(py=random.getstate(), np=np.random.get_state(),
                                                   torch=torch.get_rng_state()), **meta), last_path)
        save_history()

        done = epoch - epoch0
        t_resume_epochs = time.time() - t_start
        eta = t_resume_epochs / done * (args.epochs - epoch)
        gtxt = "" if gates == "n/a" else (" | gate(eff) " + np.array2string(
            np.array(gates["eff_mean"]), precision=2, separator=",") + " corr_tgt " + str(
            [None if c is None else round(c, 2) for c in gates["corr_scalar_vs_target"]]))
        print(f"[{tag}] EPOCH {epoch}/{args.epochs} loss {rec['train']['loss']:.4f} | VAL PSNR "
              f"mild {val['mild']['psnr']:.2f} med {val['medium']['psnr']:.2f} sev {val['severe']['psnr']:.2f} "
              f"mean {val['psnr_mean']:.3f} | SSIM mean {val['ssim_mean']:.4f}"
              f"{' *best*' if is_best else ''} | train {t_train:.0f}s val {t_val:.0f}s | "
              f"ETA {eta/60:.1f} min{gtxt}", flush=True)

        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch and epoch < args.epochs:
            print(f"[{tag}] --stop_after_epoch {epoch}: exiting early (test)", flush=True)
            return

    if epoch0 >= args.epochs:
        print(f"[{tag}] already finished ({epoch0} epochs); nothing to do.", flush=True)
    else:
        print(f"[{tag}] done: best val PSNR(mean of 3 severities) {best_psnr:.3f} | "
              f"{(time.time()-t_start)/60:.1f} min -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
