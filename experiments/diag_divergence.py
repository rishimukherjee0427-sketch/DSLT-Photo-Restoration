"""Divergence diagnostic for a Stage-3 run (e.g. Tfull seed 2026).

Re-runs the first N steps of experiments/run_ablation.py with the IDENTICAL setup (seeding, construction
order, per-epoch data order, optimiser, LR schedule for the full 20k-step run, EMA, loss, clipping) and
logs every step: losses, pre-clip global gradient norm, per-module gradient norms, translator/decoder
residual scales, translator relative update ||T(z)-z||/||z||, and batch statistics. At each epoch end
it runs the same EMA validation as training, so reproduction can be checked against the original
history (val PSNR per epoch must match).

Writes to results/v9_convergence_pilot/diag/{cfg}_seed{seed}/ (steps.jsonl, epochs.json). Nothing
existing is overwritten; the training checkpoints are not touched.
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import experiments.run_ablation as ra  # noqa: E402  (sets MPS env vars on import)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from models.latent_translator import MultiScaleDSLT  # noqa: E402
from models.losses import CombinedLoss  # noqa: E402
from utils.dataloader import OnlineDegradationDataset  # noqa: E402


def gnorm(params):
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Tfull")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--steps", type=int, default=5000)
    # must equal the original run (schedule depends on the full 20k-step total)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.005)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--val_n", type=int, default=250)
    ap.add_argument("--ref_history", default=None, help="original {cfg}_history.json for reproduction check")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "v9_convergence_pilot", "diag", f"{args.config}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    steps_path = os.path.join(out_dir, "steps.jsonl")
    if os.path.exists(steps_path):
        raise SystemExit(f"{steps_path} exists; refusing to overwrite")
    ref = None
    rp = args.ref_history or os.path.join(ROOT, "results", "v9_convergence_pilot", f"seed{args.seed}",
                                          f"{args.config}_history.json")
    if os.path.exists(rp):
        ref = {r["epoch"]: r for r in json.load(open(rp))["history"]}

    cfg = ra.CONFIGS[args.config]
    device = ra.get_device()
    total = args.epochs * args.steps_per_epoch
    train_set = OnlineDegradationDataset(os.path.join(ROOT, "data", "train_clean"), crop_size=args.crop,
                                         augment=True, severity="mixed",
                                         file_list=os.path.join(ROOT, "data", "splits", "train.txt"))
    vset = ra.ValSet(args.val_n)

    ra.seed_all(args.seed)
    nets = ra.Nets(cfg.get("translators", MultiScaleDSLT.ALL_TRANSLATORS), cfg.get("gate_mode", "film")).to(device)
    ema = ra.EMA(nets, args.ema_decay)
    params = [p for p in nets.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, ra.make_lr_lambda(total, args.warmup, args.min_lr_ratio))
    crit = CombinedLoss(device, mode="psnr", w_gate_sup=cfg["w_gate_sup"], w_gate_sparsity=cfg["w_gate_sparsity"],
                        w_gate_bimodal=0.0, vgg_normalize=True)

    groups = {"damage_estimator": nets.damage_estimator, "encoder": nets.encoder, "decoder": nets.decoder}
    trs = [k for k in MultiScaleDSLT.ALL_TRANSLATORS if k in cfg.get("translators", MultiScaleDSLT.ALL_TRANSLATORS)]
    for k in trs:
        groups[f"T_{k}"] = getattr(nets.dslt, f"translator_{k}")
    gparams = {k: [p for p in m.parameters() if p.requires_grad] for k, m in groups.items()}

    rel = {}

    def hook(name):
        def f(mod, inp, out):
            z = inp[0].detach()
            rel[name] = float((out.detach() - z).float().norm() / (z.float().norm() + 1e-12))
        return f
    for k in trs:
        getattr(nets.dslt, f"translator_{k}").register_forward_hook(hook(k))

    print(f"[diag] {args.config} seed {args.seed}: {args.steps} steps, schedule total {total}, device {device}",
          flush=True)
    fsteps = open(steps_path, "w")
    step, epochs = 0, []
    n_ep_total = -(-args.steps // args.steps_per_epoch)
    t0 = time.time()
    for epoch in range(1, n_ep_total + 1):
        es = args.seed * 100003 + epoch
        ra.seed_all(es)
        g = torch.Generator(); g.manual_seed(es)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=0, generator=g)
        nets.train()
        ep = {"loss": 0.0, "charb": 0.0, "n": 0}
        for it, (deg, clean, gt_sev) in enumerate(loader):
            if it >= args.steps_per_epoch or step >= args.steps:
                break
            deg, clean, gt_sev = deg.to(device), clean.to(device), gt_sev.to(device)
            out, sev, gates = nets.restore(deg, cfg["use_gating"])
            total_loss, l_char, _, l_ssim, _, l_dmg = crit(out, clean, pred_severity=sev, target_severity=gt_sev,
                                                           pred_gates=gates)
            rec = dict(step=step + 1, epoch=epoch, lr=sched.get_last_lr()[0])
            if not torch.isfinite(total_loss):
                rec["nonfinite"] = True
                opt.zero_grad(set_to_none=True); step += 1; sched.step()
                fsteps.write(json.dumps(rec) + "\n")
                continue
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            per = {k: gnorm(v) for k, v in gparams.items()}
            gn = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            opt.step(); sched.step(); step += 1
            ema.update(nets, step)
            rec.update(loss=total_loss.item(), charb=l_char.item(), ssim_loss=l_ssim.item(), damage=l_dmg.item(),
                       grad_norm=gn, grad=per, rel_delta=dict(rel),
                       scale={k: float(getattr(nets.dslt, f"translator_{k}").scale) for k in trs},
                       dec_res_scale=float(nets.decoder.residual_scale),
                       out_mean=float(out.detach().mean()), out_sat=float(((out.detach() <= 0) | (out.detach() >= 1)).float().mean()),
                       clean_mean=float(clean.mean()), deg_l1=float((deg - clean).abs().mean()),
                       sev_mean=float(gt_sev.mean()))
            fsteps.write(json.dumps(rec) + "\n")
            ep["loss"] += rec["loss"]; ep["charb"] += rec["charb"]; ep["n"] += 1
            if step % 100 == 0:
                fsteps.flush()
                print(f"[diag] step {step} loss {rec['loss']:.4f} charb {rec['charb']:.4f} |g| {gn:.3f} "
                      f"lr {rec['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        del loader
        ra.empty_cache(device)
        val, _ = ra.validate(ema.shadow, vset, cfg, device)
        ra.empty_cache(device)
        e = dict(epoch=epoch, step=step, train_loss=ep["loss"] / max(1, ep["n"]),
                 train_charb=ep["charb"] / max(1, ep["n"]), val_psnr_mean=val["psnr_mean"],
                 val={k: val[k] for k in ("mild", "medium", "severe")})
        if ref and epoch in ref and ep["n"] == args.steps_per_epoch:
            e["ref_val_psnr_mean"] = ref[epoch]["val"]["psnr_mean"]
            e["ref_train_loss"] = ref[epoch]["train"]["loss"]
            e["reproduces"] = bool(abs(e["ref_val_psnr_mean"] - val["psnr_mean"]) < 1e-3
                                   and abs(e["ref_train_loss"] - e["train_loss"]) < 1e-4)
        epochs.append(e)
        json.dump(dict(args=vars(args), epochs=epochs), open(os.path.join(out_dir, "epochs.json"), "w"), indent=1)
        print(f"[diag] EPOCH {epoch}: train {e['train_loss']:.4f} val {val['psnr_mean']:.3f}"
              f" (ref {e.get('ref_val_psnr_mean', float('nan')):.3f}, reproduces={e.get('reproduces')})", flush=True)
    fsteps.close()
    print(f"[diag] done in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
