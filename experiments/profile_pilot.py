"""
Pre-flight profiler for the Stage-3 convergence pilot (Tfull vs Tnone at 256x256, FP32, MPS).

    python experiments/profile_pilot.py --steps 50 --warmup_steps 10

Per model, in a fresh process: steady-state training s/step on identical batches (same loss / optimiser as
run_ablation.py), peak MPS driver memory, time for one full validation pass (250 val images x 3 severities),
and the resulting runtime estimate for 20 epochs x 1000 steps with per-epoch validation.
Writes results/v9_convergence_pilot/preflight_profile.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "v9_convergence_pilot")


def one(cfg_name, steps, warm, val_n):
    import torch
    from torch.utils.data import DataLoader
    from experiments.run_ablation import CONFIGS, Nets, seed_all, ValSet, validate, EMA
    from models.losses import CombinedLoss
    from utils.dataloader import OnlineDegradationDataset
    dev = torch.device("mps")
    cfg = CONFIGS[cfg_name]
    ds = OnlineDegradationDataset(os.path.join(ROOT, "data", "train_clean"), crop_size=256, augment=True,
                                  severity="mixed", file_list=os.path.join(ROOT, "data", "splits", "train.txt"))
    g = torch.Generator(); g.manual_seed(42); seed_all(42)
    batches = []
    for b in DataLoader(ds, batch_size=4, shuffle=True, drop_last=True, num_workers=0, generator=g):
        batches.append(b)
        if len(batches) >= steps:
            break
    seed_all(42)
    nets = Nets(cfg["translators"], cfg["gate_mode"]).to(dev)
    params = [p for p in nets.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    crit = CombinedLoss(dev, mode="psnr", vgg_normalize=True)
    ema = EMA(nets, 0.999)
    times, peak, nonfinite = [], 0, 0
    for i, (deg, clean, sev) in enumerate(batches):
        deg, clean, sev = deg.to(dev), clean.to(dev), sev.to(dev)
        torch.mps.synchronize(); t = time.time()
        out, ps, gt = nets.restore(deg, False)
        loss = crit(out, clean, pred_severity=ps, target_severity=sev)[0]
        nonfinite += int(not torch.isfinite(loss))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); ema.update(nets, i + 1)
        torch.mps.synchronize(); times.append(time.time() - t)
        peak = max(peak, torch.mps.driver_allocated_memory())
    steady = times[warm:]
    t = time.time(); vset = ValSet(val_n); t_build = time.time() - t
    t = time.time(); validate(ema.shadow, vset, cfg, dev); t_val = time.time() - t
    peak = max(peak, torch.mps.driver_allocated_memory())
    s_step = sum(steady) / len(steady)
    total_s = 20000 * s_step + 20 * t_val + t_build
    print(json.dumps(dict(cfg=cfg_name, train_s_per_step=s_step, train_it_per_s=1 / s_step, warmup_s=sum(times[:warm]),
                          nonfinite_losses=nonfinite, peak_mps_driver_GB=peak / 1e9, val_n_per_severity=val_n,
                          valset_build_s=t_build, validation_pass_s=t_val,
                          est_hours_20k_steps_20_vals=total_s / 3600)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--val_n", type=int, default=250)
    ap.add_argument("--one", default=None)
    a = ap.parse_args()
    if a.one:
        one(a.one, a.steps, a.warmup_steps, a.val_n)
        sys.exit(0)
    rows = []
    for c in ("Tfull", "Tnone"):   # fresh process per model -> clean memory measurement
        o = subprocess.check_output([sys.executable, __file__, "--one", c, "--steps", str(a.steps),
                                     "--warmup_steps", str(a.warmup_steps), "--val_n", str(a.val_n)],
                                    cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        rows.append(json.loads(o)); print(o, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(args=vars(a), rows=rows, total_est_hours=sum(r["est_hours_20k_steps_20_vals"] for r in rows)),
              open(os.path.join(OUT, "preflight_profile.json"), "w"), indent=1)
    print("total estimated hours (both models, sequential):", round(sum(r["est_hours_20k_steps_20_vals"] for r in rows), 2))
