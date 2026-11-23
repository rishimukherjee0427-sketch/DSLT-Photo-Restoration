"""
50-step MPS profiler: FP32 vs torch.amp.autocast(device_type="mps") (fp16 / bf16).

    python experiments/profile_mps.py --config C --crop 256 --steps 50 --warmup_steps 10

Measures steady-state s/step (warm-up steps excluded), peak MPS driver memory, NaN/Inf count,
loss trajectory, and gradient agreement with FP32 on an identical batch (cosine similarity).
Writes results/v7_corrected/profile/profile_{config}_{crop}.json (+ manifest).
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

import torch
from torch.utils.data import DataLoader

from experiments.run_ablation import CONFIGS, Nets, seed_all
from models.losses import CombinedLoss
from utils.dataloader import OnlineDegradationDataset
from utils.manifest import write_manifest


def flat_grad(params):
    return torch.cat([p.grad.detach().float().flatten() for p in params if p.grad is not None])


def run(mode, args, batches, device):
    seed_all(args.seed)
    nets = Nets().to(device)
    params = [p for p in nets.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    cfg = CONFIGS[args.config]
    crit = CombinedLoss(device, mode="psnr", w_gate_sup=cfg["w_gate_sup"],
                        w_gate_sparsity=cfg["w_gate_sparsity"], vgg_normalize=True)
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[mode]
    times, losses, nonfinite, grad0 = [], [], 0, None
    torch.mps.empty_cache()
    base_mem = torch.mps.driver_allocated_memory()
    peak = 0
    try:
        for i, (deg, clean, sev) in enumerate(batches):
            deg, clean, sev = deg.to(device), clean.to(device), sev.to(device)
            torch.mps.synchronize(); t = time.time()
            opt.zero_grad(set_to_none=True)
            if dtype is None:
                out, ps, g = nets.restore(deg, cfg["use_gating"])
                loss = crit(out, clean, pred_severity=ps, target_severity=sev, pred_gates=g)[0]
            else:
                with torch.amp.autocast(device_type="mps", dtype=dtype):
                    out, ps, g = nets.restore(deg, cfg["use_gating"])
                    loss = crit(out.float(), clean, pred_severity=ps.float(), target_severity=sev,
                                pred_gates=g.float())[0]
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            loss.backward()
            if i == 0:
                grad0 = flat_grad(params).cpu()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            torch.mps.synchronize()
            times.append(time.time() - t); losses.append(float(loss))
            peak = max(peak, torch.mps.driver_allocated_memory())
    except Exception as e:  # noqa: BLE001
        return dict(mode=mode, error=f"{type(e).__name__}: {e}"[:500]), None
    w = args.warmup_steps
    steady = times[w:] if len(times) > w else times
    res = dict(mode=mode, steps_completed=len(times), nonfinite_losses=nonfinite,
               warmup_s=sum(times[:w]), steady_s_per_step=sum(steady) / max(1, len(steady)),
               steady_it_per_s=len(steady) / max(1e-9, sum(steady)),
               peak_mps_driver_gb=peak / 1e9, base_mps_driver_gb=base_mem / 1e9,
               loss_first=losses[0] if losses else None, loss_last=losses[-1] if losses else None,
               loss_mean_last10=sum(losses[-10:]) / max(1, len(losses[-10:])))
    del nets, opt, crit
    torch.mps.empty_cache()
    return res, grad0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=list(CONFIGS))
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=["fp32", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("mps")

    ds = OnlineDegradationDataset(os.path.join(ROOT, "data", "train_clean"), crop_size=args.crop,
                                  augment=True, severity="mixed",
                                  file_list=os.path.join(ROOT, "data", "splits", "train.txt"))
    g = torch.Generator(); g.manual_seed(args.seed); seed_all(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, generator=g)
    batches = []
    for b in dl:  # identical batches for every precision mode
        batches.append(b)
        if len(batches) >= args.steps:
            break

    results, grads = {}, {}
    for m in args.modes:
        r, gr = run(m, args, batches, device)
        results[m], grads[m] = r, gr
        print(json.dumps(r), flush=True)
    if grads.get("fp32") is not None:
        ref = grads["fp32"]
        for m, gr in grads.items():
            if m != "fp32" and gr is not None:
                results[m]["grad_cosine_vs_fp32_step0"] = float(torch.nn.functional.cosine_similarity(ref, gr, dim=0))
                results[m]["grad_relnorm_diff_vs_fp32_step0"] = float((ref - gr).norm() / ref.norm())
                results[m]["speedup_vs_fp32"] = results["fp32"]["steady_s_per_step"] / results[m]["steady_s_per_step"] \
                    if "steady_s_per_step" in results[m] else None

    out_dir = os.path.join(ROOT, "results", "v7_corrected", "profile")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"profile_{args.config}_{args.crop}.json")
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=1)
    write_manifest(os.path.join(out_dir, f"profile_{args.config}_{args.crop}_manifest.json"), kind="profile",
                   config=vars(args), metrics=results)
    print("->", out)


if __name__ == "__main__":
    main()
