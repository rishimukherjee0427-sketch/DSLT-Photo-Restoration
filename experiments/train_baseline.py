"""
Generic trainer for the published-architecture baselines (Restormer, PromptIR)
under the same protocol as DSLT v5: same data pipeline (OnlineDegradationDataset),
same loss (CombinedLoss mode='psnr'), batch 4, 256x256 crops, AdamW 2e-4 + cosine,
grad-clip 1.0, optional EMA.  The training budget is counted in OPTIMIZER STEPS.

Examples
  # smoke test (20 steps, current dataset API)
  python experiments/train_baseline.py --model restormer --size tiny --smoke
  # final study (template; split files + mixed severity)
  python experiments/train_baseline.py --model promptir --size small --iters 100000 \
      --severity mixed --file_list data/splits/train.txt \
      --val_file_list data/splits/val.txt --val_every 2000 --ema --resume
Outputs (results/baselines/):
  {model}_{size}{tag}_best.pth   best-val weights (EMA weights if --ema) + metadata
  {model}_{size}{tag}_last.pth   full resumable state
  {model}_{size}{tag}_history.json
"""
import os
import sys
import json
import time
import math
import argparse
import copy

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from models.baselines import build_baseline, set_grad_checkpoint
from models.losses import CombinedLoss
from utils.dataloader import OnlineDegradationDataset

OUT_DIR = "results/baselines"
SEVERITIES = ['mild', 'medium', 'severe']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset(clean_dir, severity, crop, augment, file_list=None):
    """severity 'mixed' -> ConcatDataset of one dataset per severity (equal mix when shuffled)."""
    def one(sev):
        kw = dict(crop_size=crop, augment=augment, severity=sev)
        if file_list:
            kw['file_list'] = file_list
        return OnlineDegradationDataset(clean_dir, **kw)
    if severity == 'mixed':
        return ConcatDataset([one(s) for s in SEVERITIES])
    return one(severity)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # warm-up so short runs are not stuck at init
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


@torch.no_grad()
def validate(model, val_loader, device):
    from skimage.metrics import structural_similarity as ssim_fn
    model.eval()
    ps, ss = [], []
    for deg, clean, _ in val_loader:
        deg, clean = deg.to(device), clean.to(device)
        out = model(deg).clamp(0, 1)
        mse = ((out - clean) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        ps += (10 * torch.log10(1.0 / mse)).cpu().tolist()
        o, c = out.permute(0, 2, 3, 1).cpu().numpy(), clean.permute(0, 2, 3, 1).cpu().numpy()
        ss += [ssim_fn(o[i], c[i], data_range=1.0, channel_axis=2) for i in range(len(o))]
    return float(np.mean(ps)), float(np.mean(ss))


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['restormer', 'promptir'], required=True)
    ap.add_argument('--size', choices=['tiny', 'small', 'base'], default='small')
    ap.add_argument('--iters', type=int, default=None, help='optimizer steps (budget)')
    ap.add_argument('--epochs', type=int, default=None, help='alt. budget: epochs over the train set (-> steps)')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--min_lr_ratio', type=float, default=0.005)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--grad_ckpt', action='store_true', help='activation checkpointing (lower memory, slower)')
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--clean_dir', default='data/train_clean')
    ap.add_argument('--file_list', default=None, help='train split txt (needs new dataset API)')
    ap.add_argument('--severity', default='mild', help="mild|medium|severe|heavy|mixed")
    ap.add_argument('--val_clean_dir', default=None, help='defaults to --clean_dir')
    ap.add_argument('--val_file_list', default=None)
    ap.add_argument('--val_severity', default=None, help='defaults to --severity')
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--val_samples', type=int, default=96)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--resume', action='store_true', help='resume from {name}_last.pth if present')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='20 iters, mild, no val, no file_list, no saving')
    args = ap.parse_args()

    device = get_device()
    if args.smoke:
        args.iters, args.severity, args.file_list = 20, 'mild', None
        args.val_file_list, args.warmup, args.log_every, args.workers = None, 5, 2, 0
        args.val_every = 0
    name = f"{args.model}_{args.size}{args.tag}"
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path, last_path, hist_path = [f"{OUT_DIR}/{name}_{k}" for k in ('best.pth', 'last.pth', 'history.json')]

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- data ----
    train_set = build_dataset(args.clean_dir, args.severity, args.crop, True, args.file_list)
    if args.iters is None:
        if args.epochs is None:
            ap.error('give --iters or --epochs')
        args.iters = args.epochs * math.ceil(len(train_set) / args.batch_size)
    total = args.iters
    n_resume_seed = args.seed
    val_loader = None
    if args.val_file_list or (args.val_every and args.val_clean_dir):
        vset = build_dataset(args.val_clean_dir or args.clean_dir, args.val_severity or args.severity,
                             args.crop, False, args.val_file_list)
        idx = np.linspace(0, len(vset) - 1, min(args.val_samples, len(vset))).astype(int).tolist()
        val_loader = DataLoader(Subset(vset, idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- model ----
    model = build_baseline(args.model, args.size).to(device)
    if args.grad_ckpt:
        set_grad_checkpoint(model, True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {n_params/1e6:.2f}M params | device={device} | budget={total} steps "
          f"(bs={args.batch_size}, crop={args.crop}) | severity={args.severity}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(total, args.warmup, args.min_lr_ratio))
    ema = EMA(model, args.ema_decay) if args.ema else None
    criterion = CombinedLoss(device=device, mode='psnr')

    step, best_psnr, history = 0, -1.0, []
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location='cpu')
        model.load_state_dict(ck['model_raw']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        if ema is not None and ck.get('ema') is not None:
            ema.shadow.load_state_dict(ck['ema'])
        step, best_psnr, history = ck['step'], ck['best_psnr'], ck['history']
        n_resume_seed = args.seed + step
        print(f"Resumed from {last_path} at step {step} (best val PSNR {best_psnr:.2f})", flush=True)
    torch.manual_seed(n_resume_seed)

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=False)
    stream = infinite(loader)

    def save(path, best=False):
        if args.smoke:
            return
        eval_model = ema.shadow if ema is not None else model
        meta = dict(step=step, best_psnr=best_psnr, args=vars(args), n_params=n_params,
                    model_name=args.model, size=args.size, uses_ema=ema is not None)
        if best:
            torch.save(dict(model=eval_model.state_dict(), **meta), path)
        else:
            torch.save(dict(model=eval_model.state_dict(), model_raw=model.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), ema=ema.shadow.state_dict() if ema else None,
                            history=history, **meta), path)

    def save_history():
        if not args.smoke:
            with open(hist_path, 'w') as f:
                json.dump(dict(name=name, n_params=n_params, args=vars(args), history=history), f, indent=1)

    model.train()
    t0, run = time.time(), dict(loss=0.0, char=0.0, ssim=0.0, n=0)
    first_loss = None
    while step < total:
        deg, clean, _ = next(stream)
        deg, clean = deg.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        out = model(deg)
        total_loss, l_char, _, l_ssim, _, _ = criterion(out, clean)
        if not torch.isfinite(total_loss):
            print(f"!! non-finite loss at step {step}; skipping batch", flush=True)
            opt.zero_grad(set_to_none=True); step += 1; sched.step(); continue
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step(); step += 1
        if ema is not None:
            ema.update(model, step)

        lv = total_loss.item()
        first_loss = lv if first_loss is None else first_loss
        run['loss'] += lv; run['char'] += l_char.item(); run['ssim'] += l_ssim.item(); run['n'] += 1
        if step % args.log_every == 0 or step == total:
            n = run['n']; el = time.time() - t0
            rec = dict(step=step, loss=run['loss'] / n, charb=run['char'] / n, ssim_loss=run['ssim'] / n,
                       lr=sched.get_last_lr()[0], elapsed_s=el)
            history.append(rec)
            print(f"[{name}] step {step}/{total} loss {rec['loss']:.4f} (charb {rec['charb']:.4f}, "
                  f"1-ssim {rec['ssim_loss']:.3f}) lr {rec['lr']:.2e} | {el/60:.1f} min elapsed",
                  flush=True)
            run = dict(loss=0.0, char=0.0, ssim=0.0, n=0)

        if val_loader is not None and args.val_every and (step % args.val_every == 0 or step == total):
            ev = ema.shadow if ema is not None else model
            psnr, ssim = validate(ev, val_loader, device)
            model.train()
            history.append(dict(step=step, val_psnr=psnr, val_ssim=ssim))
            tag = ''
            if psnr > best_psnr:
                best_psnr, tag = psnr, ' *best*'
                save(best_path, best=True)
            print(f"[{name}] VAL step {step}: PSNR {psnr:.2f} dB  SSIM {ssim:.4f}{tag}", flush=True)
            save_history()
        if step % args.save_every == 0 and step < total:
            save(last_path); save_history()

    save(last_path); save_history()
    if val_loader is None and not args.smoke:
        # no validation: keep the final weights as "best"
        save(best_path, best=True)
    if device.type == 'mps':
        torch.mps.empty_cache()
    print(f"[{name}] done: {step} steps in {(time.time()-t0)/60:.1f} min | first loss {first_loss:.4f} "
          f"-> last logged {[h for h in history if 'loss' in h][-1]['loss']:.4f}", flush=True)


if __name__ == '__main__':
    main()
