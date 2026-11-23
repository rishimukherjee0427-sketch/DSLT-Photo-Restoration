"""
Unified benchmark harness (PSNR / SSIM / LPIPS) on held-out, deterministic degradations.

Metric definitions (ONE implementation for every model):
  * PSNR : per image, output clipped to [0,1], 10*log10(1/MSE) over all pixels/channels
           (float64), capped at PSNR_CAP=50 dB per image, then mean over images. The cap matters
           because ~3% of 'mild'/'medium' samples are left untouched by the random degradation
           (MSE=0 -> inf dB, which would otherwise inflate the mean by ~1.6 dB at a 100 dB cap).
           Also reported: psnr_median and n_pristine (images whose input == clean).
  * SSIM : skimage.metrics.structural_similarity(clean, out, data_range=1.0, channel_axis=2)
           (default 7x7 uniform window, same call as experiments/evaluate.py), mean over images.
  * LPIPS: `lpips` package, net='alex' by default, inputs scaled to [-1, 1], mean over images
           (lower is better).
  Outputs are NOT quantised to 8 bit before scoring (clip only).

Inputs: data/benchmark_cache/{split}/{severity}/*.png (8-bit PNG, rendered once by
experiments/make_benchmark.py from OnlineDegradationDataset(deterministic=True,
seed=0, crop_size=256 centre crop)), clean refs in data/benchmark_cache/{split}/clean/.
Splits come from data/splits/{train,val,test}.txt. If the cache is missing it is rendered
automatically (identical result, it is deterministic).

Restore function contract: restore_fn(x) -> y, x/y float tensors (B,3,H,W) in [0,1] on
`device`; y has the same shape (values may be slightly outside [0,1]; they are clipped).
"""
import os
import sys
import ssl
import gc
import time

# Memory-lean MPS defaults (never 0.0 = unlimited); must be set before torch initialises MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import cv2
import numpy as np
import torch

ssl._create_default_https_context = ssl._create_unverified_context  # lpips/torchvision weight download

from skimage.metrics import structural_similarity as _ssim_fn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_DIR = os.path.join(ROOT, "data", "train_clean")
SPLIT_DIR = os.path.join(ROOT, "data", "splits")
CACHE_ROOT = os.path.join(ROOT, "data", "benchmark_cache")
SEVERITIES = ("mild", "medium", "severe")


# ------------------------------------------------------------------ data / cache
def cache_dir(split, severity=None, full_res=False):
    return os.path.join(CACHE_ROOT, split + ("_full" if full_res else ""), severity or "clean")


def split_file(split):
    return os.path.join(SPLIT_DIR, split + ".txt")


def build_dataset(split, severity, crop_size=256, seed=0):
    """Deterministic dataset for a split/severity (same inputs for every model)."""
    from utils.dataloader import OnlineDegradationDataset
    return OnlineDegradationDataset(CLEAN_DIR, crop_size=crop_size, augment=False, severity=severity,
                                    file_list=split_file(split), deterministic=True, seed=seed)


def tensor_to_u8(t):
    """(3,H,W) float [0,1] -> (H,W,3) uint8 RGB."""
    a = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def render_cache(split, severities=SEVERITIES, crop_size=256, full_res=False, seed=0, overwrite=False):
    crop = None if full_res else crop_size
    ds0 = build_dataset(split, "mild", crop, seed)
    cdir = cache_dir(split, None, full_res)
    os.makedirs(cdir, exist_ok=True)
    for i, fn in enumerate(ds0.filenames):
        out = os.path.join(cdir, os.path.splitext(fn)[0] + ".png")
        if overwrite or not os.path.exists(out):
            _, clean, _ = ds0[i]
            cv2.imwrite(out, cv2.cvtColor(tensor_to_u8(clean), cv2.COLOR_RGB2BGR))
    for sev in severities:
        ds = build_dataset(split, sev, crop, seed)
        sdir = cache_dir(split, sev, full_res)
        os.makedirs(sdir, exist_ok=True)
        t0 = time.time()
        for i, fn in enumerate(ds.filenames):
            out = os.path.join(sdir, os.path.splitext(fn)[0] + ".png")
            if overwrite or not os.path.exists(out):
                deg, _, _ = ds[i]
                cv2.imwrite(out, cv2.cvtColor(tensor_to_u8(deg), cv2.COLOR_RGB2BGR))
        print(f"cache {split}/{sev}: {len(ds)} images, {time.time()-t0:.1f}s -> {sdir}")


def _read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def cached_paths(split, severity, n_images=None, full_res=False):
    """Returns (names, degraded_paths, clean_paths); renders the cache if incomplete. Images are
    streamed from disk by evaluate_model (nothing is held in RAM)."""
    names = [ln.strip() for ln in open(split_file(split)) if ln.strip()]
    if n_images:
        names = names[:n_images]
    stems = [os.path.splitext(n)[0] + ".png" for n in names]
    sdir, cdir = cache_dir(split, severity, full_res), cache_dir(split, None, full_res)
    dp = [os.path.join(sdir, t) for t in stems]
    cp = [os.path.join(cdir, t) for t in stems]
    if not all(os.path.exists(x) for x in dp + cp):
        render_cache(split, (severity,), full_res=full_res)
    return names, dp, cp


def load_cached(split, severity, n_images=None, full_res=False):
    """Convenience: (names, degraded list[uint8 HxWx3], clean list[uint8 HxWx3]) fully in RAM."""
    names, dp, cp = cached_paths(split, severity, n_images, full_res)
    return names, [_read_rgb(x) for x in dp], [_read_rgb(x) for x in cp]


# ----------------------------------------------------------------------- metrics
PSNR_CAP = 50.0


def psnr_np(out, ref, cap=PSNR_CAP):
    """out, ref float arrays in [0,1], HxWxC. out is clipped; result capped at `cap` dB."""
    mse = np.mean((np.clip(out, 0, 1).astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float(cap) if mse < 10 ** (-cap / 10) else float(10.0 * np.log10(1.0 / mse))


def ssim_np(out, ref):
    return float(_ssim_fn(ref.astype(np.float32), np.clip(out, 0, 1).astype(np.float32),
                          data_range=1.0, channel_axis=2))


_LPIPS = {}


def get_lpips(net="alex", device="cpu"):
    key = (net, str(device))
    if key not in _LPIPS:
        import lpips
        m = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _LPIPS[key] = m
    return _LPIPS[key]


@torch.no_grad()
def lpips_batch(out_t, ref_t, net="alex", device="cpu"):
    """out_t/ref_t: (B,3,H,W) in [0,1]. Returns np.array (B,)."""
    fn = get_lpips(net, device)
    o = (out_t.clamp(0, 1).to(device) * 2 - 1)
    r = (ref_t.to(device) * 2 - 1)
    return fn(o, r).flatten().float().cpu().numpy()


# ---------------------------------------------------------------- restore helpers
def pad_to_multiple(restore_fn, multiple=16, mode="reflect"):
    """Wrap restore_fn so inputs of any HxW are reflect-padded to a multiple of `multiple` and cropped back."""
    def wrapped(x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % multiple, (-w) % multiple
        if ph == 0 and pw == 0:
            return restore_fn(x)
        xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)
        return restore_fn(xp)[..., :h, :w]
    return wrapped


def identity_fn(x):
    return x


def bicubic_fn(x, factor=2):
    """Sanity baseline: bicubic down-by-`factor` then up (pure interpolation, no restoration)."""
    h, w = x.shape[-2:]
    d = torch.nn.functional.interpolate(x, size=(h // factor, w // factor), mode="bicubic", align_corners=False)
    return torch.nn.functional.interpolate(d, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)


def count_params(*modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


# ---------------------------------------------------------------------- evaluate
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _stack(imgs):
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0


def _free(device):
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_model(restore_fn, split="val", severities=SEVERITIES, n_images=None, device=None,
                   batch_size=4, lpips_net="alex", full_res=False, return_per_image=False,
                   metrics=("psnr", "ssim", "lpips")):
    """
    Evaluate restore_fn on the cached deterministic benchmark (memory-lean: images are streamed
    from disk in small batches; caches are freed between severities).

    Returns dict[severity] -> {psnr, ssim, lpips, *_std, psnr_median, n, n_pristine, time_s,
                               (per_image: {name: [..], psnr: [..], ssim: [..], lpips: [..]})}
    Images of different sizes are processed in separate batches (equal shapes are batched).
    """
    device = device or get_device()
    results = {}
    for sev in severities:
        names, dpaths, cpaths = cached_paths(split, sev, n_images, full_res)
        P, S, L = [], [], []
        pristine = 0
        t0 = time.time()
        i = 0
        while i < len(dpaths):
            deg_b = [_read_rgb(dpaths[i])]
            cln_b = [_read_rgb(cpaths[i])]
            j = i + 1
            while j < len(dpaths) and j - i < batch_size:
                d, c = _read_rgb(dpaths[j]), _read_rgb(cpaths[j])
                if d.shape != deg_b[0].shape:
                    break
                deg_b.append(d); cln_b.append(c); j += 1
            pristine += sum(int(np.array_equal(a, b)) for a, b in zip(deg_b, cln_b))
            x = _stack(deg_b).to(device)
            y = restore_fn(x)
            if y.shape != x.shape:
                raise ValueError(f"restore_fn changed shape {tuple(x.shape)} -> {tuple(y.shape)}")
            y = y.detach().float().clamp(0, 1).cpu()
            del x
            ref = _stack(cln_b)
            yn, rn = y.permute(0, 2, 3, 1).numpy(), ref.permute(0, 2, 3, 1).numpy()
            for k in range(j - i):
                if "psnr" in metrics:
                    P.append(psnr_np(yn[k], rn[k]))
                if "ssim" in metrics:
                    S.append(ssim_np(yn[k], rn[k]))
            if "lpips" in metrics:
                L.extend(lpips_batch(y, ref, lpips_net, device).tolist())
            del y, ref, yn, rn
            i = j
        if str(device).startswith("mps"):
            torch.mps.synchronize()
        r = {"n": len(dpaths), "time_s": round(time.time() - t0, 2), "n_pristine": pristine}
        for key, arr in (("psnr", P), ("ssim", S), ("lpips", L)):
            if arr:
                r[key] = float(np.mean(arr))
                r[key + "_std"] = float(np.std(arr))
        if P:
            r["psnr_median"] = float(np.median(P))
        if return_per_image:
            r["per_image"] = {"name": names, "psnr": P, "ssim": S, "lpips": L}
        results[sev] = r
        _free(device)
    return results
