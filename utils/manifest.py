"""
Experiment reproducibility manifests.

    from utils.manifest import write_manifest
    write_manifest("results/v7_corrected/seed123/A_manifest.json", kind="train", config={...},
                   metrics={...}, checkpoint="results/.../A_best.pth", extra={...})

Records: git commit + dirty status, SHA256 of data/splits/{train,val,test}.txt, Python / torch /
torchvision / lpips versions, macOS + MPS info, hardware, and caller-supplied config/metrics.
Written atomically (tmp file + os.replace).
"""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def environment_info():
    import torch
    import torchvision
    try:
        from importlib.metadata import version
        lpips_v = version("lpips")
    except Exception:
        lpips_v = None
    mem = None
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
    except Exception:
        pass
    chip = None
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return dict(
        python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__,
        lpips=lpips_v, platform=platform.platform(), macos=platform.mac_ver()[0],
        mps_available=torch.backends.mps.is_available(), chip=chip, memory_bytes=mem,
        env={k: os.environ.get(k) for k in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                                             "PYTORCH_MPS_LOW_WATERMARK_RATIO")},
    )


def split_hashes():
    out = {}
    for k in ("train", "val", "test"):
        p = os.path.join(ROOT, "data", "splits", f"{k}.txt")
        out[k] = sha256_file(p) if os.path.exists(p) else None
    return out


def write_manifest(path, kind, config=None, metrics=None, checkpoint=None, extra=None, hash_checkpoint=True):
    status = _git("status", "--porcelain")
    man = dict(
        kind=kind,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git=dict(commit=_git("rev-parse", "HEAD"), dirty=bool(status), dirty_files=(status or "").splitlines()[:50]),
        splits_sha256=split_hashes(),
        environment=environment_info(),
        config=config or {},
        metrics=metrics or {},
        checkpoint=None,
        extra=extra or {},
    )
    if checkpoint:
        man["checkpoint"] = dict(path=os.path.relpath(checkpoint, ROOT),
                                 sha256=sha256_file(checkpoint) if (hash_checkpoint and os.path.exists(checkpoint)) else None)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1, default=str)
    os.replace(tmp, path)
    return man
