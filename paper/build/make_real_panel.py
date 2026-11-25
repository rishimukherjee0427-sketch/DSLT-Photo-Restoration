"""Task 5: qualitative panel on real old photographs (no ground truth).

Follows paper/REAL_PHOTO_PANEL_PLAN.md:
- photos = paper/build/out/real_panel_selection.json (fixed before inference; md5-checked here)
- models = Stage-3 seed-42 validation-best checkpoints (full model, encoder-decoder), plus any Stage-3-protocol
  baseline checkpoint for seed 42 present under results/v10_baselines/seed42/ (labelled as such)
- RGB in [0,1]; long side resized to 512 (area); reflect-pad to a multiple of 16; one forward pass, no TTA/tiling;
  crop back. Runs on CPU by default (qualitative only; avoids competing for the GPU).
Outputs: paper/build/out/real_panel/{photo}__{model}.png and figures/fig_real_panel.{pdf,png} (300 dpi).

    python paper/build/make_real_panel.py [--device cpu|mps] [--figure_only]
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = os.path.join(ROOT, "paper", "build", "out")
PANEL = os.path.join(OUT, "real_panel")
SRC = os.path.join(ROOT, "data", "raw", "archive (2)")
S3 = os.path.join(ROOT, "results", "v9_convergence_pilot", "seed42")
BASE = os.path.join(ROOT, "results", "v10_baselines", "seed42")
COLS = [("input", "Input (real scan)"), ("Tfull", "Full model"), ("Tnone", "Encoder--decoder"),
        ("unet", "U-Net"), ("nafnet", "NAFNet"), ("restormer", "Restormer")]


def load_models(device):
    import experiments.stage2 as s2
    ms = {}
    for c in ("Tfull", "Tnone"):
        ck = torch.load(os.path.join(S3, f"{c}_best.pth"), map_location="cpu", weights_only=False)
        enc, dec, dslt = s2.build(ck)
        enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"]); dslt.load_state_dict(ck["dslt"])
        for m in (enc, dec, dslt):
            m.to(device).eval()

        def fn(x, enc=enc, dec=dec, dslt=dslt):
            b, sk = enc(x)
            bn, skt = dslt(b, sk)
            return torch.clamp(x + dec(bn, skt), 0, 1)
        ms[c] = fn
    from experiments.run_baseline_s3 import load_baseline
    for c in ("unet", "nafnet", "restormer"):
        p = os.path.join(BASE, f"{c}_best.pth")
        if os.path.exists(p) and os.path.exists(os.path.join(BASE, f"{c}_manifest.json")):   # finished runs only
            net, _ = load_baseline(p, device)
            ms[c] = net
    return ms


def prep(path):
    im = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = 512 / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return im


@torch.no_grad()
def run(fn, im, device):
    x = torch.from_numpy(im).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    H, W = x.shape[-2:]
    x = F.pad(x, (0, (-W) % 16, 0, (-H) % 16), mode="reflect")
    y = fn(x)[..., :H, :W].clamp(0, 1)
    return (y[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def figure(sel, cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    photos = [(k, f) for k in sel["selected"] for f in sel["selected"][k]]
    fig, ax = plt.subplots(len(photos), len(cols), figsize=(7.16, 1.25 * len(photos)))
    for i, (cat, f) in enumerate(photos):
        for j, (c, lab) in enumerate(cols):
            im = cv2.cvtColor(cv2.imread(os.path.join(PANEL, f"{os.path.splitext(f)[0]}__{c}.png")), cv2.COLOR_BGR2RGB)
            ax[i, j].imshow(im); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(lab, fontsize=6)
            if j == 0:
                ax[i, j].set_ylabel(cat.split("_")[0], fontsize=5)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figures", f"fig_real_panel.{ext}"), dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--figure_only", action="store_true")
    a = ap.parse_args()
    sel = json.load(open(os.path.join(OUT, "real_panel_selection.json")))
    for f, md5 in sel["md5"].items():
        assert hashlib.md5(open(os.path.join(SRC, f), "rb").read()).hexdigest() == md5, f
    os.makedirs(PANEL, exist_ok=True)
    if not a.figure_only:
        device = torch.device(a.device)
        ms = load_models(device)
        for f in sel["md5"]:
            im = prep(os.path.join(SRC, f))
            stem = os.path.splitext(f)[0]
            cv2.imwrite(os.path.join(PANEL, f"{stem}__input.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            for c, fn in ms.items():
                cv2.imwrite(os.path.join(PANEL, f"{stem}__{c}.png"), cv2.cvtColor(run(fn, im, device), cv2.COLOR_RGB2BGR))
            print("done", f, im.shape, flush=True)
    have = {c for c, _ in COLS if os.path.exists(os.path.join(PANEL, f"{os.path.splitext(next(iter(sel['md5'])))[0]}__{c}.png"))}
    figure(sel, [x for x in COLS if x[0] in have])
    print("figure columns:", [c for c, _ in COLS if c in have])


if __name__ == "__main__":
    main()
