"""
Build the held-out benchmark for DSLT / baseline comparison.

Step 1  (--splits)  Deterministic train/val/test split of data/train_clean
        -> data/splits/{train,val,test,excluded}.txt + data/splits/split_info.json
Step 2  (--cache)   Pre-render deterministic degraded inputs (8-bit PNG) so every
        model is evaluated on byte-identical inputs:
        data/benchmark_cache/{split}/clean/*.png
        data/benchmark_cache/{split}/{severity}/*.png      (severity in mild/medium/severe)
        (full-res variant, --full-res: data/benchmark_cache/{split}_full/...)

SPLIT POLICY (seed 42)
  * data/train_clean is the ONLY clean pool. data/clean_raw is its source (every
    clean_raw image has a flip-aware pHash match in train_clean), raw/ + processed/
    contain real old photos/video frames with no clean GT -> there are no genuinely
    unseen clean images on disk (unzipped); see report for zip contents.
  * train_clean has heavy duplication (jpg+png copies of the same picture, with
    random flips): images are clustered by exact md5 OR (flip-aware pHash<=8/64 AND
    flip-aware dHash<=8/64) (union-find; 'OR' of the two hashes chained into a 700-image blob), so near-duplicates stay together.
  * Clusters are shuffled with random.Random(42). The first 250 clusters -> val and
    the next 250 -> test, ONE representative image per cluster (smallest filename).
    All other members of val/test clusters go to excluded.txt (never trained on, never
    evaluated on: they would leak). Every remaining cluster -> train.
  * IMPORTANT: results/best_model_v5*.pth, ema_weights_v5.pth were trained on ALL of
    train_clean, i.e. they have seen every val/test image. They must be retrained on
    data/splits/train.txt before any number on val/test is a valid generalisation
    figure.

Usage:
  python experiments/make_benchmark.py --splits --cache
"""
import os
import sys
import json
import argparse
import hashlib
import collections
import random
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

CLEAN_DIR = os.path.join("data", "train_clean")
SPLIT_DIR = os.path.join("data", "splits")
N_VAL, N_TEST, SEED = 250, 250, 42
PHASH_T, DHASH_T = 8, 8


# --------------------------------------------------------------------- hashing
def _phash(g):
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:]))


def _dhash(g):
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def _hash_one(path):
    raw = open(path, "rb").read()
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    variants = [g, g[:, ::-1], g[::-1], g[::-1, ::-1]]  # identity, hflip, vflip, both
    return (hashlib.md5(raw).hexdigest(),
            np.stack([_phash(v) for v in variants]),
            np.stack([_dhash(v) for v in variants]))


_POP = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint8)


def _flip_aware_min_dist(H):
    """H: (n, 4, 8) packed hashes of 4 flips. Returns (n, n) min Hamming distance over flips."""
    n = H.shape[0]
    base = H[:, 0, :]
    out = np.full((n, n), 255, dtype=np.uint8)
    for k in range(4):
        for i0 in range(0, n, 500):
            x = base[i0:i0 + 500, None, :] ^ H[None, :, k, :]
            out[i0:i0 + 500] = np.minimum(out[i0:i0 + 500], _POP[x].sum(-1).astype(np.uint8))
    return out


def cluster_images(names, clean_dir=CLEAN_DIR):
    from multiprocessing import Pool
    paths = [os.path.join(clean_dir, n) for n in names]
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        res = pool.map(_hash_one, paths, chunksize=32)
    md5 = [r[0] for r in res]
    P = np.stack([r[1] for r in res])
    D = np.stack([r[2] for r in res])
    dp = _flip_aware_min_dist(P)
    dd = _flip_aware_min_dist(D)
    n = len(names)
    np.fill_diagonal(dp, 255)
    np.fill_diagonal(dd, 255)
    link = (dp <= PHASH_T) & (dd <= DHASH_T)   # both hashes must agree (OR chains into a 700-image blob)
    link = link | link.T
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ii, jj = np.nonzero(np.triu(link, 1))
    for a, b in zip(ii, jj):
        union(int(a), int(b))
    first = {}
    for i, h in enumerate(md5):
        if h in first:
            union(i, first[h])
        else:
            first[h] = i
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    stats = {
        "n_images": n,
        "exact_md5_dup_groups": sum(1 for v in collections.Counter(md5).values() if v > 1),
        "phash_pairs_le_%d" % PHASH_T: int(((dp <= PHASH_T) & np.triu(np.ones_like(dp, dtype=bool), 1)).sum()),
        "dhash_pairs_le_%d" % DHASH_T: int(((dd <= DHASH_T) & np.triu(np.ones_like(dd, dtype=bool), 1)).sum()),
        "linked_pairs_phash_and_dhash": int((link & np.triu(np.ones_like(link, dtype=bool), 1)).sum()),
    }
    return list(clusters.values()), stats


# ---------------------------------------------------------------------- splits
def make_splits(clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR):
    names = sorted(f for f in os.listdir(clean_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    t0 = time.time()
    clusters, stats = cluster_images(names, clean_dir)
    clusters = [sorted(c) for c in clusters]
    clusters.sort(key=lambda c: c[0])           # canonical order before shuffling
    sizes = [len(c) for c in clusters]
    rng = random.Random(SEED)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    val_c = order[:N_VAL]
    test_c = order[N_VAL:N_VAL + N_TEST]
    train_c = order[N_VAL + N_TEST:]
    pick = lambda cs: sorted(names[clusters[c][0]] for c in cs)
    val, test = pick(val_c), pick(test_c)
    excluded = sorted(names[i] for c in val_c + test_c for i in clusters[c][1:])
    train = sorted(names[i] for c in train_c for i in clusters[c])
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) + len(excluded) == len(names)
    os.makedirs(split_dir, exist_ok=True)
    for k, v in (("train", train), ("val", val), ("test", test), ("excluded", excluded)):
        with open(os.path.join(split_dir, k + ".txt"), "w") as f:
            f.write("\n".join(v) + "\n")
    info = dict(stats)
    info.update({
        "seed": SEED, "phash_threshold": PHASH_T, "dhash_threshold": DHASH_T,
        "n_clusters": len(clusters),
        "n_clusters_size_gt1": sum(s > 1 for s in sizes),
        "images_in_clusters_size_gt1": int(sum(s for s in sizes if s > 1)),
        "largest_cluster": max(sizes),
        "cluster_size_hist": {str(k): sizes.count(k) for k in sorted(set(sizes))},
        "sizes": {"train": len(train), "val": len(val), "test": len(test), "excluded": len(excluded)},
        "note": "v5 checkpoints in results/ were trained on ALL train_clean -> contaminated for val/test.",
    })
    with open(os.path.join(split_dir, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print("splits done in %.1fs" % (time.time() - t0))
    return info


# ----------------------------------------------------------------------- cache
def make_cache(splits=("val", "test"), severities=("mild", "medium", "severe"),
               crop_size=256, full_res=False, seed=0, overwrite=False):
    from utils.benchmark import render_cache
    for split in splits:
        render_cache(split, severities, crop_size, full_res, seed, overwrite)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", action="store_true", help="(re)create data/splits/*.txt")
    ap.add_argument("--cache", action="store_true", help="render degraded benchmark cache")
    ap.add_argument("--splits-to-cache", nargs="+", default=["val", "test"])
    ap.add_argument("--severities", nargs="+", default=["mild", "medium", "severe"])
    ap.add_argument("--full-res", action="store_true", help="full resolution instead of 256 centre crop")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.splits:
        make_splits()
    if a.cache:
        make_cache(a.splits_to_cache, tuple(a.severities), full_res=a.full_res, overwrite=a.overwrite)
