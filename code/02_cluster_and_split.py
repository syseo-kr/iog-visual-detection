# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
"""Cluster near-duplicate screenshots by perceptual-hash Hamming distance,
collapse each cluster to one representative (kills train/test template leakage),
and build label-stratified 5-fold CV over representatives."""
import os, csv, json, sys
import numpy as np

OUT = "/home/claude/artifacts"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 10   # pHash Hamming threshold for "near-duplicate"
SEED = 42

# ---- load meta ----
rows = list(csv.DictReader(open(os.path.join(OUT, "meta.csv"))))
n = len(rows)
labels = np.array([int(r["label"]) for r in rows])
# pHash hex -> 64-bit array -> bit matrix (n x 64)
hashes = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
bits = np.unpackbits(hashes.view(np.uint8).reshape(n, 8), axis=1)  # n x 64

# ---- union-find near-duplicate clustering ----
parent = np.arange(n)
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

def cluster_at(thr):
    p = np.arange(n)
    def f(a):
        while p[a] != a:
            p[a] = p[p[a]]; a = p[a]
        return a
    for i in range(n):
        d = (bits[i] != bits[i+1:]).sum(axis=1)      # hamming to all j>i
        near = np.nonzero(d <= thr)[0] + (i + 1)
        for j in near:
            ra, rb = f(i), f(int(j))
            if ra != rb: p[ra] = rb
    roots = np.array([f(i) for i in range(n)])
    _, cid = np.unique(roots, return_inverse=True)
    return cid

# ---- sensitivity table across thresholds ----
print(f"{'T':>3} {'#clusters':>10} {'IOGclu':>7} {'nIOGclu':>8} {'xlabel_clusters':>16}")
for thr in [0, 5, 8, 10, 12, 15]:
    cid = cluster_at(thr)
    k = cid.max() + 1
    # cross-label: clusters containing both labels
    xlab = 0; iog_c = 0; niog_c = 0
    for c in range(k):
        m = labels[cid == c]
        if (m == 1).any() and (m == 0).any(): xlab += 1
        if (m == 1).any(): iog_c += 1
        if (m == 0).any(): niog_c += 1
    print(f"{thr:>3} {k:>10} {iog_c:>7} {niog_c:>8} {xlab:>16}")

# ---- final clustering at chosen T ----
cid = cluster_at(T)
K = cid.max() + 1
print(f"\nChosen T={T}: {K} clusters from {n} images")

# representative per cluster = first index; representative label = majority within cluster
reps, rep_labels, rep_cid = [], [], []
xlabel_imgs = 0
for c in range(K):
    idx = np.nonzero(cid == c)[0]
    m = labels[idx]
    if (m == 1).any() and (m == 0).any(): xlabel_imgs += len(idx)
    lab = int(round(m.mean()))   # majority label
    reps.append(int(idx[0])); rep_labels.append(lab); rep_cid.append(c)
reps = np.array(reps); rep_labels = np.array(rep_labels)
print(f"cross-label clusters touch {xlabel_imgs} images (ambiguous near-dups across classes)")
print(f"clean (deduplicated) set: {len(reps)} images  "
      f"-> IOG {int((rep_labels==1).sum())} / non-IOG {int((rep_labels==0).sum())}")

# ---- stratified 5-fold CV over representatives ----
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold = np.full(len(reps), -1, dtype=int)
for fi, (_, te) in enumerate(skf.split(reps, rep_labels)):
    fold[te] = fi
for fi in range(5):
    m = rep_labels[fold == fi]
    print(f"  fold {fi}: n={len(m)}  IOG={int((m==1).sum())} non-IOG={int((m==0).sum())}")

# ---- save artifacts ----
np.save(os.path.join(OUT, "cluster_id.npy"), cid)          # per-image (aligned to meta.csv / X_hand)
np.save(os.path.join(OUT, "rep_index.npy"), reps)          # indices into the 2519 arrays
np.save(os.path.join(OUT, "rep_labels.npy"), rep_labels)
np.save(os.path.join(OUT, "rep_fold.npy"), fold)
json.dump({"T": T, "n_images": int(n), "n_clusters": int(K),
           "clean_n": int(len(reps)),
           "clean_iog": int((rep_labels==1).sum()),
           "clean_noniog": int((rep_labels==0).sum()),
           "seed": SEED},
          open(os.path.join(OUT, "split_info.json"), "w"), indent=2)
with open(os.path.join(OUT, "clusters.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["filename", "label", "cluster_id"])
    for r, c in zip(rows, cid): w.writerow([r["filename"], r["label"], int(c)])
print("\nsaved: cluster_id, rep_index, rep_labels, rep_fold, split_info, clusters.csv")
