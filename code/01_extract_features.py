# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
"""Pass 1: standardize images (decontaminate size/aspect/format provenance),
compute md5 + perceptual hash, and extract handcrafted color/texture features.
Caches everything to /home/claude/artifacts so later steps are cheap."""
import os, io, json, time, hashlib
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew
import imagehash

ROOT = "/home/claude"
OUT = os.path.join(ROOT, "artifacts"); os.makedirs(OUT, exist_ok=True)
DIRS = {1: os.path.join(ROOT, "data/iog"), 0: os.path.join(ROOT, "data/non_iog")}
S = 224  # standardized square size

def list_images():
    rows = []
    for label, d in DIRS.items():
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append((os.path.join(d, f), f, label))
    return rows

def standardize(path):
    """Return (std_uint8 HxWx3, phash_hex, md5_hex). All images pushed through an
    identical resize + JPEG q90 re-encode so format/size/aspect can't leak label."""
    raw = open(path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((S, S), Image.BICUBIC)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    im2 = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    ph = str(imagehash.phash(im2, hash_size=8))  # 64-bit pHash
    return np.asarray(im2, dtype=np.uint8), ph, md5

def color_feats(arr):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 3D color histograms (8 bins/channel), L1-normalized
    h_rgb = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).flatten()
    h_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    h_rgb /= (h_rgb.sum() + 1e-8); h_hsv /= (h_hsv.sum() + 1e-8)
    # color moments in HSV (mean, std, skew per channel)
    mom = []
    for c in range(3):
        ch = hsv[:, :, c].astype(np.float32).flatten()
        mom += [ch.mean(), ch.std(), float(skew(ch)) if ch.std() > 1e-6 else 0.0]
    return np.concatenate([h_rgb, h_hsv, np.array(mom, np.float32)])

def edge_texture_feats(arr):
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean()) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    grad = [float(mag.mean()), float(mag.std())]
    # LBP uniform (P=8,R=1) -> 10-bin normalized histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
    # HOG
    hog_v = hog(gray, orientations=9, pixels_per_cell=(32, 32),
                cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return np.concatenate([[edge_density], grad, lbp_hist.astype(np.float32), hog_v.astype(np.float32)])

def main():
    rows = list_images()
    n = len(rows); print(f"images: {n}", flush=True)
    feats, meta = [], []
    t0 = time.time()
    for i, (path, fname, label) in enumerate(rows):
        try:
            arr, ph, md5 = standardize(path)
            fv = np.concatenate([color_feats(arr), edge_texture_feats(arr)]).astype(np.float32)
        except Exception as e:
            print("ERR", fname, e, flush=True); continue
        feats.append(fv); meta.append((fname, label, md5, ph))
        if (i + 1) % 300 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  ({dt:.0f}s, {dt/(i+1)*1000:.0f} ms/img)", flush=True)
    X = np.vstack(feats).astype(np.float32)
    np.save(os.path.join(OUT, "X_hand.npy"), X)
    import csv
    with open(os.path.join(OUT, "meta.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["filename", "label", "md5", "phash"])
        w.writerows(meta)
    json.dump({"feature_dim": int(X.shape[1]), "n": int(X.shape[0]), "S": S},
              open(os.path.join(OUT, "pass1_info.json"), "w"))
    print(f"DONE  X={X.shape}  meta={len(meta)}  elapsed={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
