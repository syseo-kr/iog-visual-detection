# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
# ============================================================================
# run_clip.py  -- run this on YOUR machine or in Colab (it needs internet to
# download CLIP weights, which the analysis sandbox cannot reach).
#
# Setup (once):
#     pip install open_clip_torch torch pillow numpy
# Then set the two folder paths below and run:
#     python run_clip.py
#
# It produces 3 small files. Upload these back into the chat:
#     clip_zeroshot.csv            (filename, label, p_iog, pred)
#     clip_image_embeddings.npz    (names, labels, embeddings)   ~5 MB
#     clip_text_class_embeddings.npy
# I will align them to the same dedup clusters / CV folds and report
# CLIP zero-shot AND CLIP linear-probe next to the LMMs on identical folds.
# ============================================================================
import os, csv, numpy as np
from PIL import Image
import torch, open_clip

# ---- EDIT THESE TWO PATHS ----
IOG_DIR    = "iog"        # folder of IOG screenshots
NONIOG_DIR = "non_iog"    # folder of non-IOG screenshots
# ------------------------------

MODEL, PRETRAINED = "ViT-B-32", "openai"   # standard light CLIP; for a stronger run use "ViT-L-14"
BATCH = 32

# Zero-shot prompt ensembles (tweak freely; CLIP is English-trained so keep English).
POS_PROMPTS = [
    "a screenshot of an online gambling website",
    "a screenshot of an online sports betting website",
    "a screenshot of an online casino or poker website",
    "a screenshot of an illegal horse racing betting site",
    "a webpage for placing bets and gambling for money",
]
NEG_PROMPTS = [
    "a screenshot of an ordinary website",
    "a screenshot of a normal business or company website",
    "a screenshot of a personal blog or informational website",
    "a screenshot of an online shop or news website",
    "a webpage with no gambling or betting content",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  model={MODEL}/{PRETRAINED}")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
model.eval().to(device)
tokenizer = open_clip.get_tokenizer(MODEL)

def class_text_embedding(prompts):
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(device)).float()
        t = t / t.norm(dim=-1, keepdim=True)
        c = t.mean(dim=0); c = c / c.norm()
    return c
class_emb = torch.stack([class_text_embedding(NEG_PROMPTS),   # index 0 = non-IOG
                         class_text_embedding(POS_PROMPTS)])  # index 1 = IOG
logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0

# gather images
items = []
for label, d in [(1, IOG_DIR), (0, NONIOG_DIR)]:
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            items.append((os.path.join(d, f), os.path.basename(f), label))
print(f"images: {len(items)}")

names, labels, embs, rows = [], [], [], []
for i in range(0, len(items), BATCH):
    chunk = items[i:i+BATCH]
    imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p, _, _ in chunk]).to(device)
    with torch.no_grad():
        e = model.encode_image(imgs).float()
        e = e / e.norm(dim=-1, keepdim=True)
        logits = logit_scale * e @ class_emb.T          # [B, 2]
        prob = logits.softmax(dim=-1)[:, 1]             # P(IOG)
    for (p, base, lab), ev, pr in zip(chunk, e.cpu().numpy(), prob.cpu().numpy()):
        names.append(base); labels.append(lab); embs.append(ev.astype(np.float32))
        rows.append((base, lab, float(pr), int(pr >= 0.5)))
    if (i // BATCH) % 10 == 0:
        print(f"  {min(i+BATCH, len(items))}/{len(items)}")

with open("clip_zeroshot.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["filename", "label", "p_iog", "pred"]); w.writerows(rows)
np.savez_compressed("clip_image_embeddings.npz",
                    names=np.array(names), labels=np.array(labels),
                    embeddings=np.vstack(embs))
np.save("clip_text_class_embeddings.npy", class_emb.cpu().numpy())

acc = np.mean([r[3] == r[1] for r in rows])
print(f"\nDONE. quick zero-shot accuracy (threshold 0.5): {acc*100:.1f}%")
print("Upload: clip_zeroshot.csv, clip_image_embeddings.npz, clip_text_class_embeddings.npy")
