# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
# ============================================================================
# clip_extra.py  -- run on YOUR machine / Colab (needs internet for CLIP weights).
#   pip install open_clip_torch torch pillow numpy
#
# TWO things, controlled by the flags below:
#   1) PROMPT SENSITIVITY on ViT-B/32  -> outputs ONLY text embeddings for several
#      prompt sets. No image re-encoding needed (the analysis side already has your
#      B/32 image embeddings). Fast: a few seconds.
#   2) (optional) ViT-L/14 -> re-encodes all images (different embedding space) and
#      also outputs its prompt-set text embeddings + a zero-shot CSV.
#
# Upload back whatever it produces:
#   clip_text_variants_ViT-B-32.npz                       (always)
#   clip_image_embeddings_ViT-L-14.npz                    (if DO_VIT_L14)
#   clip_text_variants_ViT-L-14.npz                       (if DO_VIT_L14)
#   clip_zeroshot_ViT-L-14.csv                            (if DO_VIT_L14)
# I will evaluate every prompt set (and L/14) on the SAME dedup clusters / CV folds.
# ============================================================================
import os, csv, numpy as np
import torch, open_clip

# ---- flags ----
DO_VIT_L14 = False                 # set True to also run the stronger ViT-L/14
IOG_DIR    = "iog"                 # only needed if DO_VIT_L14 = True
NONIOG_DIR = "non_iog"             # only needed if DO_VIT_L14 = True
BATCH = 32
# ----------------

# Several prompt ensembles to measure sensitivity. row order saved as [NEG, POS].
PROMPT_SETS = {
    "orig": (  # the set used in the first run
        ["a screenshot of an online gambling website",
         "a screenshot of an online sports betting website",
         "a screenshot of an online casino or poker website",
         "a screenshot of an illegal horse racing betting site",
         "a webpage for placing bets and gambling for money"],
        ["a screenshot of an ordinary website",
         "a screenshot of a normal business or company website",
         "a screenshot of a personal blog or informational website",
         "a screenshot of an online shop or news website",
         "a webpage with no gambling or betting content"]),
    "terse": (["gambling website", "betting website", "casino website"],
              ["normal website", "ordinary website", "regular website"]),
    "action": (["a website to deposit money and place bets",
                "a site to bet on sports, casino games or horse racing"],
               ["a website with articles, products, or services",
                "a site to read information or buy products"]),
    "style": (["a flashy website with neon colors, odds and betting buttons",
               "a dark gambling site with chips, cards and jackpot banners"],
              ["a clean ordinary website with text and a simple layout",
               "a plain corporate or personal homepage"]),
    "illegal": (["an illegal unlicensed online gambling website",
                 "an underground betting site operating without a license"],
                ["a legitimate licensed website",
                 "a normal lawful website"]),
    "korean_ctx": (["a Korean online gambling or sports betting website",
                    "a Korean illegal casino or horse-racing betting site"],
                   ["a Korean ordinary company or shopping website",
                    "a Korean news, blog or informational website"]),
}

def text_variants(model_name):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.eval()
    tok = open_clip.get_tokenizer(model_name)
    ls = model.logit_scale.exp().item()
    out = {}
    for name, (pos, neg) in PROMPT_SETS.items():
        with torch.no_grad():
            tp = model.encode_text(tok(pos)).float(); tp = tp / tp.norm(dim=-1, keepdim=True)
            tn = model.encode_text(tok(neg)).float(); tn = tn / tn.norm(dim=-1, keepdim=True)
            cp = tp.mean(0); cp = cp / cp.norm()
            cn = tn.mean(0); cn = cn / cn.norm()
        out[name] = torch.stack([cn, cp]).cpu().numpy().astype(np.float32)  # [NEG, POS]
    out["__logit_scale__"] = np.array(ls, dtype=np.float32)
    return model, preprocess, ls, out

# ---- 1) prompt sensitivity on ViT-B/32 (text only) ----
print("ViT-B/32: encoding prompt-set text embeddings ...")
_, _, _, tvB = text_variants("ViT-B-32")
np.savez("clip_text_variants_ViT-B-32.npz", **tvB)
print("  saved clip_text_variants_ViT-B-32.npz")

# ---- 2) optional ViT-L/14 (full image re-encode) ----
if DO_VIT_L14:
    print("ViT-L/14: loading model + encoding images (slower) ...")
    model, preprocess, ls, tvL = text_variants("ViT-L-14")
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    np.savez("clip_text_variants_ViT-L-14.npz", **tvL)
    items = []
    for label, d in [(1, IOG_DIR), (0, NONIOG_DIR)]:
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(d, f), os.path.basename(f), label))
    from PIL import Image
    cls = torch.tensor(tvL["orig"]).to(device)  # default prompt set for the csv
    names, labels, embs, rows = [], [], [], []
    for i in range(0, len(items), BATCH):
        ch = items[i:i+BATCH]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p, _, _ in ch]).to(device)
        with torch.no_grad():
            e = model.encode_image(imgs).float(); e = e / e.norm(dim=-1, keepdim=True)
            prob = (ls * e @ cls.T).softmax(-1)[:, 1]
        for (p, b, lab), ev, pr in zip(ch, e.cpu().numpy(), prob.cpu().numpy()):
            names.append(b); labels.append(lab); embs.append(ev.astype(np.float32))
            rows.append((b, lab, float(pr), int(pr >= 0.5)))
        if (i // BATCH) % 10 == 0: print(f"  {min(i+BATCH,len(items))}/{len(items)}")
    np.savez_compressed("clip_image_embeddings_ViT-L-14.npz",
                        names=np.array(names), labels=np.array(labels), embeddings=np.vstack(embs))
    with open("clip_zeroshot_ViT-L-14.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["filename", "label", "p_iog", "pred"]); w.writerows(rows)
    print("  saved L/14 embeddings, text variants, zeroshot csv")

print("\nDONE. Upload the clip_*_ViT-* files that were produced.")
