# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
"""Complementarity: LMM unanimous false-negatives vs cheap-model coverage (out-of-fold),
the reverse, the residual 'hard core' missed by everyone, and augmented recall."""
import os, csv
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

OUT="/home/claude/artifacts"; UP="/mnt/user-data/uploads"; SEED=42
meta=[r["filename"] for r in csv.DictReader(open(os.path.join(OUT,"meta.csv")))]; n=len(meta)
cid=np.load(os.path.join(OUT,"cluster_id.npy")); X=np.load(os.path.join(OUT,"X_hand.npy"))
df=pd.read_excel("/home/claude/lmm.xlsx").drop_duplicates("filename"); mp={"O":1,"X":0}
gt={f:mp[v] for f,v in zip(df["filename"],df["ground_truth"])}; y=np.array([gt[b] for b in meta])
g=lambda col: np.array([mp[v] for v in df.set_index("filename")[col].reindex(meta)])
gpt,claude,gemini=g("gpt"),g("claude"),g("gemini"); OR=((gpt+claude+gemini)>=1).astype(int)
_zL=np.load(UP+"/clip_image_embeddings_ViT-L-14.npz",allow_pickle=True)
_embL={k:v for k,v in zip(_zL["names"].tolist(),_zL["embeddings"])}
EL=np.vstack([_embL[b] for b in meta]).astype(np.float32)

folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y,groups=cid))
def oof(make,Xm):
    p=np.full(n,-1)
    for tr,te in folds:
        c=make(); c.fit(Xm[tr],y[tr]); p[te]=c.predict(Xm[te])
    return p
oof_clip=oof(lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=4000))]),EL)
oof_hgb =oof(lambda:HistGradientBoostingClassifier(random_state=0),X)

pos=(y==1); P=int(pos.sum())
U=pos & (gpt==0)&(claude==0)&(gemini==0)            # LMM unanimous false negatives
print(f"IOG sites: {P}")
print(f"LMM unanimous false-negatives (missed by GPT&Claude&Gemini): {int(U.sum())} ({U.sum()/P*100:.1f}% of IOG)")
for nm,p in [("CLIP-L/14 lin-probe",oof_clip),("HistGradBoost",oof_hgb)]:
    cov=(p[U]==1).mean() if U.sum() else 0
    print(f"   -> {nm} catches {int((p[U]==1).sum())}/{int(U.sum())} of them ({cov*100:.1f}%) [out-of-fold]")

print()
V=pos & (oof_clip==0)                                # CLIP-L/14 LP false negatives
print(f"CLIP-L/14 lin-probe false-negatives: {int(V.sum())} ({V.sum()/P*100:.1f}% of IOG)")
print(f"   -> OR-ensemble(LMM) catches {int((OR[V]==1).sum())}/{int(V.sum())} of them ({(OR[V]==1).mean()*100:.1f}%)")
print(f"   -> Claude catches {int((claude[V]==1).sum())}/{int(V.sum())} ({(claude[V]==1).mean()*100:.1f}%)")

print()
hard=pos & (gpt==0)&(claude==0)&(gemini==0)&(oof_clip==0)
print(f"HARD CORE missed by ALL (3 LMMs + CLIP-L/14 LP): {int(hard.sum())} ({hard.sum()/P*100:.1f}% of IOG)")
print(f"   vs LMM-only unanimous frontier {int(U.sum())}: adding CLIP-L/14 LP shrinks it by {int(U.sum()-hard.sum())}")

print()
def rec(pred): return (pred[pos]==1).mean()*100
aug_or_clip=((OR==1)|(oof_clip==1)).astype(int)
aug_claude_clip=((claude==1)|(oof_clip==1)).astype(int)
print("Recall (sensitivity) comparison:")
print(f"   Claude {rec(claude):.1f} | OR-ensemble {rec(OR):.1f} | CLIP-L/14 LP {rec(oof_clip):.1f}")
print(f"   Claude OR CLIP-L/14 LP : {rec(aug_claude_clip):.1f}   |   OR(LMM) OR CLIP-L/14 LP : {rec(aug_or_clip):.1f}")
