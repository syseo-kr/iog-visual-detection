# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
"""Ablation: do the evaluation controls matter?
 A) Near-duplicate LEAKAGE: naive random K-fold vs group-aware K-fold (clusters intact).
 B) Provenance CONFOUND: a classifier using ONLY image metadata (size/aspect/format),
    no pixels content -> if it scores high on raw images, the label leaks through
    capture provenance; standardization collapses it."""
import os, csv
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

OUT="/home/claude/artifacts"; UP="/mnt/user-data/uploads"; SEED=42
meta=[r["filename"] for r in csv.DictReader(open(os.path.join(OUT,"meta.csv")))]; n=len(meta)
cid=np.load(os.path.join(OUT,"cluster_id.npy")); X=np.load(os.path.join(OUT,"X_hand.npy"))
df=pd.read_excel("/home/claude/lmm.xlsx").drop_duplicates("filename"); mp={"O":1,"X":0}
gt={f:mp[v] for f,v in zip(df["filename"],df["ground_truth"])}; y=np.array([gt[b] for b in meta])
zL=np.load(UP+"/clip_image_embeddings_ViT-L-14.npz",allow_pickle=True)
embL={k:v for k,v in zip(zL["names"].tolist(),zL["embeddings"])}
EL=np.vstack([embL[b] for b in meta]).astype(np.float32)

def path(b):
    for d in ("data/iog","data/non_iog"):
        p=os.path.join(d,b)
        if os.path.exists(p): return p
    return None

# raw metadata features
W=[];H=[];JP=[];SZ=[]
for b in meta:
    p=path(b)
    with Image.open(p) as im: w,h=im.size
    W.append(w);H.append(h);JP.append(1 if b.lower().endswith(('.jpg','.jpeg')) else 0);SZ.append(os.path.getsize(p))
W=np.array(W,float);H=np.array(H,float);JP=np.array(JP,float);SZ=np.array(SZ,float)
META=np.column_stack([W,H,W/H,JP,np.log1p(SZ)])  # width, height, aspect, is_jpeg, log size

folds_group=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y,groups=cid))
folds_naive=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y))

def af(yt,yp):
    tp=((yp==1)&(yt==1)).sum();tn=((yp==0)&(yt==0)).sum();fp=((yp==1)&(yt==0)).sum();fn=((yp==0)&(yt==1)).sum()
    acc=(tp+tn)/len(yt);pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1);f1=2*pr*rc/max(pr+rc,1e-12);return acc*100,f1*100
def cv(make,Xm,folds):
    A=[];F=[]
    for tr,te in folds:
        c=make();c.fit(Xm[tr],y[tr]);a,f=af(y[te],c.predict(Xm[te]));A.append(a);F.append(f)
    return np.mean(A),np.std(A),np.mean(F),np.std(F)

print("="*78)
print("ABLATION A — near-duplicate LEAKAGE  (naive random split vs group-aware split)")
print("="*78)
print(f"{'Model':<22}{'split':<14}{'Acc':>12}{'F1':>12}")
for nm,mk,Xm in [("CLIP-L/14 lin-probe",lambda:Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=4000))]),EL),
                 ("Handcrafted LogReg",lambda:Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=4000))]),X)]:
    an,sa,fn_,sf=cv(mk,Xm,folds_naive); ag,sga,fg,sfg=cv(mk,Xm,folds_group)
    print(f"{nm:<22}{'naive random':<14}{an:>7.1f}±{sa:>3.1f}{fn_:>7.1f}±{sf:>3.1f}")
    print(f"{'':<22}{'group-aware':<14}{ag:>7.1f}±{sga:>3.1f}{fg:>7.1f}±{sfg:>3.1f}")
    print(f"{'':<22}{'-> inflation':<14}{an-ag:>+7.1f}{'':>4}{fn_-fg:>+7.1f}")

print("\n"+"="*78)
print("ABLATION B — provenance CONFOUND  (classifier using ONLY metadata, no pixels)")
print("="*78)
am,sm,fm,sfm=cv(lambda:HistGradientBoostingClassifier(random_state=0),META,folds_group)
print(f"metadata-only [width,height,aspect,is_jpeg,log size], group-aware CV:  Acc {am:.1f}±{sm:.1f}  F1 {fm:.1f}±{sfm:.1f}")
print(f"  (majority floor = {max((y==1).mean(),(y==0).mean())*100:.1f}%)")
# what standardization does to those features:
print("  After standardization every image -> 224x224, aspect 1.0, JPEG q90:")
print(f"    raw distinct (w,h): {len(set(zip(W.astype(int),H.astype(int))))} sizes, jpeg fraction {JP.mean()*100:.0f}%  ->  standardized: 1 size, aspect var=0, 100% same format")
print("    => metadata signal is removed by construction (collapses to the majority floor).")
