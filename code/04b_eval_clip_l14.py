# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
"""CLIP prompt-sensitivity (6 prompt sets x {B/32, L/14}) + ViT-L/14 zero-shot &
linear-probe, all on the SAME canonical labels and folds as the final tables."""
import os, csv, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

OUT="/home/claude/artifacts"; UP="/mnt/user-data/uploads"; SEED=42
meta=[r["filename"] for r in csv.DictReader(open(os.path.join(OUT,"meta.csv")))]; n=len(meta)
cid=np.load(os.path.join(OUT,"cluster_id.npy"))
df=pd.read_excel("/home/claude/lmm.xlsx").drop_duplicates("filename"); mp={"O":1,"X":0}
gt={f:mp[v] for f,v in zip(df["filename"],df["ground_truth"])}
y=np.array([gt[b] for b in meta])

def load_emb(fn):
    z=np.load(UP+"/"+fn, allow_pickle=True); d={k:v for k,v in zip(z["names"].tolist(),z["embeddings"])}
    return np.vstack([d[b] for b in meta]).astype(np.float32)
EB=load_emb("clip_image_embeddings.npz")               # B/32 (512)
EL=load_emb("clip_image_embeddings_ViT-L-14.npz")      # L/14 (768)
TVB=np.load(UP+"/clip_text_variants_ViT-B-32.npz")
TVL=np.load(UP+"/clip_text_variants_ViT-L-14.npz")
PROMPTS=["orig","terse","action","style","illegal","korean_ctx"]

# folds (identical to final_tables)
full_folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y,groups=cid))
K=cid.max()+1; reps=[]; ry=[]
for c in range(K):
    idx=np.nonzero(cid==c)[0]; reps.append(int(idx[0])); ry.append(int(round(y[idx].mean())))
reps=np.array(reps); ry=np.array(ry); rf=np.full(len(reps),-1)
for fi,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(reps,ry)): rf[te]=fi
dedup_folds=[(reps[rf!=f],reps[rf==f]) for f in range(5)]

def metrics(yt,yp):
    tp=int(((yp==1)&(yt==1)).sum());tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum());fn=int(((yp==0)&(yt==1)).sum())
    acc=(tp+tn)/max(tp+tn+fp+fn,1);pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1)
    sp=tn/max(tn+fp,1);f1=2*pr*rc/max(pr+rc,1e-12);return np.array([acc,pr,rc,f1,sp])
def agg(rows):a=np.array(rows);return a.mean(0),a.std(0)
def zs_pred(E,C,ls):
    e=E/np.linalg.norm(E,axis=1,keepdims=True)
    logit=ls*(e@C.T); p=np.exp(logit); p=p/p.sum(1,keepdims=True)
    return (p[:,1]>=0.5).astype(int)
def fixed(pred,folds):return agg([metrics(y[te],pred[te]) for _,te in folds])
def lp(E,folds,bal):
    out=[]
    for tr,te in folds:
        c=Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=4000,class_weight=("balanced" if bal else None)))])
        c.fit(E[tr],y[tr]);out.append(metrics(y[te],c.predict(E[te])))
    return agg(out)

# ---- 1) prompt sensitivity (full set) ----
print("="*70,"\nPROMPT SENSITIVITY (zero-shot, FULL set) — Acc / F1 per prompt set\n"+"="*70)
print(f"{'prompt set':<13}{'B/32 Acc':>10}{'B/32 F1':>9}   {'L/14 Acc':>10}{'L/14 F1':>9}")
sens={"B/32":{"acc":[],"f1":[]},"L/14":{"acc":[],"f1":[]}}
for ps in PROMPTS:
    mB,_=fixed(zs_pred(EB,TVB[ps],float(TVB["__logit_scale__"])),full_folds)
    mL,_=fixed(zs_pred(EL,TVL[ps],float(TVL["__logit_scale__"])),full_folds)
    sens["B/32"]["acc"].append(mB[0]); sens["B/32"]["f1"].append(mB[3])
    sens["L/14"]["acc"].append(mL[0]); sens["L/14"]["f1"].append(mL[3])
    print(f"{ps:<13}{mB[0]*100:>9.1f}{mB[3]*100:>9.1f}   {mL[0]*100:>9.1f}{mL[3]*100:>9.1f}")
for m in ["B/32","L/14"]:
    a=np.array(sens[m]["acc"])*100; f=np.array(sens[m]["f1"])*100
    print(f"  {m} across 6 prompts:  Acc {a.mean():.1f} (range {a.min():.1f}–{a.max():.1f}, sd {a.std():.1f}) | "
          f"F1 {f.mean():.1f} (range {f.min():.1f}–{f.max():.1f}, sd {f.std():.1f})")

# ---- 2) B/32 vs L/14 consolidated (orig prompt) on both populations ----
cols=["Acc","Prec","Rec","F1","Spec"]
def block(title,folds,bal):
    print("\n"+"="*70+f"\n{title}\n"+"="*70)
    print(f"{'Method':<22}"+"".join(f"{c:>9}" for c in cols))
    def line(nm,mu,sd): print(f"{nm:<22}"+"".join(f" {mu[i]*100:5.1f}" for i in range(5)))
    line("CLIP-B/32 zero-shot",*fixed(zs_pred(EB,TVB["orig"],float(TVB["__logit_scale__"])),folds))
    line("CLIP-L/14 zero-shot",*fixed(zs_pred(EL,TVL["orig"],float(TVL["__logit_scale__"])),folds))
    line("CLIP-B/32 lin-probe",*lp(EB,folds,bal))
    line("CLIP-L/14 lin-probe",*lp(EL,folds,bal))
block("FULL SET (vs LMM: Claude 91.1/90.6, OR 91.4/90.9)", full_folds, bal=False)
block("DEDUP SET (vs LMM: Claude 92.6/88.4, OR 92.7/88.7)", dedup_folds, bal=True)
print("\ndone")
