# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

#!/usr/bin/env python3
"""FINAL locked comparison. Canonical labels = verified ground_truth from the LMM
xlsx (corrects the 46 mis-foldered IOG images). Every method evaluated on the SAME
folds: classical ML + CLIP (zero-shot & linear-probe) + the three LMMs + ensembles."""
import os, csv, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

OUT="/home/claude/artifacts"; UP="/mnt/user-data/uploads"; SEED=42
meta=list(csv.DictReader(open(os.path.join(OUT,"meta.csv"))))
base=[r["filename"] for r in meta]; n=len(base)
cid=np.load(os.path.join(OUT,"cluster_id.npy"))
X=np.load(os.path.join(OUT,"X_hand.npy"))

# ---- canonical labels from verified ground_truth ----
df=pd.read_excel("/home/claude/lmm.xlsx"); mp={"O":1,"X":0}
df=df.drop_duplicates(subset="filename", keep="first").reset_index(drop=True)
gt={f:mp[v] for f,v in zip(df["filename"], df["ground_truth"])}
y=np.array([gt[b] for b in base])
gpt=np.array([mp[v] for v in df.set_index("filename")["gpt"].reindex(base)])
claude=np.array([mp[v] for v in df.set_index("filename")["claude"].reindex(base)])
gemini=np.array([mp[v] for v in df.set_index("filename")["gemini"].reindex(base)])
OR=((gpt+claude+gemini)>=1).astype(int)
MAJ=((gpt+claude+gemini)>=2).astype(int)
print(f"canonical: IOG={int(y.sum())} non-IOG={int((y==0).sum())} (was 1254/1265 by folder)")

# ---- CLIP: zero-shot preds + embeddings, aligned ----
zr={r["filename"]:float(r["p_iog"]) for r in csv.DictReader(open(UP+"/clip_zeroshot.csv"))}
zpred=np.array([1 if zr[b]>=0.5 else 0 for b in base])
z=np.load(UP+"/clip_image_embeddings.npz", allow_pickle=True)
emb={nm:e for nm,e in zip(z["names"].tolist(), z["embeddings"])}
E=np.vstack([emb[b] for b in base]).astype(np.float32)

# ---- folds (rebuilt on canonical labels) ----
full_folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y,groups=cid))
# dedup: one representative per cluster, label = majority
K=cid.max()+1; reps=[]; ry=[]
for c in range(K):
    idx=np.nonzero(cid==c)[0]; maj=int(round(y[idx].mean()))
    cand=idx[y[idx]==maj]  # representative drawn from the cluster's majority-label members
    reps.append(int(cand[0] if len(cand) else idx[0])); ry.append(maj)
reps=np.array(reps); ry=np.array(ry)
rf=np.full(len(reps),-1)
for fi,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(reps,ry)): rf[te]=fi
dedup_folds=[(reps[rf!=f], reps[rf==f]) for f in range(5)]
print(f"dedup clean set: {len(reps)} (IOG {int(ry.sum())} / non-IOG {int((ry==0).sum())})")

def metrics(yt,yp):
    tp=int(((yp==1)&(yt==1)).sum());tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum());fn=int(((yp==0)&(yt==1)).sum())
    acc=(tp+tn)/max(tp+tn+fp+fn,1);pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1)
    sp=tn/max(tn+fp,1);f1=2*pr*rc/max(pr+rc,1e-12);return np.array([acc,pr,rc,f1,sp])
def agg(rows):a=np.array(rows);return a.mean(0),a.std(0)
def fixed(pred,folds):return agg([metrics(y[te],pred[te]) for _,te in folds])
def knn(folds,k=3):
    h=np.array([int(r["phash"],16) for r in meta],dtype=np.uint64)
    b=np.unpackbits(h.view(np.uint8).reshape(n,8),axis=1)
    out=[]
    for tr,te in folds:
        preds=[int(round(y[tr][np.argsort((b[tr]!=b[i]).sum(1))[:k]].mean())) for i in te]
        out.append(metrics(y[te],np.array(preds)))
    return agg(out)
def cv(make,Xm,folds):
    out=[]
    for tr,te in folds:
        c=make();c.fit(Xm[tr],y[tr]);out.append(metrics(y[te],c.predict(Xm[te])))
    return agg(out)

def classical(bal):
    cw="balanced" if bal else None
    return {
        "LogReg":lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=4000,class_weight=cw))]),
        "RBF-SVM":lambda:Pipeline([("s",StandardScaler()),("m",SVC(kernel="rbf",class_weight=cw))]),
        "RandomForest":lambda:RandomForestClassifier(n_estimators=400,n_jobs=-1,class_weight=("balanced_subsample" if bal else None),random_state=0),
        "HistGradBoost":lambda:HistGradientBoostingClassifier(class_weight=cw,random_state=0),
    }

cols=["Acc","Prec","Rec","F1","Spec"]
def show(title,folds,bal):
    print("\n"+"="*84+f"\n{title}\n"+"="*84)
    print(f"{'Method':<24}"+"".join(f"{c:>11}" for c in cols))
    rows={}
    def line(nm,mu,sd):
        rows[nm]=(mu.tolist(),sd.tolist())
        print(f"{nm:<24}"+"".join(f" {mu[i]*100:5.1f}±{sd[i]*100:3.1f}" for i in range(5)))
    line("All-negative baseline",*fixed(np.zeros(n,int),folds))
    line("pHash-kNN (k=3)",*knn(folds))
    for nm,mk in classical(bal).items(): line(nm+(" (bal)" if bal else ""),*cv(mk,X,folds))
    line("CLIP zero-shot (t=.5)",*fixed(zpred,folds))
    line("CLIP linear-probe",*cv(lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=4000,class_weight=("balanced" if bal else None)))]),E,folds))
    line("GPT (zero-shot)",*fixed(gpt,folds))
    line("Gemini (zero-shot)",*fixed(gemini,folds))
    line("Claude (zero-shot)",*fixed(claude,folds))
    line("OR-ensemble (LMM)",*fixed(OR,folds))
    line("Majority-vote (LMM)",*fixed(MAJ,folds))
    return rows

import sys
WHICH=sys.argv[1] if len(sys.argv)>1 else "both"
res={}
if WHICH in ("full","both"):
    res["full"]=show("FULL SET  (group-CV, every image, balanced ~50%)  --  comparable to LMM table", full_folds, bal=False)
if WHICH in ("dedup","both"):
    res["dedup"]=show("DEDUP SET (novel-template generalization, imbalanced)", dedup_folds, bal=True)
json.dump({"columns":cols,"results":res},open(os.path.join(OUT,f"results_final_{WHICH}.json"),"w"),indent=2)
print(f"\nsaved results_final_{WHICH}.json")
