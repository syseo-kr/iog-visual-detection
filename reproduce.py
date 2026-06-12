#!/usr/bin/env python3
"""Reproduce the reported results from the released derived-data bundle in data/.
No raw images and no figures: this regenerates the result TABLES (the numbers),
the complementarity analysis, and the two ablations. Mirrors the as-run analysis
code/ exactly, reading the anonymized id-keyed bundle instead of the raw inputs.

Usage:  python reproduce.py            # full set + dedup set + complementarity + ablations
"""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"data"); SEED=42
s=pd.read_csv(os.path.join(D,"samples.csv"))
n=len(s); y=s["label"].to_numpy(); cid=s["cluster_id"].to_numpy()
gpt,claude,gemini=s["gpt_pred"].to_numpy(),s["claude_pred"].to_numpy(),s["gemini_pred"].to_numpy()
OR=((gpt+claude+gemini)>=1).astype(int); MAJ=((gpt+claude+gemini)>=2).astype(int)
X=np.load(os.path.join(D,"features_handcrafted.npy"))
Eb=np.load(os.path.join(D,"clip_embeddings_vitb32.npy")); El=np.load(os.path.join(D,"clip_embeddings_vitl14.npy"))
zb=s["clip_zs_b32_pred"].to_numpy(); zl=s["clip_zs_l14_pred"].to_numpy()
phash=s["phash"].tolist()
print(f"loaded bundle: {n} samples | IOG {int(y.sum())} / non-IOG {int((y==0).sum())} | clusters {cid.max()+1}")

# ---- folds (group-aware full set; one-rep-per-cluster dedup set) ----
full_folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y,groups=cid))
K=cid.max()+1; reps=[]; ry=[]
for c in range(K):
    idx=np.nonzero(cid==c)[0]; maj=int(round(y[idx].mean()))
    cand=idx[y[idx]==maj]  # representative drawn from the cluster's majority-label members
    reps.append(int(cand[0] if len(cand) else idx[0])); ry.append(maj)
reps=np.array(reps); ry=np.array(ry); rf=np.full(len(reps),-1)
for fi,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(reps,ry)): rf[te]=fi
dedup_folds=[(reps[rf!=f],reps[rf==f]) for f in range(5)]

def metrics(yt,yp):
    tp=int(((yp==1)&(yt==1)).sum());tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum());fn=int(((yp==0)&(yt==1)).sum())
    acc=(tp+tn)/max(tp+tn+fp+fn,1);pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1)
    sp=tn/max(tn+fp,1);f1=2*pr*rc/max(pr+rc,1e-12);return np.array([acc,pr,rc,f1,sp])
def agg(rows): a=np.array(rows); return a.mean(0),a.std(0)
def fixed(pred,folds): return agg([metrics(y[te],pred[te]) for _,te in folds])
def knn(folds,k=3):
    h=np.array([int(x,16) for x in phash],dtype=np.uint64)
    b=np.unpackbits(h.view(np.uint8).reshape(n,8),axis=1); out=[]
    for tr,te in folds:
        preds=[int(round(y[tr][np.argsort((b[tr]!=b[i]).sum(1))[:k]].mean())) for i in te]
        out.append(metrics(y[te],np.array(preds)))
    return agg(out)
def cv(make,Xm,folds):
    out=[]
    for tr,te in folds:
        c=make();c.fit(Xm[tr],y[tr]);out.append(metrics(y[te],c.predict(Xm[te])))
    return agg(out)
def probe(bal): return lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=4000,class_weight=("balanced" if bal else None)))])
def classical(bal):
    cw="balanced" if bal else None
    return {"LogReg":lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=4000,class_weight=cw))]),
            "RBF-SVM":lambda:Pipeline([("s",StandardScaler()),("m",SVC(kernel="rbf",class_weight=cw))]),
            "RandomForest":lambda:RandomForestClassifier(n_estimators=400,n_jobs=-1,class_weight=("balanced_subsample" if bal else None),random_state=0),
            "HistGradBoost":lambda:HistGradientBoostingClassifier(class_weight=cw,random_state=0)}

cols=["Acc","Prec","Rec","F1","Spec"]
def show(title,folds,bal):
    print("\n"+"="*86+f"\n{title}\n"+"="*86)
    print(f"{'Method':<24}"+"".join(f"{c:>11}" for c in cols)); rows={}
    def line(nm,mu,sd):
        rows[nm]=[round(float(v)*100,1) for v in mu]
        print(f"{nm:<24}"+"".join(f" {mu[i]*100:5.1f}±{sd[i]*100:3.1f}" for i in range(5)))
    line("All-negative baseline",*fixed(np.zeros(n,int),folds))
    line("pHash-kNN (k=3)",*knn(folds))
    for nm,mk in classical(bal).items(): line(nm+(" (bal)" if bal else ""),*cv(mk,X,folds))
    line("CLIP-B/32 zero-shot",*fixed(zb,folds))
    line("CLIP-B/32 linear-probe",*cv(probe(bal),Eb,folds))
    line("CLIP-L/14 zero-shot",*fixed(zl,folds))
    line("CLIP-L/14 linear-probe",*cv(probe(bal),El,folds))
    line("GPT (zero-shot)",*fixed(gpt,folds))
    line("Gemini (zero-shot)",*fixed(gemini,folds))
    line("Claude (zero-shot)",*fixed(claude,folds))
    line("OR-ensemble (LMM)",*fixed(OR,folds))
    line("Majority-vote (LMM)",*fixed(MAJ,folds))
    return rows

res={"full":show("FULL SET  (group-aware 5-fold CV, every image)",full_folds,bal=False),
     "dedup":show("DEDUP SET (one representative per cluster; novel-template)",dedup_folds,bal=True)}

# ---- complementarity (out-of-fold) ----
def oof(make,Xm):
    p=np.full(n,-1)
    for tr,te in full_folds:
        c=make();c.fit(Xm[tr],y[tr]);p[te]=c.predict(Xm[te])
    return p
oof_clip=oof(probe(False),El)
pos=(y==1); P=int(pos.sum())
U=pos&(gpt==0)&(claude==0)&(gemini==0)
hard=U&(oof_clip==0)
print("\n"+"="*86+"\nCOMPLEMENTARITY\n"+"="*86)
print(f"IOG sites: {P}")
print(f"missed by all three LMMs (unanimous FN): {int(U.sum())} ({U.sum()/P*100:.1f}% of IOG)")
print(f"   CLIP-L/14 linear-probe recovers: {int((oof_clip[U]==1).sum())}/{int(U.sum())} ({(oof_clip[U]==1).mean()*100:.1f}%)")
print(f"HARD CORE missed by all (3 LMMs + CLIP-L/14 LP): {int(hard.sum())} ({hard.sum()/P*100:.1f}% of IOG)")
rec=lambda pred:(pred[pos]==1).mean()*100
print(f"recall:  Claude {rec(claude):.1f} | OR {rec(OR):.1f} | CLIP-L/14 LP {rec(oof_clip):.1f} | Claude+CLIP-L/14 {rec(((claude==1)|(oof_clip==1)).astype(int)):.1f}")
res["complementarity"]={"IOG":P,"lmm_unanimous_FN":int(U.sum()),"clip_recovers":int((oof_clip[U]==1).sum()),
                        "hard_core":int(hard.sum()),"recall_combined":round(rec(((claude==1)|(oof_clip==1)).astype(int)),1)}

# ---- ablation: leakage + provenance confound ----
pm=pd.read_csv(os.path.join(D,"provenance_metadata.csv"))
META=np.column_stack([pm["width"],pm["height"],pm["aspect"],pm["is_jpeg"],np.log1p(pm["filesize"])])
naive=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros((n,1)),y))
def af(make,Xm,folds):
    A=[];F=[]
    for tr,te in folds:
        c=make();c.fit(Xm[tr],y[tr]);m=metrics(y[te],c.predict(Xm[te]));A.append(m[0]);F.append(m[3])
    return np.mean(A)*100,np.mean(F)*100
print("\n"+"="*86+"\nABLATION\n"+"="*86)
print("A) near-duplicate leakage (naive random vs group-aware split):")
for nm,mk,Xm in [("CLIP-L/14 linear-probe",probe(False),El),("Handcrafted LogReg",probe(False),X)]:
    an,_=af(mk,Xm,naive); ag,_=af(mk,Xm,full_folds)
    print(f"   {nm:<22} naive {an:5.1f}  group {ag:5.1f}  inflation {an-ag:+.1f}")
amet,_=af(lambda:HistGradientBoostingClassifier(random_state=0),META,full_folds)
print(f"B) provenance confound: metadata-only classifier (no pixels) = {amet:.1f}% accuracy  (floor {max((y==1).mean(),(y==0).mean())*100:.1f}%)")
res["ablation"]={"metadata_only_acc":round(amet,1)}

json.dump(res,open(os.path.join(HERE,"reproduced_results.json"),"w"),indent=2)
print("\nsaved reproduced_results.json")
