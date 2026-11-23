#!/usr/bin/env python3
"""Recompute cluster-aware conditional accuracy intervals and exact sign-flip tests.
Runs without raw screenshots. The bootstrap seed is fixed for reproducibility;
small interval differences from earlier random draws are expected.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'followups'
PAIRS=[('CLIP','Claude'),('CLIP','LMM_OR'),('CLIP','Gemini'),('CLIP','GPT5'),('LMM_OR','Claude')]
RNG_SEED=20261123; B=8000

def exact_signflip_p(effects):
    # Exact distribution of sum_j sign_j * effect_j (integer cluster scores).
    dp={0:1.0}
    for e in effects:
        e=int(e)
        if e==0:continue
        q=defaultdict(float)
        for total,prob in dp.items():
            q[total+e]+=0.5*prob;q[total-e]+=0.5*prob
        dp=q
    obs=abs(int(np.sum(effects)))
    return sum(prob for score,prob in dp.items() if abs(score)>=obs)

def main():
    f=pd.read_csv(F/'revision_full_oof.csv')
    arr=pd.DataFrame({'cluster':f.cluster_id,'label':f.label})
    preds={
        'CLIP':f.clip_old_pred.to_numpy(dtype=int),
        'Claude':f.claude_pred.to_numpy(dtype=int),
        'LMM_OR':(f.gpt_pred.astype(bool)|f.claude_pred.astype(bool)|f.gemini_pred.astype(bool)).astype(int).to_numpy(),
        'Gemini':f.gemini_pred.to_numpy(dtype=int),
        'GPT5':f.gpt_pred.to_numpy(dtype=int),
    }
    labels=f.label.to_numpy(dtype=int)
    stats={}
    for name,pred in preds.items():
        ok=(pred==labels).astype(int)
        per=pd.DataFrame({'cluster':f.cluster_id,'correct':ok}).groupby('cluster').correct.agg(['sum','count'])
        stats[name]=per
    names=list(preds);clusters=stats[names[0]].index
    cmat=np.stack([stats[n]['sum'].loc[clusters].to_numpy() for n in names],axis=1)
    nmat=stats[names[0]]['count'].loc[clusters].to_numpy()
    rng=np.random.default_rng(RNG_SEED)
    total=cmat.sum(axis=0)/len(f)*100
    vals=np.empty((B,len(names)))
    # cluster resampling (a replicate has random number of images)
    for b in range(B):
        idx=rng.integers(0,len(clusters),size=len(clusters))
        vals[b]=100*cmat[idx].sum(axis=0)/nmat[idx].sum()
    out=[{'model':name,'pooled_accuracy_pct':round(total[i],4),
          'bootstrap_CI_lower':round(np.percentile(vals[:,i],2.5),4),
          'bootstrap_CI_upper':round(np.percentile(vals[:,i],97.5),4),
          'bootstrap_iterations':B,'bootstrap_seed':RNG_SEED}for i,name in enumerate(names)]
    pd.DataFrame(out).to_csv(F/'recomputed_cluster_bootstrap_accuracy.csv',index=False)
    comparisons=[]
    for a,b in PAIRS:
        e=(stats[a]['sum']-stats[b]['sum']).to_numpy()
        comparisons.append({'comparison':a+' - '+b,
                            'accuracy_delta_pp':round(float(total[names.index(a)]-total[names.index(b)]),5),
                            'exact_cluster_signflip_p':exact_signflip_p(e)})
    ordered=sorted(comparisons,key=lambda x:x['exact_cluster_signflip_p'])
    prev=0
    for i,row in enumerate(ordered):
        prev=max(prev,min(1,(len(ordered)-i)*row['exact_cluster_signflip_p']))
        row['holm_adjusted_p']=prev
    pd.DataFrame(comparisons).to_csv(F/'recomputed_cluster_paired_tests.csv',index=False)
    print('Cluster bootstrap pooled accuracy CIs:')
    print(pd.DataFrame(out).to_string(index=False))
    print('Exact cluster signflip and Holm correction:')
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':main()
