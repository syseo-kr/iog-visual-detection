#!/usr/bin/env python3
"""Recompute recorded fixed-fold chaining exclusion metrics from public predictions.
The original model refitting that generated those predictions requires the released
embeddings and a separate experimental protocol; this is an outcome-level audit.
"""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'followups'

def report(x,col):
    y=x.label; p=x[col];tp=((y==1)&(p==1)).sum();fp=((y==0)&(p==1)).sum();fn=((y==1)&(p==0)).sum();tn=((y==0)&(p==0)).sum()
    return {'n':len(x),'accuracy_pct':round(100*(tp+tn)/len(x),4),'precision_pct':round(100*tp/max(tp+fp,1),4),'recall_pct':round(100*tp/max(tp+fn,1),4),'f1_pct':round(100*2*tp/max(2*tp+fp+fn,1),4),'specificity_pct':round(100*tn/max(tn+fp,1),4)}

def main():
    x=pd.read_csv(F/'chaining_sensitivity_predictions.csv')
    out=[]
    for desc,data in [('all',x),('chaining',x[x.is_chaining_cluster.eq(1)]),('non_chaining',x[x.is_chaining_cluster.eq(0)])]:
        for label,col in [('original','baseline_pred'),('training_purged','chaining_excluded_pred')]:
            out.append({'population':desc,'condition':label,**report(data,col)})
    df=pd.DataFrame(out);df.to_csv(F/'recomputed_chaining_metrics.csv',index=False)
    print(df.to_string(index=False))

if __name__=='__main__':main()
