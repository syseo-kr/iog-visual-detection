#!/usr/bin/env python3
"""Fast, deterministic checks for the public revision archive; no API or raw images."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data'; F=ROOT/'followups'

def check(cond,message):
    if not bool(cond):raise AssertionError(message)
    print('[PASS]',message)

def main():
    s=pd.read_csv(D/'samples.csv')
    check(len(s)==2519 and s.label.sum()==1300 and (s.label==0).sum()==1219,'canonical dataset size and labels')
    check(s.id.is_unique and s.cluster_id.nunique()==1775,'unique opaque IDs and pHash components')
    check(s.is_dedup_rep.sum()==1775 and int(s.loc[s.is_dedup_rep.eq(1),'label'].sum())==624,
          'corrected one-representative-per-component selection, IOG 624/control 1151')
    check(s.groupby('cluster_id').is_dedup_rep.sum().eq(1).all(), 'each component has exactly one representative')
    check(s.groupby('cluster_id').fold_full.nunique().eq(1).all(),'pHash components do not cross full CV folds')
    check(s.loc[s.is_dedup_rep.eq(0),'fold_dedup'].eq(-1).all(),'non-representative dedup fold sentinel')
    check(s.loc[s.is_dedup_rep.eq(1),'fold_dedup'].between(0,4).all(),'dedup fold values')
    grp=s.groupby('cluster_id').label
    check((grp.nunique()==2).sum()==8 and s[s.cluster_id.isin(grp.nunique()[lambda x:x.eq(2)].index)].shape[0]==59,
          '8 mixed-label pHash components comprising 59 screenshots')
    check(s.loc[s.label.eq(1),'md5'].nunique()==875,'875 raw-byte MD5 values among 1300 IOG images')
    f=pd.read_csv(F/'revision_full_oof.csv')
    check(len(f)==2519 and f.id.tolist()==s.id.tolist() and f.label.tolist()==s.label.tolist(),'OOF full-set alignment')
    y=f.label; c=f.clip_old_pred.astype(int)
    check(round((y==c).mean()*100,2)==94.05,'94.05% original CLIP main-set accuracy')
    missed=(f.label.eq(1)&f.gpt_pred.eq(0)&f.claude_pred.eq(0)&f.gemini_pred.eq(0))
    check(missed.sum()==203 and (missed&c.eq(1)).sum()==138 and (missed&c.eq(0)).sum()==65,
          '203/138/65 LMM miss/CLIP recover/all missed on original predictions')
    orp=(f.gpt_pred.eq(1)|f.claude_pred.eq(1)|f.gemini_pred.eq(1)|c.eq(1))
    check(round(100*(orp&f.label.eq(1)).sum()/1300,1)==95.0,'retrospective 95.0% IOG recall')
    d=pd.read_csv(F/'revision_dedup_oof.csv')
    rep=s[s.is_dedup_rep.eq(1)].set_index('id')
    check(len(d)==1775 and set(d.id)==set(rep.index),'dedup OOF IDs match corrected representatives')
    dref=d.set_index('id');check(dref.fold_dedup.sort_index().equals(rep.fold_dedup.sort_index()),'dedup fold assignment alignment')
    check(round((d.label==d.clip_old_pred).mean()*100,1)==94.4,'dedup CLIP original pooled accuracy rounds to 94.4%')
    p=pd.read_csv(F/'lmm_prompt_600_predictions.csv')
    check(len(p)==7080 and p.pred.isin(['O','X']).all(),'7080 final main600 valid prompt judgments')
    sets=[set(z.id) for _,z in p.groupby(['provider','prompt_id'])]
    common=set.intersection(*sets)
    check(len(common)==570 and p[p.id.isin(common)].drop_duplicates('id').label.sum()==285,
          '570 common images, 285 IOG and 285 controls')
    z=pd.read_csv(F/'lmm_resolution_160_predictions.csv')
    check(z.groupby(['provider','condition']).size().eq(160).all() and len(z)==1440,
          '160 images per provider and each of three transformations')
    usage=pd.read_csv(F/'api_request_usage_160.csv')
    check(len(usage)==96 and usage.images_in_batch.eq(10).all(),'96 successful usage records of 10 images each')
    gem=usage.groupby(['provider','condition']).input_tokens.sum()
    check(gem['gemini','pad224']==gem['gemini','pad1024']==49680,
          'Gemini 49,680 input tokens in both padded resolutions')
    check(usage.loc[usage.provider.eq('openai'),'thinking_tokens_separate'].isna().all(),
          'GPT reasoning tokens not separately retained in logs')
    print('All fast public-data checks passed. Full ML retraining not included in this check.')

if __name__=='__main__':main()
