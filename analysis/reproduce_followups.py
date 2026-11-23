#!/usr/bin/env python3
"""Regenerate numerical tables from anonymized offline records, no raw images/API.
Creates `followups/recomputed_{prompts,resolution,api_costs}.csv`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'followups'

def metrics(g):
    truth=g.label.to_numpy(); guess=g.pred.eq('O').to_numpy().astype(int)
    tp=np.sum((truth==1)&(guess==1));fp=np.sum((truth==0)&(guess==1));fn=np.sum((truth==1)&(guess==0))
    return {'n':len(g),'accuracy_pct':round(100*np.mean(truth==guess),4),
            'precision_pct':round(100*tp/max(tp+fp,1),4),
            'recall_pct':round(100*tp/max(tp+fn,1),4),
            'f1_pct':round(100*2*tp/max(2*tp+fp+fn,1),4)}

def main():
    prm=pd.read_csv(F/'lmm_prompt_600_predictions.csv')
    common=set.intersection(*(set(d.id) for _,d in prm.groupby(['provider','prompt_id'])))
    rows=[]
    for (provider,prompt),g in prm[prm.id.isin(common)].groupby(['provider','prompt_id']):
        rows.append({'provider':provider,'prompt_id':prompt,**metrics(g)})
    pd.DataFrame(rows).sort_values(['provider','prompt_id']).to_csv(F/'recomputed_prompts.csv',index=False)
    res=pd.read_csv(F/'lmm_resolution_160_predictions.csv')
    rows=[]
    for (provider,condition),g in res.groupby(['provider','condition']):
        rows.append({'provider':provider,'condition':condition,**metrics(g)})
    pd.DataFrame(rows).sort_values(['provider','condition']).to_csv(F/'recomputed_resolution.csv',index=False)
    usage=pd.read_csv(F/'api_request_usage_160.csv')
    rates=pd.read_csv(F/'cost_rate_assumptions.csv').set_index('provider')
    rows=[]
    for (provider,cond),g in usage.groupby(['provider','condition']):
        inp=int(g.input_tokens.sum());out=int(g.output_tokens.sum());think=int(g.thinking_tokens_separate.sum());
        # Gemini caches thinking as a separate field; GPT counts reasoning inside output.
        billable_out=out+(think if provider=='gemini' else 0)
        price=(inp*float(rates.loc[provider,'input_USD_per_million'])+billable_out*float(rates.loc[provider,'output_USD_per_million']))/1e6
        n=int(g.images_in_batch.sum())
        rows.append({'provider':provider,'condition':cond,'batches':len(g),'images':n,
                     'input_tokens':inp,'output_tokens':out,'separately_logged_thinking_tokens':think if provider=='gemini' else 'n/r',
                     'latency_ms_per_image':round(1000*g.latency_seconds.sum()/n,3),
                     'estimated_USD_for_images':round(price,7),'estimated_USD_per_1000_images':round(1000*price/n,6)})
    pd.DataFrame(rows).sort_values(['provider','condition']).to_csv(F/'recomputed_api_costs.csv',index=False)
    print('Wrote 3 tables: prompts, resolution, API costs in followups/.')
    print(pd.DataFrame(rows)[['provider','condition','estimated_USD_per_1000_images']].to_string(index=False))

if __name__=='__main__':main()
