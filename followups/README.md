# Additional results, anonymized derived tables

- `lmm_prompt_600_predictions.csv`: 7,080 predictions from 3 models and 4 prompts;
  the primary paired analysis uses 570 IDs present for all 12 model-prompt pairs.
- `lmm_resolution_160_predictions.csv`: each of 160 screenshots is evaluated
  under square224, pad224, and pad1024 for all three providers (1,440 rows).
- `api_request_usage_160.csv`: 96 successful ten-image API requests for pad224
  and pad1024, containing only usage numbers, models and batch IDs.
- `cost_rate_assumptions.csv`: source rate assumptions, not current prices.
- `clip_cpu_five_runs.csv`: 5 locally measured CLIP CPU timing runs.
- `revision_full_oof.csv`, `revision_dedup_oof.csv`: original and secondary CLIP
  OOF predictions for condition-specific uncertainty and robustness analysis.
  `clip_old_pred` denotes the original main-benchmark OOF probe. Do not replace
  original Fig. 2's recall calculation with `clip_std_pred`.
- `chaining_sensitivity_predictions.csv` and JSON: matched original folds, with
  chaining components removed from training only; does not isolate causal effect.
- `domain_fold_audit_summary.json`: source-file audit summary; 1,524 source files
  carried hostnames, corresponding to 1,517 distinct registered domains, whereas
  1,519 matched screenshots had domain evidence in the 2,519-row canonical set.
  There were three recovered registered domains spanning folds, all in controls.
  This does not establish complete site-level independence.
- `domain_coverage_by_class.csv`: 1,254/1,300 IOG (96.46%) and 265/1,219
  controls (21.74%) with recoverable domain evidence.
- `domain_overlap_detail_safe.csv`: opaque/hashed identifiers only, no domain URLs.
- `clip_preprocess_comparison.json`: summary of earlier and standardized CLIP
  preprocessing sensitivity.
- `LMM_P0_P3_exact_prompts.txt`: exact follow-up text for P0 through P3, without
  any real filenames or sensitive images.

Sources: finalized user-provided predictions files, numeric API log summaries,
source audit summaries and the original released derived dataset. These tables
represent completed *offline* evaluations and do not require additional API calls.
