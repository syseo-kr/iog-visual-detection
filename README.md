# IOG visual-detection revision reproducibility archive

This is a candidate *new version* of the original Zenodo release, based directly on
`iog-visual-detection-v1.0.2.zip` (28 original entries). It retains the previously
released derived dataset, as-run analysis code, code/data licenses, and `reproduce.py`.
It has **not** been deposited to Zenodo by this package builder.

## What changed

* `data/samples.csv`: corrected majority-label representative flags and deduplicated
  fold metadata (8 representative flags and 8 dedup fold values changed). All 2,519
  labels, 1,775 pHash component IDs, original model predictions and embeddings are unchanged.
* Corrected the data dictionary: `md5` hashes original image-file bytes, whereas
  `phash` is computed after 224x224 standardization.
* `followups/` contains anonymized per-screenshot predictions for the 600-screenshot
  four-prompt experiment, three transformation conditions on 160 screenshots,
  cluster chaining sensitivity, CV prediction records, source-domain audit aggregates,
  provider API token/time usage summaries, and CLIP CPU timing summaries.
* `analysis/verify_release.py` provides a fast deterministic check of data alignment
  and key original/new results without retraining.
* `analysis/reproduce_followups.py` regenerates descriptive metrics for the two
  LMM follow-up studies and CPU/API usage tables.
* `analysis/reproduce_statistics.py` regenerates the five exact cluster sign-flip
  tests with Holm correction and conditional cluster-bootstrap accuracy intervals.
* `analysis/reproduce_chaining.py` regenerates fixed-fold outcome metrics from
  released chaining-exclusion predictions (without claiming to refit the models).

## Quick start

Python 3.10+ recommended. Install `pip install -r requirements-analysis.txt` for the offline checks.
Use `requirements.txt` only for the entire upstream pipeline, which may need
PyTorch, CLIP and provider SDKs.

```bash
python analysis/verify_release.py
python analysis/reproduce_followups.py
python analysis/reproduce_statistics.py
python analysis/reproduce_chaining.py
```

For the original full ML classifier training evaluation, run `python reproduce.py`.
This is computationally heavier because it includes five-fold CV for multiple
classifiers, an RBF-SVM and the handcrafted feature matrix.

## Interpretation and scope

The original main-set predictions and the follow-up experiment predictions are
**not** interchangeable: the latter used distinct samples, batches, and image
transformation conditions. The main 600-image prompt analysis uses the **570 common
complete-coverage screenshots** (285 IOG + 285 controls). Exactly three Gemini
batches had no valid outputs under each of four prompts, leaving 12 incomplete
model/prompt/batch combinations out of 720.

The square224 condition is from the earlier 160-image P0 follow-up, whereas
pad224 and pad1024 are from a subsequent 160-image resolution run on the same
sample IDs. Comparisons thus measure observed input-condition differences, not
solely the isolated effect of image resolution; providers could internally
transform images.

Gemini had identical input token totals (49,680) under pad224 and pad1024.
This **does not prove** the effective internal image resolution was identical.
GPT-5's separately reported reasoning-token count was not retained; reported
output token totals include billable reasoning tokens. Gemini thinking tokens
are recorded separately from output tokens for cost estimation.

API costs are estimates using explicit assumptions in `cost_rate_assumptions.csv`.
These figures are neither actual invoices nor guaranteed current provider prices.
Local-CPU cost (power/hardware) was not monetized.

## Privacy and rights

Raw KRA IOG screenshots, any actual site/domain strings, source image paths,
API keys, full raw API responses and inaccessible account details are **excluded**.
All public tables are keyed by opaque `img_XXXXX` identifiers or aggregated counts.
The code is provided as archival as-run software and may need environment-specific
paths and legitimate access to raw images for upstream re-extraction.
All code is subject to the MIT license (`LICENSE`); previously released derived
content remains under the CC BY 4.0 data license (`LICENSE-data`).

## Citations and Zenodo

This package derives from the earlier public record DOI
[10.5281/zenodo.20636926](https://doi.org/10.5281/zenodo.20636926).
The updated version DOI is **not assigned until deposition**, and must be inserted
in the final paper and rebuttal only after upload and verification.
See `CHANGELOG.md`, `DATA_AVAILABILITY.md`, and `followups/README.md`.
