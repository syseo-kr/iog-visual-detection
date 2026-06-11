# Leakage and provenance confounds in visual detection of illegal online gambling websites

Code and derived data for the study *"Leakage and provenance confounds in visual
detection of illegal online gambling websites: a controlled evaluation of large
multimodal models and lightweight vision baselines."*

The study evaluates three frontier large multimodal models (GPT-5, Claude Sonnet 4.5,
Gemini 2.5 Flash) against low-cost vision baselines (handcrafted color/texture features,
perceptual-hash matching, and CLIP zero-shot and linear-probe) for classifying website
screenshots as illegal online gambling (IOG) or not, under a leakage-controlled protocol.

## What is in this repository

```
iog-visual-detection/
├── reproduce.py            Regenerates every reported result TABLE from data/ (no raw images)
├── data/                   Anonymized, id-keyed derived data (sufficient to reproduce all results)
│   ├── samples.csv         Per-image: id, label, cluster_id, folds, LMM & CLIP zero-shot predictions, phash, md5
│   ├── provenance_metadata.csv  Per-image: original width/height/aspect/format/filesize (for the confound ablation)
│   ├── features_handcrafted.npy 2519 x 2342 handcrafted feature matrix (row order = samples.csv)
│   ├── clip_embeddings_vitb32.npy 2519 x 512 frozen CLIP ViT-B/32 image embeddings
│   ├── clip_embeddings_vitl14.npy 2519 x 768 frozen CLIP ViT-L/14 image embeddings
│   ├── clip_text/          CLIP text-prompt embeddings (zero-shot class & prompt variants)
│   └── README.md           Data dictionary
├── code/                   The exact analysis scripts used to perform the study
│   ├── 00_run_lmm_detection.py   Query GPT-5 / Claude / Gemini on screenshots -> predictions
│   ├── 01_extract_features.py    Handcrafted features from raw screenshots
│   ├── 02_cluster_and_split.py   Perceptual-hash near-duplicate clustering + group-aware folds
│   ├── 03_run_clip.py / 03b_clip_extra.py  CLIP image/text embeddings & zero-shot
│   ├── 04_eval_fullset.py        Full-set & dedup CV (classical, CLIP B/32, LMMs, ensembles)
│   ├── 04b_eval_clip_l14.py      CLIP ViT-L/14 zero-shot & linear-probe
│   ├── 05_complementarity.py     LMM-missed vs low-cost-model coverage, residual hard core
│   └── 06_ablation.py            Leakage (naive vs group split) + provenance (metadata-only) confounds
├── requirements.txt
├── LICENSE                 Code: MIT
├── LICENSE-data            Derived data: CC BY 4.0
└── DATA_AVAILABILITY.md    Data/code availability statement
```

## Reproducing the results

The released derived data are sufficient to reproduce every reported number **without the
raw screenshots**. With Python 3.10+:

```bash
pip install -r requirements.txt     # numpy, pandas, scikit-learn
python reproduce.py
```

This regenerates the full-set and deduplicated comparison tables (Accuracy/Precision/Recall/
F1/Specificity for all methods), the complementarity analysis, and the two ablations, and
writes `reproduced_results.json`. Runtime is a few minutes (the RBF-SVM on the full feature
matrix is the slow step). All folds are reconstructed deterministically (seed 42), so the
numbers match the paper exactly.

`code/` contains the analysis scripts **as run**; their paths reflect the authors' working
environment and the upstream steps (`00`, `01`, `03`) require the raw screenshots. For
verification on the released data, use `reproduce.py`.

## Data availability

- **Control (general-website) images** are publicly available from the Kaggle datasets cited
  in the paper.
- **IOG screenshots** were provided by the Korea Racing Authority (KRA) and depict unlicensed
  gambling operations. As third-party data that identify illegal services, they are not
  redistributed here; they are available from the KRA on reasonable request, subject to the
  KRA's terms. See `DATA_AVAILABILITY.md`.
- **Derived data** (features, embeddings, labels, fold assignments, predictions) are released
  in `data/` with every site identifier replaced by an opaque id (`img_00001`, ...). Domains
  never appear in this repository.

## Running the LMM detector (`code/00`)

API keys are read from environment variables; set them before running:

```bash
export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...   GEMINI_API_KEY=...
```

Models used: `gpt-5-2025-08-07`, `claude-sonnet-4-5-20250929`, `gemini-2.5-flash`. The exact
classification prompt is embedded in the script.

## Third-party software

CLIP (Radford et al., 2021) via OpenCLIP (Ilharco et al.; https://doi.org/10.5281/zenodo.18794821),
and scikit-learn (Pedregosa et al., 2011).

## Citation

If you use this code or data, please cite the paper (citation to be added on acceptance) and
this archive (https://doi.org/10.5281/zenodo.20636926).
