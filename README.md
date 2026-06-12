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
│   ├── samples.csv         Per-image: id, label, cluster\_id, folds, LMM \& CLIP zero-shot predictions, phash, md5
│   ├── provenance\_metadata.csv  Per-image: original width/height/aspect/format/filesize (for the confound ablation)
│   ├── features\_handcrafted.npy 2519 x 2342 handcrafted feature matrix (row order = samples.csv)
│   ├── clip\_embeddings\_vitb32.npy 2519 x 512 frozen CLIP ViT-B/32 image embeddings
│   ├── clip\_embeddings\_vitl14.npy 2519 x 768 frozen CLIP ViT-L/14 image embeddings
│   ├── clip\_text/          CLIP text-prompt embeddings (zero-shot class \& prompt variants)
│   └── README.md           Data dictionary
├── code/                   The exact analysis scripts used to perform the study
│   ├── 00\_run\_lmm\_detection.py   Query GPT-5 / Claude / Gemini on screenshots -> predictions
│   ├── 01\_extract\_features.py    Handcrafted features from raw screenshots
│   ├── 02\_cluster\_and\_split.py   Perceptual-hash near-duplicate clustering + group-aware folds
│   ├── 03\_run\_clip.py / 03b\_clip\_extra.py  CLIP image/text embeddings \& zero-shot
│   ├── 04\_eval\_fullset.py        Full-set \& dedup CV (classical, CLIP B/32, LMMs, ensembles)
│   ├── 04b\_eval\_clip\_l14.py      CLIP ViT-L/14 zero-shot \& linear-probe
│   ├── 05\_complementarity.py     LMM-missed vs low-cost-model coverage, residual hard core
│   └── 06\_ablation.py            Leakage (naive vs group split) + provenance (metadata-only) confounds
├── requirements.txt
├── LICENSE                 Code: MIT
├── LICENSE-data            Derived data: CC BY 4.0
└── DATA\_AVAILABILITY.md    Data/code availability statement
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
writes `reproduced\_results.json`. Runtime is a few minutes (the RBF-SVM on the full feature
matrix is the slow step). All folds are reconstructed deterministically (seed 42), so the
numbers match the paper exactly.

`code/` contains the analysis scripts **as run**; their paths reflect the authors' working
environment and the upstream steps (`00`, `01`, `03`) require the raw screenshots. For
verification on the released data, use `reproduce.py`.

## Notes on inputs and protocol

* **Input standardization (applied to every model).** Before any feature extraction,
embedding, or model query, each screenshot is standardized identically: decoded,
resized to a 224x224 square by bicubic interpolation, and re-encoded through a single
JPEG pass at quality 90. The handcrafted-feature pipeline, CLIP (zero-shot and
linear-probe), and the three LMMs all receive this standardized image, so capture
provenance (absolute resolution, aspect ratio, file format) is removed by construction
for every method. The image-folder inputs to `00\_run\_lmm\_detection.py` and
`03\_run\_clip.py` are these standardized images (the same standardization implemented
in `01\_extract\_features.py`).
* **Deduplicated-set representatives.** For the deduplicated (novel-template) evaluation,
each near-duplicate cluster contributes one representative, drawn from the cluster's
majority-label members so the representative's label always matches the label it is
evaluated under. Of 1,775 clusters, 8 are mixed-label (59 images); this rule affects
only those clusters' representative selection.

## Data availability

* **Control (general-website) images** are publicly available from the Kaggle datasets cited
in the paper.
* **IOG screenshots** were provided by the Korea Racing Authority (KRA) and depict unlicensed
gambling operations. As third-party data that identify illegal services, they are not
redistributed here; they are available from the KRA on reasonable request, subject to the
KRA's terms. See `DATA\_AVAILABILITY.md`.
* **Derived data** (features, embeddings, labels, fold assignments, predictions) are released
in `data/` with every site identifier replaced by an opaque id (`img\_00001`, ...). Domains
never appear in this repository.

## Running the LMM detector (`code/00`)

API keys are read from environment variables; set them before running:

```bash
export OPENAI\_API\_KEY=...   ANTHROPIC\_API\_KEY=...   GEMINI\_API\_KEY=...
```

Models used: `gpt-5-2025-08-07`, `claude-sonnet-4-5-20250929`, `gemini-2.5-flash`. The exact
classification prompt is embedded in the script.

## Third-party software

CLIP (Radford et al., 2021) via OpenCLIP (Ilharco et al.; doi.org/10.5281/zenodo.18794821),
and scikit-learn (Pedregosa et al., 2011).

## Citation

If you use this code or data, please cite the paper (citation to be added on acceptance) and
this archive.

