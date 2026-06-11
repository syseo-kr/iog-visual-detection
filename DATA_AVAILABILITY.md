# Data and code availability statement

**Code.** All analysis code is in `code/`, with `reproduce.py` regenerating every reported
result table from the released derived data. The repository is archived at Zenodo
(DOI: to be assigned on deposit).

**Control images.** The general-website (non-IOG) control images are publicly available from
the Kaggle datasets cited in the paper.

**IOG screenshots.** The illegal-online-gambling screenshots were provided by the Korea Racing
Authority (KRA) and depict unlicensed gambling operations. As third-party data that identify
illegal services, their public redistribution raises legal and safety concerns; they are
therefore not deposited here and are available from the KRA upon reasonable request, subject to
the KRA's terms.

**Derived data.** To enable full reproduction without the raw screenshots, all derived data
required to regenerate every table and figure are released in `data/`: extracted handcrafted
features, CLIP image embeddings, perceptual hashes, ground-truth labels, cross-validation fold
assignments, original capture metadata, and per-model predictions, with each site identifier
replaced by an opaque id. Domains do not appear in the release. The raw screenshots are needed
only to re-extract features and embeddings; every downstream result is reproducible from the
deposited derived data alone.

**Third-party software.** CLIP (Radford et al., 2021) via OpenCLIP (Ilharco et al.; Zenodo DOI)
and scikit-learn (Pedregosa et al., 2011).
