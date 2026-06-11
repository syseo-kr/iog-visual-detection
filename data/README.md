# Data dictionary

All files are keyed by an opaque image id (`img_00001` ... `img_02519`). Row order is identical
across `samples.csv`, `provenance_metadata.csv`, and the `.npy` matrices, so row *i* in every
file refers to the same image. Domains never appear here.

Analysis set: 2,519 screenshots (1,300 IOG, 1,219 control), in 1,775 near-duplicate clusters.

## samples.csv
| column | description |
|---|---|
| `id` | opaque image identifier |
| `label` | ground-truth class: 1 = IOG, 0 = control (verified labels; corrects 46 mis-filed images) |
| `label_folder` | original folder label before correction (for transparency only) |
| `cluster_id` | near-duplicate cluster index (perceptual-hash union-find, Hamming <= 10) |
| `is_dedup_rep` | 1 if this image is its cluster's representative (used for the deduplicated set) |
| `fold_full` | 0-4: test fold in the full-set group-aware 5-fold CV (groups = clusters, seed 42) |
| `fold_dedup` | 0-4 for representatives; -1 otherwise: test fold in the deduplicated 5-fold CV |
| `gpt_pred`, `claude_pred`, `gemini_pred` | LMM zero-shot predictions (1 = IOG, 0 = not) |
| `clip_zs_b32_piog`, `clip_zs_b32_pred` | CLIP ViT-B/32 zero-shot P(IOG) and thresholded (>=0.5) prediction |
| `clip_zs_l14_piog`, `clip_zs_l14_pred` | CLIP ViT-L/14 zero-shot P(IOG) and prediction |
| `phash` | 64-bit perceptual hash (hex), used by the pHash-kNN baseline |
| `md5` | MD5 of the standardized image (duplicate detection) |

## provenance_metadata.csv
Original (pre-standardization) capture metadata, used for the provenance-confound ablation.
| column | description |
|---|---|
| `id` | opaque image identifier |
| `width`, `height` | original pixel dimensions |
| `aspect` | width / height |
| `is_jpeg` | 1 if the source file was JPEG, else 0 |
| `filesize` | original file size in bytes |

## Matrices (`numpy.save` format, float32)
| file | shape | description |
|---|---|---|
| `features_handcrafted.npy` | 2519 x 2342 | HSV/RGB color histograms, color moments, edge density, gradient stats, LBP, HOG, on standardized 224x224 images |
| `clip_embeddings_vitb32.npy` | 2519 x 512 | frozen CLIP ViT-B/32 image embeddings |
| `clip_embeddings_vitl14.npy` | 2519 x 768 | frozen CLIP ViT-L/14 image embeddings |

## clip_text/
Text-prompt embeddings for the CLIP zero-shot classifier:
`clip_text_class_b32.npy` (class prompts) and `clip_text_variants_{b32,l14}.npz`
(the six prompt-set variants used in the prompt-sensitivity analysis).
