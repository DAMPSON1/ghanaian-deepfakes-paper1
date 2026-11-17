# Paper 1 — public data and code availability

## Ghanaian authentic facial-image dataset

**Published dataset:** [Ghanaian Facial Image Dataset for Dataset-Source Effects in Image-Level Deepfake Detection](https://zenodo.org/records/24320872)  
**DOI:** https://doi.org/10.5281/zenodo.24320872  
**Published date:** 2026-11-17 (as shown on the Zenodo record)

The Zenodo deposit lists 43,986 authentic Ghanaian facial-image crops from 72 publicly accessible video clips featuring 18 public figures, resized to 224 × 224 pixels. It contains:
- `paper1_ghanaian_real_train.zip`: 25,058 images
- `paper1_ghanaian_real_val.zip`: 7,600 images
- `paper1_ghanaian_real_test.zip`: 11,328 images
- `dataset_summary.json`
- `ghanaian_authentic_image_manifest.csv`

The research author confirmed redistribution rights for this Ghanaian image collection. Use remains subject to the licensing and reuse conditions stated in the Zenodo record and other applicable obligations.

## Code and recorded results

This public GitHub repository provides:
- `source_discrimination.py`, the author-supplied authentic-only ResNet18 source-discrimination script.
- `results/test_metrics.json`, aggregate held-out metrics.
- `results/training_history.csv`, recorded training history.

**Scope distinction:** The Zenodo archives contain the 43,986 original-baseline Ghanaian crops; the source-control experiment used 3,600 selected Ghanaian images plus 26,444 authentic FF++ images. The original selection/split manifest for the source-control run is not in this public repository, and the Zenodo baseline partitions are not necessarily identical to that control experiment's splits.

FaceForensics++ media are not redistributed in Zenodo. The FF++ research dataset has its own access conditions: https://github.com/ondyari/FaceForensics.

## Ethics and research contact

Publication of a dataset does not by itself establish whether institutional ethics approval or informed consent was required or obtained. Any response to a journal's human-data requirements must truthfully reflect the underlying approvals, exemptions, or consents.

Corresponding author: Eric Amoah Dampson — eadampson@ualr.edu.
