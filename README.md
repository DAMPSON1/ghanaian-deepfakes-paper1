# Dataset-Source Effects in Image-Level Deepfake Detection: A Ghanaian Facial Image Case Study

**Paper 1 — public reproducibility materials.**  
Corresponding author: Eric Amoah Dampson (eadampson@ualr.edu).

This repository contains the authentic-only dataset-source discrimination control associated with Paper 1. It is **separate from the ongoing Paper 2 video-level study**, which is not included here.

## Research objective

Assess whether a ResNet18 classifier can distinguish **authentic Ghanaian facial images** (class 0) from **authentic FaceForensics++ (FF++) facial images** (class 1) **without any manipulated images** in the source-discrimination experiment. This is a test of dataset-provenance cues, not evidence that the model recognizes deepfake manipulations.

## Repository contents

- [`source_discrimination.py`](source_discrimination.py): original training and evaluation script supplied by the author.
- [`results/test_metrics.json`](results/test_metrics.json): recorded held-out metrics.
- [`results/training_history.csv`](results/training_history.csv): recorded validation/training history.
- [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md): information about image access, limitations, and research contact.

## Recorded held-out results

- Test: 6,034 authentic images (800 Ghanaian, 5,234 FF++); source-video/subject-grouped splitting.
- Accuracy: **99.97%**; balanced accuracy: **99.98%**; ROC-AUC: **0.99999976**.
- Confusion matrix (actual rows, predicted columns; Ghanaian then FF++): `[[800, 0], [2, 5232]]`.
- Seed 42; ten epochs; best epoch 8; cap of 200 images per group.

## Reproducing the experiment

Install Python packages: PyTorch, torchvision, Pillow, numpy, scikit-learn, matplotlib.

The original script supports two input layouts: either `--data-root` containing `ghanaian_real/<group>/` and `ffpp_real/<group>/`, or `--manifest` with columns `path,source,group_id`. Example:

```bash
python source_discrimination.py --data-root /path/to/data --manifest /path/to/approved_manifest.csv --output-dir results_local --epochs 10 --batch-size 32 --seed 42 --max-images-per-group 200
```

This example requires the original approved input files; it cannot run on the released metadata alone. Re-execution has **not** been independently demonstrated from this public repository.

## Public dataset and data availability

The **43,986 authentic Ghanaian face crops used in the original cross-source baseline** are published separately on Zenodo:

- **Dataset record:** https://zenodo.org/records/24320872
- **DOI:** https://doi.org/10.5281/zenodo.24320872
- **Files deposited:** `paper1_ghanaian_real_train.zip` (25,058 images), `paper1_ghanaian_real_val.zip` (7,600), `paper1_ghanaian_real_test.zip` (11,328), `dataset_summary.json`, and `ghanaian_authentic_image_manifest.csv`.

This repository provides public code and aggregated results, rather than duplicating the Zenodo ZIP archives. The original authentic-only *source-control* experiment selected a subset of Ghanaian images (3,600 in total) and authentic FF++ images (26,444 in total); the Zenodo split archives describe the original Ghanaian baseline image partitions and **should not be assumed to reproduce the source-control split without its original selection manifest**.

FaceForensics++ images are **not** included in the Zenodo deposit; obtain them through the dataset's official access process. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for details.

For research inquiries, contact **eadampson@ualr.edu**.
