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

## Data availability and restrictions

**This is a public code-and-aggregate-results repository, not a public download of the Ghanaian face-image dataset or FF++ image files.** The source videos were publicly viewable; rights to redistribute derived identifiable facial images are separate. The image datasets and original individual-level prediction/manifests are not distributed here. Please consult [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

For research inquiries, contact **eadampson@ualr.edu**.
