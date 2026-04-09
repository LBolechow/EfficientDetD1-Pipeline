# EfficientDet Training Pipeline

## Overview

This repository provides a complete training and evaluation pipeline for object detection using the EfficientDet architecture implemented via the `effdet` library. The workflow is designed for COCO-format datasets and includes support for mixed precision training, gradient accumulation, checkpointing, and detailed evaluation using COCO metrics.

The implementation is structured to be reproducible, scalable, and suitable for experimentation as well as production-level training.

---

## Features

* Training using EfficientDet (D1 backbone by default)
* Full COCO-format dataset support
* Automatic mixed precision (AMP)
* Gradient accumulation
* Cosine annealing learning rate scheduler
* Early stopping based on validation mAP
* Resume training from checkpoint
* COCO evaluation metrics (mAP, AR)
* CSV logging of training progress

---

## Project Structure

```
project_root/
│
├── annotations/
│   ├── instances_Train.json
│   └── instances_Validation.json
│
├── images/
│   ├── Train/
│   └── Validation/
│
├── effdet_best.pth
├── effdet_last.pth
├── effdet_last_checkpoint.pth
├── training_results_effdet.csv
└── train.py
```

---

## Requirements

Install dependencies before running the training:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install albumentations
pip install pycocotools
pip install effdet
```

---

## Dataset Format

The dataset must follow the COCO annotation format.

### Images

* Training images: `images/Train/`
* Validation images: `images/Validation/`

### Annotations

* Training annotations: `annotations/instances_Train.json`
* Validation annotations: `annotations/instances_Validation.json`

Each annotation file must include:

* `images`
* `annotations`
* `categories`

Bounding boxes are expected in COCO format: `[x, y, width, height]`.

---

## Configuration

All configuration parameters are defined in the `Config` class:

* `IMG_SIZE` – input image resolution
* `BATCH_SIZE` – batch size per iteration
* `ACCUMULATION_STEPS` – gradient accumulation steps
* `EPOCHS` – total number of epochs
* `LR` – learning rate
* `PATIENCE` – early stopping patience
* `DEVICE` – training device (CPU/GPU)

Paths for datasets, checkpoints, and logs are also defined in this class.

---

## Training

To start training:

```bash
python train.py
```

### Training Behavior

* Model: EfficientDet-D1 initialized with pretrained weights
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR
* Loss scaling via AMP when CUDA is available
* Gradient clipping applied to stabilize training

---

## Evaluation

Evaluation is performed at the end of each epoch using COCO metrics.

Metrics include:

* mAP (IoU 0.50:0.95)
* mAP (IoU 0.50)
* mAP (IoU 0.75)
* mAP for small, medium, large objects
* Average Recall (AR) at different detection limits

---

## Checkpointing

The training process automatically saves:

* `effdet_best.pth` – best model based on mAP
* `effdet_last.pth` – last epoch model
* `effdet_last_checkpoint.pth` – full training state

If `RESUME_TRAINING = True`, training resumes from the last checkpoint.

---

## Logging

Training progress is stored in:

```
training_results_effdet.csv
```

Each row contains:

* Epoch
* Training loss
* Validation loss
* Learning rate
* Epoch duration
* COCO evaluation metrics

---

## Early Stopping

Training stops automatically when no improvement in mAP is observed for a number of epochs defined by `PATIENCE`.

---

## Reproducibility

A fixed random seed is used to ensure reproducible results across runs. This includes:

* Python random
* NumPy
* PyTorch (CPU and CUDA)

---

## Notes

* Input images are resized while preserving aspect ratio and padded to a square resolution.
* Bounding boxes are converted internally to match EfficientDet expectations.
* Empty annotations are handled safely during both training and evaluation.

---

## License

This project does not include a license by default. Add an appropriate license file if distribution is intended.

---

## Acknowledgments

* EfficientDet implementation provided by the `effdet` library
* COCO evaluation tools via `pycocotools`
