# LEGO Brick Detection with EfficientDet D1

Part of a broader comparative study on CNN vs. Transformer-based object detection architectures. This repo contains the EfficientDet D1 training pipeline used in the experiment, built from scratch with a custom data pipeline, COCO-format evaluation, and full training state checkpointing.

## The Study

The goal was to find the best model for a real-world LEGO brick detection application, comparing five architectures across two major design philosophies:

**CNN-based:** Faster R-CNN (ResNet-50), EfficientDet D1, YOLOv8s

**Transformer-based:** RT-DETR-L, Deformable DETR

All models were trained on a custom synthetic dataset generated in BrickPoint Studio 2.0, annotated in CVAT, and evaluated under identical conditions on an NVIDIA RTX 4060 (8GB VRAM).

## Dataset

600 synthetic images generated in Studio 2.0, each containing all 10 brick classes. 6000 annotated instances total.

**10 brick classes across two groups:**

Geometrically similar (the hard ones): `2x2`, `2x3`, `1x1`, `1x2`, `1x3`, `2x2corner`, `1x1round`, `1x1curved`

Unique shapes: `2x2roundtile`, `panel1x2x1`

Each scene was randomized across brick placement, orientation, color, background, camera angle, lighting type/direction, and surface wear simulation. Images are 640x480px, exported in COCO 1.0 format.

The dataset was split into four size variants (100 / 200 / 400 / 600 images) to measure how each architecture scales with data volume.

## Results

### Validation set (best checkpoint per dataset size)

| Model | mAP@50 | mAP@50-95 | Train time (600 imgs) |
|---|---|---|---|
| RT-DETR-L | 0.946 | **0.885** | 898s |
| YOLOv8s | 0.949 | 0.873 | **188s** |
| EfficientDet D1 | 0.951 | 0.848 | 1218s |
| Deformable DETR | 0.923 | 0.815 | 3023s |
| Faster R-CNN | 0.891 | 0.767 | 1508s |

### Test set (600 images, harder scenes)

| Model | mAP@50 | mAP@50-95 | Inference [ms/img] |
|---|---|---|---|
| Deformable DETR | - | **0.566** | 79.2 |
| RT-DETR-L | 0.717 | 0.653 | 21.2 |
| YOLOv8s | 0.690 | 0.607 | **8.8** |
| EfficientDet D1 | 0.655 | 0.555 | 29.7 |
| Faster R-CNN | 0.556 | 0.446 | 33.1 |

The gap between validation and test results is intentional: the test set contained significantly harder scenes (overlapping bricks, unknown brick types, varied visual conditions). Deformable DETR, which appeared unimpressive on validation, generalized best to the hard test set, though at the cost of a 79ms inference time that makes it impractical for real-time use.

EfficientDet D1 showed a notable gap between its validation performance and test set generalization, which was the main finding for this architecture.

## Project Structure

```
.
├── train.py                  # entry point
├── configs/
│   └── effdet_d1.yaml        # all hyperparameters here
└── src/
    ├── config.py             # dataclasses loading the YAML
    ├── dataset.py            # LegoDataset + albumentations pipeline
    ├── evaluator.py          # COCO eval, prediction collection, rescaling
    ├── trainer.py            # Trainer, EarlyStopping, CheckpointManager, CsvLogger
    └── utils.py              # seed, collate_fn, batch padding
```

## Setup

```bash
pip install torch torchvision effdet pycocotools albumentations opencv-python pyyaml
```

Dataset structure expected:

```
.
├── annotations/
│   ├── instances_Train.json
│   └── instances_Validation.json
└── images/
    ├── Train/
    └── Validation/
```

## Training

```bash
python train.py
```

Resumes automatically from `effdet_last_checkpoint.pth` if it exists. To start fresh, set `resume: false` in the config or delete the checkpoint.

Key config options in `configs/effdet_d1.yaml`:

```yaml
training:
  batch_size: 8
  accumulation_steps: 2   # effective batch = 16
  epochs: 200
  learning_rate: 5.0e-4
  patience: 50
```

## Training Setup

- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- AMP (automatic mixed precision) when CUDA is available
- Gradient accumulation (effective batch size 16)
- Gradient clipping (max norm 5.0)
- Early stopping on mAP@50-95
- Full training state checkpointing (model, optimizer, scheduler, scaler, early stopping counter)
- Per-epoch CSV logging with all COCO metrics

## Environment

- WSL2 / Ubuntu 22.04
- CUDA 12.8 + cuDNN (deterministic mode)
- NVIDIA RTX 4060 8GB
- PyTorch with AMP

## Other Models from the Study

The remaining four architectures (YOLOv8s, RT-DETR-L, Faster R-CNN, Deformable DETR) were implemented separately and are not included in this repository.
