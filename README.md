# LEGO Brick Detection with EfficientDet D1

This repository contains a custom **EfficientDet D1 training and evaluation pipeline** developed for a master's thesis comparing CNN-based and Transformer-based object detectors. The EfficientDet architecture itself comes from the `effdet` library; this repository implements the surrounding data-processing, training, checkpointing, and COCO evaluation workflow.

## The Study

The study compared five object detectors to investigate accuracy–speed trade-offs for a potential LEGO brick detection application:

- **CNN-based:** Faster R-CNN (ResNet-50), EfficientDet D1, YOLOv8s
- **Transformer-based / hybrid:** RT-DETR-L, Deformable DETR

The experiments used a custom synthetic dataset, consistent data splits, and an NVIDIA RTX 4060 (8 GB VRAM). Each model used an appropriate implementation and training configuration, so the study compares the selected models rather than isolating architecture as the only variable.

## Dataset

The full dataset consists of **600 synthetic images** (640 × 480 px), generated in **BrickLink Studio 2.0** and annotated in **CVAT**. Each image contains all 10 selected LEGO brick classes, with **6,000 annotated instances** in total.

**10 classes:**

- Geometrically similar: `2x2`, `2x3`, `1x1`, `1x2`, `1x3`, `2x2corner`, `1x1round`, `1x1curved`
- Distinctive shapes: `2x2roundtile`, `panel1x2x1`

The scenes vary in brick placement, orientation, color, background, camera position, lighting, and simulated surface wear. Annotations use COCO format.

Four dataset size variants were evaluated: **100, 200, 400, and 600 images**, each with an **80% train / 10% validation / 10% test** split. For the 600-image variant, that means 480 training, 60 validation, and 60 test images.

The held-out test scenes were intentionally more difficult than the training and validation scenes, including brick overlap, more extreme lighting, and simulated scratches or UV degradation. Additional qualitative tests explored unfamiliar brick types and challenging visual appearances.

## Results

All values below refer to the **600-image dataset variant**. Validation scores correspond to the best checkpoint; training times are the total durations reported for this dataset size, with early stopping where applicable.

### Validation set (600-image dataset variant)

| Model | mAP@50 | mAP@50–95 | Training time |
|---|---:|---:|---:|
| RT-DETR-L | 0.9460 | **0.8845** | 5,116.63 s |
| YOLOv8s | 0.9487 | 0.8733 | **564.27 s** |
| EfficientDet D1 | **0.9511** | 0.8482 | 1,217.50 s |
| Deformable DETR | 0.9234 | 0.8153 | 12,944.40 s |
| Faster R-CNN | 0.8911 | 0.7668 | 1,507.90 s |

### Held-out test set (60 images from the 600-image variant)

| Model | mAP@50 | mAP@50–95 | Inference (ms/image) |
|---|---:|---:|---:|
| RT-DETR-L | **0.7170** | **0.6530** | 21.2 |
| YOLOv8s | 0.6900 | 0.6070 | **8.8** |
| Deformable DETR | 0.6642 | 0.5658 | 79.2 |
| EfficientDet D1 | 0.6553 | 0.5548 | 29.7 |
| Faster R-CNN | 0.5560 | 0.4456 | 33.1 |

**Key findings:**

- **RT-DETR-L achieved the highest test mAP@50–95 (0.6530)** on the 600-image dataset variant.
- **YOLOv8s was the fastest** (8.8 ms/image), providing a strong accuracy–speed trade-off.
- **EfficientDet D1 was competitive on validation** (0.8482 mAP@50–95) but dropped to 0.5548 on the deliberately harder test set.
- The validation–test gap partly reflects the different difficulty of test scenes; it is not by itself proof of ordinary overfitting.
- Additional qualitative analysis exposed difficulties with unfamiliar bricks and unusual surface properties.

These are results for the tested model configurations on this synthetic dataset, not general model rankings.

## Project Structure

```text
.
├── train.py                  # entry point
├── configs/
│   └── effdet_d1.yaml        # training configuration
└── src/
    ├── config.py             # load YAML configuration
    ├── dataset.py            # LegoDataset and augmentations
    ├── evaluator.py          # predictions, rescaling, COCO evaluation
    ├── trainer.py            # training, early stopping, checkpoints
    └── utils.py              # seeds, collate function, batch padding
```

## Setup

```bash
pip install torch torchvision effdet pycocotools albumentations opencv-python pyyaml
```

Expected training/validation dataset structure:

```text
.
├── annotations/
│   ├── instances_Train.json
│   └── instances_Validation.json
└── images/
    ├── Train/
    └── Validation/
```

The separately prepared test set was used for final evaluation, not training.

## Training

```bash
python train.py
```

Training can resume from `effdet_last_checkpoint.pth` if present and enabled. For a fresh run, set `resume: false` in the configuration or remove the checkpoint.

Selected options from `configs/effdet_d1.yaml`:

```yaml
training:
  batch_size: 8
  accumulation_steps: 2  # effective batch size: 16
  epochs: 200
  learning_rate: 5.0e-4
  patience: 50
```

### Training Features

- AdamW optimizer and CosineAnnealingLR scheduler
- Automatic mixed precision (AMP) when CUDA is available
- Gradient accumulation (effective batch size 16)
- Gradient clipping (maximum norm 5.0)
- Early stopping based on validation mAP@50–95
- Checkpointing of model and training state
- COCO-style evaluation and per-epoch CSV metric logging

## Environment

- WSL2 / Ubuntu 22.04
- NVIDIA RTX 4060 8 GB
- PyTorch with CUDA and AMP

Exact dependency and CUDA versions should match the installation used to run the repository.

## Other Models from the Study

YOLOv8s, RT-DETR-L, Faster R-CNN and Deformable DETR were evaluated as part of the broader study. Their implementations are maintained separately and are not included in this repository.
