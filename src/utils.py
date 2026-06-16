from __future__ import annotations

import random
import time
from typing import Sequence

import numpy as np
import torch


CSV_HEADERS = [
    "Epoch", "Train_Loss", "Val_Loss", "LR", "Time_Sec",
    "mAP_0.50:0.95_All", "mAP_0.50_All", "mAP_0.75_All",
    "mAP_Small", "mAP_Medium", "mAP_Large",
    "AR_1", "AR_10", "AR_100", "AR_Small", "AR_Medium", "AR_Large",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def format_time(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def collate_fn(batch: list) -> tuple:
    return tuple(zip(*batch))


def prepare_batch_targets(
    targets: Sequence[dict],
    device: torch.device,
    img_size: int,
) -> dict:
    """Pad bounding boxes and labels to the same length within a batch."""
    max_boxes = max(t["bboxes"].shape[0] for t in targets)
    max_boxes = max(max_boxes, 1)  # avoid zero-size tensors

    batch_size = len(targets)
    batch_boxes = torch.full((batch_size, max_boxes, 4), -1, dtype=torch.float32, device=device)
    batch_labels = torch.full((batch_size, max_boxes), -1, dtype=torch.float32, device=device)

    for i, t in enumerate(targets):
        n = t["bboxes"].shape[0]
        if n > 0:
            batch_boxes[i, :n] = t["bboxes"].to(device)
            batch_labels[i, :n] = t["labels"].to(device)

    return {
        "bbox": batch_boxes,
        "cls": batch_labels,
        "img_scale": torch.ones(batch_size, device=device),
        "img_size": torch.tensor(
            [(img_size, img_size)] * batch_size,
            device=device,
            dtype=torch.float32,
        ),
    }
