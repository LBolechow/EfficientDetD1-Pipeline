from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from src.utils import prepare_batch_targets


@dataclass
class EvalResult:
    stats: list[float]
    val_loss: float

    @property
    def map_50_95(self) -> float:
        return self.stats[0]

    @property
    def map_50(self) -> float:
        return self.stats[1]

    def stats_as_strings(self) -> list[str]:
        return [f"{s:.4f}" for s in self.stats]


def _rescale_prediction(
    box: np.ndarray,
    img_info: dict,
    img_size: int,
) -> tuple[float, float, float, float]:
    """Convert padded model-space box back to original image coordinates."""
    scale = max(img_info["height"], img_info["width"]) / img_size
    pad_x = (img_size - int(img_info["width"] / scale)) // 2
    pad_y = (img_size - int(img_info["height"] / scale)) // 2

    x_min = max(0.0, (box[0] - pad_x) * scale)
    y_min = max(0.0, (box[1] - pad_y) * scale)
    w = (box[2] - box[0]) * scale
    h = (box[3] - box[1]) * scale
    return x_min, y_min, w, h


def _collect_predictions(
    outputs: list[torch.Tensor],
    targets: list[dict],
    coco_gt: COCO,
    idx2cat: dict,
    img_size: int,
) -> list[dict]:
    predictions = []
    for i, out in enumerate(outputs):
        if out.numel() == 0:
            continue

        img_id = targets[i]["img_id"].item()
        img_info = coco_gt.loadImgs(img_id)[0]

        boxes = out[:, :4].cpu().numpy()
        scores = out[:, 4].cpu().numpy()
        labels = out[:, 5].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, w, h = _rescale_prediction(box, img_info, img_size)
            predictions.append({
                "image_id": img_id,
                "category_id": idx2cat[int(label)],
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "score": float(score),
            })

    return predictions


def _run_coco_eval(coco_gt: COCO, predictions: list[dict]) -> list[float]:
    if not predictions:
        return [0.0] * 12

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return list(coco_eval.stats)


@torch.no_grad()
def evaluate(
    eval_bench: torch.nn.Module,
    train_bench: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    img_size: int,
) -> EvalResult:
    eval_bench.eval()
    train_bench.eval()

    coco_gt = data_loader.dataset.coco
    idx2cat = data_loader.dataset.idx2cat
    total_val_loss = 0.0
    all_predictions = []

    for images, targets in data_loader:
        images = torch.stack(images).to(device, non_blocking=True)
        bench_targets = prepare_batch_targets(targets, device, img_size)

        total_val_loss += train_bench(images, bench_targets)["loss"].item()

        outputs = eval_bench(images)
        all_predictions.extend(
            _collect_predictions(outputs, targets, coco_gt, idx2cat, img_size)
        )

    avg_val_loss = total_val_loss / len(data_loader)
    stats = _run_coco_eval(coco_gt, all_predictions)

    return EvalResult(stats=stats, val_loss=avg_val_loss)
