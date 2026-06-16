from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class LegoDataset(Dataset):
    def __init__(self, img_dir: str | Path, ann_path: str | Path, img_size: int) -> None:
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_path))
        self.ids = list(self.coco.imgs.keys())
        self.img_size = img_size

        cat_ids = sorted(self.coco.getCatIds())
        self.cat2idx = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.idx2cat = {i + 1: cat_id for i, cat_id in enumerate(cat_ids)}

        self.transform = build_transforms(img_size)

    @property
    def num_classes(self) -> int:
        return len(self.cat2idx)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        img_id = self.ids[index]
        image = self._load_image(img_id)
        boxes, labels = self._load_annotations(img_id)

        sample = self.transform(image=image, bboxes=boxes, labels=labels)

        image_tensor = sample["image"]
        transformed_boxes = np.array(sample["bboxes"]) if sample["bboxes"] else np.zeros((0, 4), dtype=np.float32)
        transformed_labels = np.array(sample["labels"], dtype=np.int64)

        # [y1, x1, y2, x2]
        if len(transformed_boxes) > 0:
            transformed_boxes = transformed_boxes[:, [1, 0, 3, 2]]

        return image_tensor, {
            "bboxes": torch.as_tensor(transformed_boxes, dtype=torch.float32),
            "labels": torch.as_tensor(transformed_labels, dtype=torch.int64),
            "img_id": torch.tensor([img_id]),
        }

    def _load_image(self, img_id: int) -> np.ndarray:
        img_info = self.coco.loadImgs(img_id)[0]
        image = cv2.imread(str(self.img_dir / img_info["file_name"]))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _load_annotations(self, img_id: int) -> tuple[list, list]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat2idx[ann["category_id"]])
        return boxes, labels
