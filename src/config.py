from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    image_size: int


@dataclass
class TrainingConfig:
    batch_size: int
    accumulation_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    patience: int
    seed: int
    resume: bool


@dataclass
class DataloaderConfig:
    num_workers: int
    pin_memory: bool


@dataclass
class PathConfig:
    base: Path
    annotations_dir: Path
    images_dir: Path
    train_images: Path
    val_images: Path
    train_json: Path
    val_json: Path
    output_best: Path
    output_last: Path
    checkpoint: Path
    csv_log: Path

    @classmethod
    def from_dict(cls, raw: dict, base: Path) -> PathConfig:
        annotations_dir = base / raw["annotations_dir"]
        images_dir = base / raw["images_dir"]
        return cls(
            base=base,
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            train_images=images_dir / raw["train_images"],
            val_images=images_dir / raw["val_images"],
            train_json=annotations_dir / raw["train_json"],
            val_json=annotations_dir / raw["val_json"],
            output_best=base / raw["output_best"],
            output_last=base / raw["output_last"],
            checkpoint=base / raw["checkpoint"],
            csv_log=base / raw["csv_log"],
        )


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig
    dataloader: DataloaderConfig
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, base_dir: str | Path | None = None) -> Config:
        yaml_path = Path(yaml_path)
        base = Path(base_dir) if base_dir else yaml_path.parent.parent

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            paths=PathConfig.from_dict(raw["paths"], base),
            dataloader=DataloaderConfig(**raw["dataloader"]),
        )

    @property
    def use_amp(self) -> bool:
        return self.device.type == "cuda"