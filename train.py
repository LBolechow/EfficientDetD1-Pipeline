from __future__ import annotations

from pathlib import Path

import torch
from effdet import create_model, DetBenchPredict, DetBenchTrain
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import LegoDataset
from src.trainer import Trainer
from src.utils import collate_fn, set_seed


CONFIG_PATH = Path(__file__).parent / "configs" / "effdet_d1.yaml"


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_ds = LegoDataset(cfg.paths.train_images, cfg.paths.train_json, cfg.model.image_size)
    val_ds = LegoDataset(cfg.paths.val_images, cfg.paths.val_json, cfg.model.image_size)

    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def build_model(cfg: Config, num_classes: int) -> tuple:
    model = create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=num_classes,
        image_size=(cfg.model.image_size, cfg.model.image_size),
    )
    train_bench = DetBenchTrain(model).to(cfg.device)
    eval_bench = DetBenchPredict(model).to(cfg.device)
    return model, train_bench, eval_bench


def build_training_components(cfg: Config, train_bench: torch.nn.Module) -> tuple:
    optimizer = torch.optim.AdamW(
        train_bench.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)
    return optimizer, scheduler, scaler


def main() -> None:
    cfg = Config.from_yaml(CONFIG_PATH)
    set_seed(cfg.training.seed)

    train_loader, val_loader = build_dataloaders(cfg)
    num_classes = train_loader.dataset.num_classes

    model, train_bench, eval_bench = build_model(cfg, num_classes)
    optimizer, scheduler, scaler = build_training_components(cfg, train_bench)

    trainer = Trainer(
        model=model,
        train_bench=train_bench,
        eval_bench=eval_bench,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cfg=cfg,
    )
    trainer.run()


if __name__ == "__main__":
    main()
