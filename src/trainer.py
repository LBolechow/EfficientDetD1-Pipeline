from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.evaluator import EvalResult, evaluate
from src.utils import CSV_HEADERS, format_time, prepare_batch_targets


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.epochs_without_improvement = 0
        self.best_map = 0.0

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def update(self, current_map: float) -> bool:
        """Returns True if a new best was achieved."""
        if current_map > self.best_map:
            self.best_map = current_map
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False


class CheckpointManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def save_best(self, model: torch.nn.Module) -> None:
        torch.save(model.state_dict(), self.cfg.paths.output_best)

    def save_last(self, model: torch.nn.Module) -> None:
        torch.save(model.state_dict(), self.cfg.paths.output_last)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        early_stopping: EarlyStopping,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_map": early_stopping.best_map,
                "epochs_without_improvement": early_stopping.epochs_without_improvement,
            },
            self.cfg.paths.checkpoint,
        )

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        early_stopping: EarlyStopping,
    ) -> int:
        """Returns the epoch to resume from."""
        ckpt_path = self.cfg.paths.checkpoint
        if not (self.cfg.training.resume and ckpt_path.exists()):
            return 0

        ckpt = torch.load(ckpt_path, map_location=self.cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"]
        early_stopping.best_map = ckpt["best_map"]
        early_stopping.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)

        print(
            f"--> Wznowiono od epoki {start_epoch + 1} "
            f"(Best mAP: {early_stopping.best_map:.4f}, "
            f"Bez poprawy od: {early_stopping.epochs_without_improvement} epok)"
        )
        return start_epoch


class CsvLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def init(self) -> None:
        with open(self.log_path, mode="w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)

    def log(
        self,
        epoch: int,
        train_loss: float,
        eval_result: EvalResult,
        lr: float,
        duration: float,
    ) -> None:
        row = [
            epoch,
            f"{train_loss:.4f}",
            f"{eval_result.val_loss:.4f}",
            f"{lr:.6f}",
            f"{duration:.2f}",
            *eval_result.stats_as_strings(),
        ]
        with open(self.log_path, mode="a", newline="") as f:
            csv.writer(f).writerow(row)


def train_one_epoch(
    train_bench: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
) -> float:
    train_bench.train()
    optimizer.zero_grad()
    total_loss = 0.0
    accum_steps = cfg.training.accumulation_steps

    for step, (images, targets) in enumerate(data_loader):
        images = torch.stack(images).to(cfg.device, non_blocking=True)
        bench_targets = prepare_batch_targets(targets, cfg.device, cfg.model.image_size)

        with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
            loss = train_bench(images, bench_targets)["loss"] / accum_steps

        scaler.scale(loss).backward()

        is_last_step = (step + 1) == len(data_loader)
        if (step + 1) % accum_steps == 0 or is_last_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                train_bench.parameters(), cfg.training.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(data_loader)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_bench: torch.nn.Module,
        eval_bench: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        cfg: Config,
    ) -> None:
        self.model = model
        self.train_bench = train_bench
        self.eval_bench = eval_bench
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.cfg = cfg

        self.early_stopping = EarlyStopping(patience=cfg.training.patience)
        self.checkpoint_mgr = CheckpointManager(cfg)
        self.csv_logger = CsvLogger(cfg.paths.csv_log)

    def setup(self) -> int:
        """Load checkpoint if resuming, otherwise init CSV. Returns start epoch."""
        start_epoch = self.checkpoint_mgr.load_checkpoint(
            self.model, self.optimizer, self.scheduler, self.scaler, self.early_stopping
        )
        if start_epoch == 0:
            self.csv_logger.init()
        return start_epoch

    def run(self) -> None:
        start_epoch = self.setup()

        for epoch in range(start_epoch, self.cfg.training.epochs):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                self.train_bench, self.train_loader, self.optimizer, self.scaler, self.cfg
            )

            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            epoch_dur = time.time() - epoch_start

            print(f"\n--- Ewaluacja (Epoka {epoch + 1}) ---")
            eval_result = evaluate(
                self.eval_bench,
                self.train_bench,
                self.val_loader,
                self.cfg.device,
                self.cfg.model.image_size,
            )

            is_new_best = self.early_stopping.update(eval_result.map_50_95)
            if is_new_best:
                self.checkpoint_mgr.save_best(self.model)
                print(f"✔ Nowy rekord mAP: {self.early_stopping.best_map:.4f}! Zapisano model best.")
            else:
                print(
                    f"✘ Brak poprawy (Best mAP: {self.early_stopping.best_map:.4f}). "
                    f"Licznik cierpliwości: {self.early_stopping.epochs_without_improvement}/{self.cfg.training.patience}"
                )

            self.checkpoint_mgr.save_checkpoint(
                epoch + 1, self.model, self.optimizer, self.scheduler, self.scaler, self.early_stopping
            )
            self.checkpoint_mgr.save_last(self.model)

            self.csv_logger.log(epoch + 1, train_loss, eval_result, current_lr, epoch_dur)

            print(
                f"Epoch {epoch + 1}/{self.cfg.training.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val_Loss: {eval_result.val_loss:.4f} | "
                f"mAP: {eval_result.map_50_95:.4f} | "
                f"Time: {format_time(epoch_dur)}"
            )

            if self.early_stopping.should_stop:
                print(f"\n[!] EARLY STOPPING: Brak poprawy przez {self.cfg.training.patience} epok.")
                break
