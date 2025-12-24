"""Checkpoint utilities for model saving and loading."""

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tinytimemixers.config import TrainingConfig, TTMConfig


def save_model(
    model: nn.Module,
    path: str | Path,
    config: TTMConfig | None = None,
    training_config: TrainingConfig | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Save model to disk.

    Args:
        model: Model to save
        path: Save path (without extension)
        config: Model configuration
        training_config: Training configuration
        metadata: Additional metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = path.with_suffix(".pt")
    torch.save(model.state_dict(), weights_path)

    # Save config as JSON
    config_data = {}
    if config is not None:
        config_data["model_config"] = {
            k: v for k, v in config.__dict__.items() if not k.startswith("_")
        }
    if training_config is not None:
        config_data["training_config"] = {
            k: v for k, v in training_config.__dict__.items() if not k.startswith("_")
        }
    if metadata is not None:
        config_data["metadata"] = metadata

    config_path = path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2, default=str)

    print(f"Model saved to {weights_path}")


def load_model(
    model_class: type,
    path: str | Path,
    device: str | torch.device = "cpu",
    **model_kwargs,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model from disk.

    Args:
        model_class: Model class to instantiate
        path: Path to saved model (without extension)
        device: Device to load model to
        **model_kwargs: Additional model constructor arguments

    Returns:
        Tuple of (model, config_data)
    """
    path = Path(path)

    # Load config
    config_path = path.with_suffix(".json")
    config_data = {}
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)

    # Reconstruct config if available
    if "model_config" in config_data:
        config = TTMConfig(**config_data["model_config"])
        model_kwargs["config"] = config

    # Create model
    model = model_class(**model_kwargs)

    # Load weights
    weights_path = path.with_suffix(".pt")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    return model, config_data


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    config: TTMConfig | None = None,
    training_config: TrainingConfig | None = None,
    metrics: dict[str, float] | None = None,
    **extra,
):
    """Save full training checkpoint.

    Args:
        path: Save path
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        config: Model configuration
        training_config: Training configuration
        metrics: Training metrics
        **extra: Additional data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["model_config"] = config

    if training_config is not None:
        checkpoint["training_config"] = training_config

    if metrics is not None:
        checkpoint["metrics"] = metrics

    checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load to
        strict: Strict state dict loading

    Returns:
        Checkpoint data dictionary
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def cleanup_checkpoints(
    checkpoint_dir: str | Path,
    keep_best: bool = True,
    keep_last: int = 3,
):
    """Clean up old checkpoints, keeping only recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Always keep best.pt
        keep_last: Number of recent checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Keep the most recent ones
    to_remove = checkpoints[keep_last:]

    for ckpt in to_remove:
        ckpt.unlink()
        print(f"Removed checkpoint: {ckpt}")


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        save_best: bool = True,
    ):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            save_best: Track and save best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = float("inf")
        self.checkpoints: list[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int = 0,
        step: int = 0,
        metric: float | None = None,
        config: TTMConfig | None = None,
        **extra,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            step: Current step
            metric: Metric value (for best model tracking)
            config: Model configuration
            **extra: Additional data

        Returns:
            Path to saved checkpoint
        """
        # Save regular checkpoint
        filename = f"epoch_{epoch:04d}.pt"
        path = self.checkpoint_dir / filename

        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            config=config,
            metrics={"val_metric": metric} if metric is not None else None,
            **extra,
        )

        self.checkpoints.append(path)

        # Cleanup old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()

        # Save best if improved
        if self.save_best and metric is not None and metric < self.best_metric:
            self.best_metric = metric
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy(path, best_path)
            print(f"New best model saved (metric: {metric:.6f})")

        return path

    def load_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        """Load the best checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            device: Device to load to

        Returns:
            Checkpoint data or None if no best checkpoint
        """
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            return None

        return load_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        """Load the latest checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            device: Device to load to

        Returns:
            Checkpoint data or None if no checkpoints
        """
        latest = get_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            return None

        return load_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            device=device,
        )
