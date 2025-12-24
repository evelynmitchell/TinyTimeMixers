"""Training loop for TinyTimeMixers."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinytimemixers.config import TrainingConfig, TTMConfig
from tinytimemixers.training.losses import ForecastLoss, get_loss_fn
from tinytimemixers.training.optimizer import (
    WarmupScheduler,
    create_optimizer,
    create_scheduler,
    get_parameter_count,
)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    train_loss: float = 0.0
    val_loss: float | None = None
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    epoch_time: float = 0.0
    best_val_loss: float = float("inf")
    history: list[dict[str, Any]] = field(default_factory=list)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """Initialize EarlyStopping.

        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement required
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Training loop for TTM models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        training_config: TrainingConfig | None = None,
        model_config: TTMConfig | None = None,
        loss_fn: ForecastLoss | None = None,
        device: str = "auto",
        checkpoint_dir: str | Path = "./checkpoints",
    ):
        """Initialize Trainer.

        Args:
            model: TTM model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            training_config: Training configuration
            model_config: Model configuration (for loss type)
            loss_fn: Loss function (overrides config if provided)
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.training_config = training_config or TrainingConfig()
        self.model_config = model_config

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif model_config is not None:
            self.loss_fn = get_loss_fn(model_config.loss_type)
        else:
            self.loss_fn = get_loss_fn("mse")

        # Setup optimizer
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=self.training_config.betas,
            freeze_backbone=self.training_config.freeze_backbone,
        )

        # Setup scheduler with warmup
        base_scheduler = create_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            num_epochs=self.training_config.num_epochs,
        )
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_steps=self.training_config.warmup_steps,
            base_scheduler=base_scheduler,
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.metrics = TrainingMetrics()
        self.global_step = 0
        self.current_epoch = 0

        # Log model info
        num_params = get_parameter_count(self.model)
        print(f"Model parameters: {num_params:,}")
        print(f"Device: {self.device}")

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            context, target = batch
            context = context.to(self.device)
            target = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(context)
            loss = self.loss_fn(pred, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            context, target = batch
            context = context.to(self.device)
            target = target.to(self.device)

            pred = self.model(context)
            loss = self.loss_fn(pred, target)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        num_epochs: int | None = None,
        early_stopping: bool = True,
        patience: int = 10,
        save_best: bool = True,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """Run training loop.

        Args:
            num_epochs: Number of epochs (overrides config)
            early_stopping: Enable early stopping
            patience: Early stopping patience
            save_best: Save best model checkpoint
            verbose: Print progress

        Returns:
            Training metrics
        """
        num_epochs = num_epochs or self.training_config.num_epochs
        early_stopper = EarlyStopping(patience=patience) if early_stopping else None

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate() if self.val_loader is not None else None

            epoch_time = time.time() - start_time

            # Update metrics
            self.metrics.train_loss = train_loss
            self.metrics.val_loss = val_loss
            self.metrics.epoch = epoch
            self.metrics.step = self.global_step
            self.metrics.epoch_time = epoch_time
            self.metrics.learning_rate = self.scheduler.get_last_lr()[0]

            # Track history
            self.metrics.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.metrics.learning_rate,
                }
            )

            # Save best model
            if val_loss is not None and val_loss < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_loss
                if save_best:
                    self.save_checkpoint("best.pt")

            # Logging
            if verbose:
                msg = f"Epoch {epoch + 1}/{num_epochs} | Train: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val: {val_loss:.6f}"
                msg += f" | LR: {self.metrics.learning_rate:.2e}"
                msg += f" | Time: {epoch_time:.1f}s"
                print(msg)

            # Early stopping
            if early_stopper is not None and val_loss is not None:
                if early_stopper(val_loss):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Periodic checkpoint
            if (
                self.training_config.save_every_n_epochs > 0
                and (epoch + 1) % self.training_config.save_every_n_epochs == 0
            ):
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")

        return self.metrics

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": {
                "train_loss": self.metrics.train_loss,
                "val_loss": self.metrics.val_loss,
                "best_val_loss": self.metrics.best_val_loss,
            },
            "training_config": self.training_config,
        }

        if self.model_config is not None:
            checkpoint["model_config"] = self.model_config

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path):
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        if "metrics" in checkpoint:
            self.metrics.train_loss = checkpoint["metrics"].get("train_loss", 0.0)
            self.metrics.val_loss = checkpoint["metrics"].get("val_loss")
            self.metrics.best_val_loss = checkpoint["metrics"].get(
                "best_val_loss", float("inf")
            )


def train_ttm(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor | None = None,
    context_length: int = 512,
    prediction_length: int = 96,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "auto",
    mode: Literal["pretrain", "finetune"] = "pretrain",
    **kwargs,
) -> TrainingMetrics:
    """Convenience function to train TTM model.

    Args:
        model: TTM model
        train_data: Training tensor (num_series, channels, seq_len)
        val_data: Validation tensor (optional)
        context_length: Context window length
        prediction_length: Prediction horizon
        batch_size: Batch size
        num_epochs: Training epochs
        learning_rate: Learning rate
        device: Device to use
        mode: Training mode ("pretrain" or "finetune")
        **kwargs: Additional trainer arguments

    Returns:
        Training metrics
    """
    from tinytimemixers.data.dataset import TimeSeriesDataset

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None
    if val_data is not None:
        val_dataset = TimeSeriesDataset(
            val_data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Configure for mode
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        freeze_backbone=(mode == "finetune"),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        device=device,
        **kwargs,
    )

    # Train
    return trainer.train()
