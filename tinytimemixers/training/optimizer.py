"""Optimizer and scheduler utilities for TinyTimeMixers."""

from typing import Literal

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
)


def create_optimizer(
    model: nn.Module,
    optimizer_type: Literal["adamw", "adam", "sgd"] = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    freeze_backbone: bool = False,
    backbone_lr_scale: float = 0.1,
) -> Optimizer:
    """Create optimizer with optional parameter group configuration.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer
        learning_rate: Base learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Adam betas
        freeze_backbone: If True, freeze backbone parameters
        backbone_lr_scale: LR multiplier for backbone (for fine-tuning)

    Returns:
        Configured optimizer
    """
    # Separate backbone and other parameters
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if freeze_backbone and "backbone" in name:
            param.requires_grad = False
        elif "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups
    if backbone_params and backbone_lr_scale != 1.0:
        param_groups = [
            {"params": other_params, "lr": learning_rate},
            {"params": backbone_params, "lr": learning_rate * backbone_lr_scale},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": learning_rate}]

    # Create optimizer
    if optimizer_type == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: Literal[
        "cosine", "cosine_warmup", "plateau", "onecycle", "none"
    ] = "cosine_warmup",
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    steps_per_epoch: int | None = None,
) -> LRScheduler | None:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        steps_per_epoch: Steps per epoch (for step-based schedulers)

    Returns:
        Configured scheduler or None
    """
    if scheduler_type == "none":
        return None

    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
        )

    if scheduler_type == "cosine_warmup":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs // 4,
            T_mult=2,
            eta_min=min_lr,
        )

    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=min_lr,
        )

    if scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=num_epochs * steps_per_epoch,
            pct_start=warmup_epochs / num_epochs,
            anneal_strategy="cos",
            final_div_factor=optimizer.param_groups[0]["lr"] / min_lr,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_type}")


class WarmupScheduler:
    """Linear warmup scheduler wrapper.

    Wraps another scheduler and applies linear warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: LRScheduler | None = None,
    ):
        """Initialize WarmupScheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            base_scheduler: Scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, metrics: float | None = None):
        """Take a scheduler step.

        Args:
            metrics: Validation metric (for ReduceLROnPlateau)
        """
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs, strict=True
            ):
                param_group["lr"] = base_lr * warmup_factor
        elif self.base_scheduler is not None:
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Get current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Get scheduler state."""
        state = {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }
        if self.base_scheduler is not None:
            state["base_scheduler"] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]
        if self.base_scheduler is not None and "base_scheduler" in state_dict:
            self.base_scheduler.load_state_dict(state_dict["base_scheduler"])


def get_parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: Model to count
        trainable_only: Only count trainable parameters

    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups_summary(model: nn.Module) -> dict[str, int]:
    """Get parameter count by module group.

    Args:
        model: Model to analyze

    Returns:
        Dictionary of module name to parameter count
    """
    summary = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            count = sum(p.numel() for p in module.parameters())
            if count > 0:
                # Group by top-level module
                top_level = name.split(".")[0] if "." in name else name
                summary[top_level] = summary.get(top_level, 0) + count
    return summary
