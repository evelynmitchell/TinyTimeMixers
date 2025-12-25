"""Loss functions for time series forecasting."""

from typing import Literal

import torch
import torch.nn as nn


class ForecastLoss(nn.Module):
    """Base class for forecast losses."""

    def __init__(self, reduction: str = "mean"):
        """Initialize ForecastLoss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            pred: Predictions (batch, channels, prediction_length)
            target: Targets (batch, channels, prediction_length)
            mask: Optional mask for valid timesteps

        Returns:
            Loss value
        """
        raise NotImplementedError

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MSELoss(ForecastLoss):
    """Mean Squared Error loss for forecasting."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            pred: Predictions
            target: Targets
            mask: Optional mask

        Returns:
            MSE loss
        """
        loss = (pred - target) ** 2

        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp(min=1)

        return self._apply_reduction(loss)


class MAELoss(ForecastLoss):
    """Mean Absolute Error loss for forecasting."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MAE loss.

        Args:
            pred: Predictions
            target: Targets
            mask: Optional mask

        Returns:
            MAE loss
        """
        loss = torch.abs(pred - target)

        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp(min=1)

        return self._apply_reduction(loss)


class HuberLoss(ForecastLoss):
    """Huber loss (smooth L1) for forecasting.

    Less sensitive to outliers than MSE.
    """

    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        """Initialize HuberLoss.

        Args:
            reduction: Reduction method
            delta: Threshold for switching between L1 and L2
        """
        super().__init__(reduction)
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Huber loss."""
        abs_diff = torch.abs(pred - target)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear

        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp(min=1)

        return self._apply_reduction(loss)


class QuantileLoss(ForecastLoss):
    """Quantile loss for probabilistic forecasting.

    Useful for predicting specific quantiles of the distribution.
    """

    def __init__(self, reduction: str = "mean", quantile: float = 0.5):
        """Initialize QuantileLoss.

        Args:
            reduction: Reduction method
            quantile: Target quantile (0.5 = median)
        """
        super().__init__(reduction)
        self.quantile = quantile

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute quantile loss."""
        diff = target - pred
        loss = torch.where(
            diff >= 0,
            self.quantile * diff,
            (self.quantile - 1) * diff,
        )

        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp(min=1)

        return self._apply_reduction(loss)


class MASELoss(ForecastLoss):
    """Mean Absolute Scaled Error.

    Scales MAE by the naive forecast error (seasonal random walk).
    Good for comparing across different time series.
    """

    def __init__(
        self,
        reduction: str = "mean",
        seasonality: int = 1,
        eps: float = 1e-8,
    ):
        """Initialize MASELoss.

        Args:
            reduction: Reduction method
            seasonality: Seasonal period for naive forecast
            eps: Small constant for numerical stability
        """
        super().__init__(reduction)
        self.seasonality = seasonality
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MASE loss.

        Args:
            pred: Predictions
            target: Targets
            mask: Optional mask
            context: Historical context for computing scale

        Returns:
            MASE loss
        """
        # Compute MAE
        mae = torch.abs(pred - target)

        # Compute scale from context if provided
        if context is not None:
            # Naive forecast error: |y_t - y_{t-seasonality}|
            naive_errors = torch.abs(
                context[..., self.seasonality:] - context[..., : -self.seasonality]
            )
            scale = naive_errors.mean(dim=-1, keepdim=True) + self.eps
            mae = mae / scale

        if mask is not None:
            mae = mae * mask
            if self.reduction == "mean":
                return mae.sum() / mask.sum().clamp(min=1)

        return self._apply_reduction(mae)


def get_loss_fn(
    loss_type: Literal["mse", "mae", "huber", "quantile"] = "mse",
    **kwargs,
) -> ForecastLoss:
    """Get loss function by name.

    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance
    """
    loss_map = {
        "mse": MSELoss,
        "mae": MAELoss,
        "huber": HuberLoss,
        "quantile": QuantileLoss,
    }

    if loss_type not in loss_map:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from {list(loss_map.keys())}"
        )

    return loss_map[loss_type](**kwargs)
