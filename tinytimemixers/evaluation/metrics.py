"""Evaluation metrics for time series forecasting."""

from typing import Any

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def MSE(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Mean Squared Error.

    Args:
        pred: Predictions
        target: Ground truth
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        MSE value(s)
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    errors = (pred - target) ** 2

    if reduction == "mean":
        return float(np.mean(errors))
    elif reduction == "none":
        return np.mean(errors, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def MAE(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Mean Absolute Error.

    Args:
        pred: Predictions
        target: Ground truth
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        MAE value(s)
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    errors = np.abs(pred - target)

    if reduction == "mean":
        return float(np.mean(errors))
    elif reduction == "none":
        return np.mean(errors, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def RMSE(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Root Mean Squared Error.

    Args:
        pred: Predictions
        target: Ground truth
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        RMSE value(s)
    """
    mse = MSE(pred, target, reduction=reduction)
    if isinstance(mse, np.ndarray):
        return np.sqrt(mse)
    return float(np.sqrt(mse))


def MAPE(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Mean Absolute Percentage Error.

    Args:
        pred: Predictions
        target: Ground truth
        eps: Small constant to avoid division by zero
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        MAPE value(s) as percentage
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    # Avoid division by zero
    denominator = np.abs(target) + eps
    errors = np.abs(pred - target) / denominator * 100

    if reduction == "mean":
        return float(np.mean(errors))
    elif reduction == "none":
        return np.mean(errors, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def SMAPE(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Symmetric Mean Absolute Percentage Error.

    More robust than MAPE for values near zero.

    Args:
        pred: Predictions
        target: Ground truth
        eps: Small constant to avoid division by zero
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        SMAPE value(s) as percentage
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    numerator = np.abs(pred - target)
    denominator = (np.abs(pred) + np.abs(target)) / 2 + eps
    errors = numerator / denominator * 100

    if reduction == "mean":
        return float(np.mean(errors))
    elif reduction == "none":
        return np.mean(errors, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def MASE(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    context: torch.Tensor | np.ndarray | None = None,
    seasonality: int = 1,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Mean Absolute Scaled Error.

    Scale-independent metric comparing against naive seasonal forecast.
    MASE < 1 means better than naive forecast.

    Args:
        pred: Predictions
        target: Ground truth
        context: Historical context for computing scale
        seasonality: Seasonal period (1 for naive random walk)
        eps: Small constant for numerical stability
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        MASE value(s)
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    # Compute MAE
    mae = np.abs(pred - target)

    # Compute naive forecast error from context
    if context is not None:
        context = _to_numpy(context)
        # Naive forecast: y_t = y_{t-seasonality}
        naive_errors = np.abs(context[..., seasonality:] - context[..., :-seasonality])
        scale = np.mean(naive_errors, axis=-1, keepdims=True) + eps
    else:
        # Without context, use target's own variability
        naive_errors = np.abs(target[..., seasonality:] - target[..., :-seasonality])
        scale = np.mean(naive_errors, axis=-1, keepdims=True) + eps

    mase = mae / scale

    if reduction == "mean":
        return float(np.mean(mase))
    elif reduction == "none":
        return np.mean(mase, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def CRPS(
    pred_samples: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Continuous Ranked Probability Score.

    Proper scoring rule for probabilistic forecasts.
    Lower is better.

    Args:
        pred_samples: Predicted samples (num_samples, ..., prediction_length)
        target: Ground truth (..., prediction_length)
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        CRPS value(s)
    """
    pred_samples = _to_numpy(pred_samples)
    target = _to_numpy(target)

    # Ensure pred_samples has samples dimension
    if pred_samples.ndim == target.ndim:
        # Single prediction, treat as deterministic
        pred_samples = pred_samples[np.newaxis, ...]

    num_samples = pred_samples.shape[0]

    # Sort samples
    sorted_samples = np.sort(pred_samples, axis=0)

    # Compute CRPS using quantile formulation
    # CRPS = integral over q of (F(q) - indicator(q >= target))^2 dq
    # Approximated using sorted samples

    crps_values = np.zeros(target.shape)
    for i in range(num_samples):
        indicator = (sorted_samples[i] >= target).astype(float)
        quantile = (i + 0.5) / num_samples
        crps_values += (quantile - indicator) ** 2

    crps_values = crps_values / num_samples

    if reduction == "mean":
        return float(np.mean(crps_values))
    elif reduction == "none":
        return np.mean(crps_values, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def coverage(
    pred_lower: torch.Tensor | np.ndarray,
    pred_upper: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Prediction interval coverage.

    Measures fraction of targets within prediction interval.

    Args:
        pred_lower: Lower bound predictions
        pred_upper: Upper bound predictions
        target: Ground truth
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        Coverage value(s) as fraction
    """
    pred_lower = _to_numpy(pred_lower)
    pred_upper = _to_numpy(pred_upper)
    target = _to_numpy(target)

    covered = (target >= pred_lower) & (target <= pred_upper)

    if reduction == "mean":
        return float(np.mean(covered))
    elif reduction == "none":
        return np.mean(covered, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def interval_width(
    pred_lower: torch.Tensor | np.ndarray,
    pred_upper: torch.Tensor | np.ndarray,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Mean prediction interval width.

    Measures sharpness of probabilistic predictions.

    Args:
        pred_lower: Lower bound predictions
        pred_upper: Upper bound predictions
        reduction: "mean" for scalar, "none" for per-sample

    Returns:
        Width value(s)
    """
    pred_lower = _to_numpy(pred_lower)
    pred_upper = _to_numpy(pred_upper)

    width = pred_upper - pred_lower

    if reduction == "mean":
        return float(np.mean(width))
    elif reduction == "none":
        return np.mean(width, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_all_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    context: torch.Tensor | np.ndarray | None = None,
    seasonality: int = 1,
    include_crps: bool = False,
    pred_samples: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all standard metrics.

    Args:
        pred: Point predictions
        target: Ground truth
        context: Historical context (for MASE)
        seasonality: Seasonal period (for MASE)
        include_crps: Whether to compute CRPS
        pred_samples: Predicted samples (for CRPS)

    Returns:
        Dictionary of metric name to value
    """
    metrics: dict[str, Any] = {
        "MSE": MSE(pred, target),
        "MAE": MAE(pred, target),
        "RMSE": RMSE(pred, target),
        "MAPE": MAPE(pred, target),
        "SMAPE": SMAPE(pred, target),
        "MASE": MASE(pred, target, context=context, seasonality=seasonality),
    }

    if include_crps and pred_samples is not None:
        metrics["CRPS"] = CRPS(pred_samples, target)

    return metrics


class MetricTracker:
    """Track metrics during training/evaluation."""

    def __init__(self, metric_names: list[str] | None = None):
        """Initialize MetricTracker.

        Args:
            metric_names: List of metrics to track (default: MSE, MAE)
        """
        if metric_names is None:
            metric_names = ["MSE", "MAE"]
        self.metric_names = metric_names
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.values: dict[str, list[float]] = {name: [] for name in self.metric_names}
        self.count = 0

    def update(
        self,
        pred: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
        **kwargs,
    ):
        """Update metrics with new predictions.

        Args:
            pred: Predictions
            target: Ground truth
            **kwargs: Additional arguments for metrics
        """
        all_metrics = compute_all_metrics(pred, target, **kwargs)

        for name in self.metric_names:
            if name in all_metrics:
                self.values[name].append(all_metrics[name])

        self.count += 1

    def compute(self) -> dict[str, float]:
        """Compute average metrics.

        Returns:
            Dictionary of metric name to average value
        """
        return {
            name: float(np.mean(values)) if values else 0.0
            for name, values in self.values.items()
        }

    def __repr__(self) -> str:
        metrics = self.compute()
        return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
