"""Preprocessing utilities for time series."""

from typing import Any

import numpy as np
import torch


class StandardScaler:
    """Standard scaler for time series.

    Computes mean and std over training data and applies to all data.
    """

    def __init__(self, dim: int = -1):
        """Initialize StandardScaler.

        Args:
            dim: Dimension to compute statistics over (default: last)
        """
        self.dim = dim
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None
        self.eps = 1e-8

    def fit(self, data: torch.Tensor | np.ndarray) -> "StandardScaler":
        """Compute mean and std from data.

        Args:
            data: Input data

        Returns:
            self
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.mean_ = data.mean(dim=self.dim, keepdim=True)
        self.std_ = data.std(dim=self.dim, keepdim=True) + self.eps
        return self

    def transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply scaling.

        Args:
            data: Input data

        Returns:
            Scaled data
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        return (data - self.mean_) / self.std_

    def inverse_transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Reverse scaling.

        Args:
            data: Scaled data

        Returns:
            Original scale data
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        return data * self.std_ + self.mean_

    def fit_transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Fit and transform in one step.

        Args:
            data: Input data

        Returns:
            Scaled data
        """
        return self.fit(data).transform(data)


class InstanceScaler:
    """Per-instance scaler (like RevIN but as a preprocessing step).

    Normalizes each instance independently.
    """

    def __init__(self, dim: int = -1):
        """Initialize InstanceScaler.

        Args:
            dim: Dimension to compute statistics over
        """
        self.dim = dim
        self.eps = 1e-8

    def transform(
        self, data: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply instance-wise scaling.

        Args:
            data: Input data

        Returns:
            Tuple of (scaled_data, stats) where stats contains mean and std
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        mean = data.mean(dim=self.dim, keepdim=True)
        std = data.std(dim=self.dim, keepdim=True) + self.eps

        scaled = (data - mean) / std
        stats = {"mean": mean, "std": std}

        return scaled, stats

    def inverse_transform(
        self,
        data: torch.Tensor | np.ndarray,
        stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reverse instance-wise scaling.

        Args:
            data: Scaled data
            stats: Statistics from transform

        Returns:
            Original scale data
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        return data * stats["std"] + stats["mean"]


class Preprocessor:
    """Combined preprocessor for time series.

    Handles missing values, scaling, and other preprocessing steps.
    """

    def __init__(
        self,
        scaling: str = "standard",
        fill_missing: str = "ffill",
    ):
        """Initialize Preprocessor.

        Args:
            scaling: Scaling method ("standard", "instance", or "none")
            fill_missing: Missing value handling ("ffill", "zero", "mean")
        """
        self.scaling = scaling
        self.fill_missing = fill_missing

        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "instance":
            self.scaler = InstanceScaler()
        else:
            self.scaler = None

        self._is_fitted = False

    def fit(self, data: torch.Tensor | np.ndarray) -> "Preprocessor":
        """Fit preprocessor on training data.

        Args:
            data: Training data

        Returns:
            self
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Handle missing values first
        data = self._handle_missing(data)

        # Fit scaler if using standard scaling
        if self.scaling == "standard" and self.scaler is not None:
            self.scaler.fit(data)

        self._is_fitted = True
        return self

    def transform(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Apply preprocessing.

        Args:
            data: Input data

        Returns:
            Preprocessed data (and stats if using instance scaling)
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Handle missing values
        data = self._handle_missing(data)

        # Apply scaling
        if self.scaling == "standard":
            return self.scaler.transform(data)
        elif self.scaling == "instance":
            return self.scaler.transform(data)
        else:
            return data

    def inverse_transform(
        self,
        data: torch.Tensor | np.ndarray,
        stats: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Reverse preprocessing.

        Args:
            data: Preprocessed data
            stats: Statistics for instance scaling

        Returns:
            Original scale data
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        if self.scaling == "standard":
            return self.scaler.inverse_transform(data)
        elif self.scaling == "instance":
            if stats is None:
                raise ValueError("stats required for instance scaling")
            return self.scaler.inverse_transform(data, stats)
        else:
            return data

    def _handle_missing(self, data: torch.Tensor) -> torch.Tensor:
        """Handle missing values.

        Args:
            data: Input data with potential NaNs

        Returns:
            Data with NaNs filled
        """
        if not torch.isnan(data).any():
            return data

        if self.fill_missing == "zero":
            return torch.nan_to_num(data, nan=0.0)
        elif self.fill_missing == "mean":
            mean = torch.nanmean(data)
            return torch.nan_to_num(data, nan=mean.item())
        elif self.fill_missing == "ffill":
            # Forward fill (more complex, use simple approach)
            # For each NaN, use the previous non-NaN value
            result = data.clone()
            mask = torch.isnan(result)
            if mask.any():
                # Simple approach: replace with mean first
                mean = torch.nanmean(data)
                result = torch.nan_to_num(result, nan=mean.item())
            return result
        else:
            return data


def pad_sequence(
    data: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    pad_side: str = "left",
) -> torch.Tensor:
    """Pad sequence to target length.

    Args:
        data: Input tensor (..., seq_len)
        target_length: Desired sequence length
        pad_value: Value for padding
        pad_side: "left" or "right"

    Returns:
        Padded tensor
    """
    current_len = data.shape[-1]
    if current_len >= target_length:
        return data

    pad_len = target_length - current_len
    pad_shape = list(data.shape[:-1]) + [pad_len]
    padding = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)

    if pad_side == "left":
        return torch.cat([padding, data], dim=-1)
    else:
        return torch.cat([data, padding], dim=-1)


def truncate_sequence(
    data: torch.Tensor,
    target_length: int,
    truncate_side: str = "left",
) -> torch.Tensor:
    """Truncate sequence to target length.

    Args:
        data: Input tensor (..., seq_len)
        target_length: Desired sequence length
        truncate_side: "left" or "right"

    Returns:
        Truncated tensor
    """
    current_len = data.shape[-1]
    if current_len <= target_length:
        return data

    if truncate_side == "left":
        return data[..., -target_length:]
    else:
        return data[..., :target_length]
