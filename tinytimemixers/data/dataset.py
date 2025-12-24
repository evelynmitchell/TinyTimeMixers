"""PyTorch Dataset classes for time series."""

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting.

    Stores complete time series and generates (context, target) pairs
    using a sliding window approach.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
    ):
        """Initialize TimeSeriesDataset.

        Args:
            data: Time series data of shape (num_series, channels, seq_len)
                  or (channels, seq_len) for single series
            context_length: Number of timesteps for input context
            prediction_length: Number of timesteps to predict
            stride: Step size between windows
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Ensure 3D: (num_series, channels, seq_len)
        if data.dim() == 2:
            data = data.unsqueeze(0)

        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.window_size = context_length + prediction_length

        # Compute number of windows per series
        self.num_series = data.shape[0]
        self.seq_len = data.shape[2]

        if self.seq_len < self.window_size:
            raise ValueError(
                f"Sequence length ({self.seq_len}) must be >= "
                f"window size ({self.window_size})"
            )

        self.windows_per_series = (self.seq_len - self.window_size) // stride + 1
        self.total_windows = self.num_series * self.windows_per_series

    def __len__(self) -> int:
        """Return total number of windows."""
        return self.total_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a (context, target) pair.

        Args:
            idx: Window index

        Returns:
            Tuple of (context, target) tensors
            - context: (channels, context_length)
            - target: (channels, prediction_length)
        """
        # Map index to series and window position
        series_idx = idx // self.windows_per_series
        window_idx = idx % self.windows_per_series
        start = window_idx * self.stride

        # Extract window
        series = self.data[series_idx]
        context = series[:, start : start + self.context_length]
        target = series[:, start + self.context_length : start + self.window_size]

        return context, target

    def get_series(self, idx: int) -> torch.Tensor:
        """Get a complete series by index.

        Args:
            idx: Series index

        Returns:
            Complete series tensor (channels, seq_len)
        """
        return self.data[idx]


class TimeSeriesWindowDataset(Dataset):
    """Pre-windowed dataset where each item is already a (context, target) pair.

    Useful when windows are pre-computed or loaded from disk.
    """

    def __init__(
        self,
        contexts: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
    ):
        """Initialize TimeSeriesWindowDataset.

        Args:
            contexts: Context windows (num_windows, channels, context_length)
            targets: Target windows (num_windows, channels, prediction_length)
        """
        if isinstance(contexts, np.ndarray):
            contexts = torch.from_numpy(contexts).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        if len(contexts) != len(targets):
            raise ValueError(
                f"contexts ({len(contexts)}) and targets ({len(targets)}) "
                "must have same length"
            )

        self.contexts = contexts
        self.targets = targets

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.contexts[idx], self.targets[idx]


class TimeSeriesIterableDataset(IterableDataset):
    """Iterable dataset for streaming large time series.

    Useful when data doesn't fit in memory.
    """

    def __init__(
        self,
        data_generator,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
    ):
        """Initialize TimeSeriesIterableDataset.

        Args:
            data_generator: Generator yielding (channels, seq_len) tensors
            context_length: Number of timesteps for input context
            prediction_length: Number of timesteps to predict
            stride: Step size between windows
        """
        self.data_generator = data_generator
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.window_size = context_length + prediction_length

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over windows from all series."""
        for series in self.data_generator():
            if isinstance(series, np.ndarray):
                series = torch.from_numpy(series).float()

            seq_len = series.shape[-1]
            if seq_len < self.window_size:
                continue

            # Generate windows for this series
            for start in range(0, seq_len - self.window_size + 1, self.stride):
                context = series[:, start : start + self.context_length]
                target = series[
                    :, start + self.context_length : start + self.window_size
                ]
                yield context, target


def create_train_val_split(
    dataset: TimeSeriesDataset,
    val_ratio: float = 0.2,
    temporal_split: bool = True,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
    """Split dataset into train and validation.

    Args:
        dataset: Original dataset
        val_ratio: Fraction for validation
        temporal_split: If True, uses later timesteps for validation.
            If False, uses random series.

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if temporal_split:
        # Use last portion of each series for validation
        split_point = int(dataset.seq_len * (1 - val_ratio))
        train_data = dataset.data[:, :, :split_point]
        val_data = dataset.data[:, :, split_point:]

        train_dataset = TimeSeriesDataset(
            train_data,
            dataset.context_length,
            dataset.prediction_length,
            dataset.stride,
        )
        val_dataset = TimeSeriesDataset(
            val_data,
            dataset.context_length,
            dataset.prediction_length,
            dataset.stride,
        )
    else:
        # Random series split
        num_val = int(dataset.num_series * val_ratio)
        indices = torch.randperm(dataset.num_series)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_dataset = TimeSeriesDataset(
            dataset.data[train_indices],
            dataset.context_length,
            dataset.prediction_length,
            dataset.stride,
        )
        val_dataset = TimeSeriesDataset(
            dataset.data[val_indices],
            dataset.context_length,
            dataset.prediction_length,
            dataset.stride,
        )

    return train_dataset, val_dataset
