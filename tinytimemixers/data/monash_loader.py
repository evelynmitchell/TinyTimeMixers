"""Monash Time Series Repository data loader.

The Monash repository contains diverse time series datasets for
pre-training and benchmarking. Available via HuggingFace datasets.

Reference: https://huggingface.co/datasets/monash_tsf
"""

from typing import Any

import numpy as np
import torch

# Dataset metadata from Monash repository
MONASH_DATASETS = {
    # Hourly datasets
    "electricity_hourly": {"frequency": "H", "domain": "Energy"},
    "traffic_hourly": {"frequency": "H", "domain": "Transport"},
    "solar_weekly": {"frequency": "W", "domain": "Energy"},
    "hospital": {"frequency": "M", "domain": "Healthcare"},
    "covid_deaths": {"frequency": "D", "domain": "Healthcare"},
    "weather": {"frequency": "D", "domain": "Nature"},
    "tourism_monthly": {"frequency": "M", "domain": "Economics"},
    "tourism_quarterly": {"frequency": "Q", "domain": "Economics"},
    "tourism_yearly": {"frequency": "Y", "domain": "Economics"},
    "m1_yearly": {"frequency": "Y", "domain": "Economics"},
    "m1_quarterly": {"frequency": "Q", "domain": "Economics"},
    "m1_monthly": {"frequency": "M", "domain": "Economics"},
    "m3_yearly": {"frequency": "Y", "domain": "Economics"},
    "m3_quarterly": {"frequency": "Q", "domain": "Economics"},
    "m3_monthly": {"frequency": "M", "domain": "Economics"},
    "m3_other": {"frequency": "Other", "domain": "Economics"},
    "m4_yearly": {"frequency": "Y", "domain": "Economics"},
    "m4_quarterly": {"frequency": "Q", "domain": "Economics"},
    "m4_monthly": {"frequency": "M", "domain": "Economics"},
    "m4_weekly": {"frequency": "W", "domain": "Economics"},
    "m4_daily": {"frequency": "D", "domain": "Economics"},
    "m4_hourly": {"frequency": "H", "domain": "Economics"},
    "car_parts": {"frequency": "M", "domain": "Sales"},
    "fred_md": {"frequency": "M", "domain": "Economics"},
    "nn5_weekly": {"frequency": "W", "domain": "Finance"},
    "nn5_daily": {"frequency": "D", "domain": "Finance"},
    "kaggle_web_traffic_daily": {"frequency": "D", "domain": "Web"},
    "kaggle_web_traffic_weekly": {"frequency": "W", "domain": "Web"},
    "australian_electricity_demand": {"frequency": "30T", "domain": "Energy"},
}

# Frequencies to exclude from pre-training (too low frequency)
EXCLUDED_FREQUENCIES = {"Y", "Q", "M", "W", "Other"}


def get_pretraining_datasets() -> list[str]:
    """Get list of datasets suitable for pre-training.

    Excludes yearly, quarterly, monthly, and weekly datasets
    as they have too few samples per series.

    Returns:
        List of dataset names
    """
    return [
        name
        for name, info in MONASH_DATASETS.items()
        if info["frequency"] not in EXCLUDED_FREQUENCIES
    ]


class MonashLoader:
    """Loader for Monash Time Series Repository datasets.

    Uses HuggingFace datasets library for downloading and caching.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
    ):
        """Initialize MonashLoader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir
        self._datasets_available = False

        # Check if datasets library is available
        try:
            import datasets

            self._datasets_available = True
            self._datasets = datasets
        except ImportError:
            pass

    def load(
        self,
        dataset_name: str,
        split: str = "train",
    ) -> dict[str, Any]:
        """Load a Monash dataset.

        Args:
            dataset_name: Name of dataset (e.g., "electricity_hourly")
            split: Data split ("train", "test", or "validation")

        Returns:
            Dictionary with:
                - data: List of time series (each is a numpy array)
                - frequency: Sampling frequency
                - domain: Application domain
        """
        if not self._datasets_available:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        if dataset_name not in MONASH_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(MONASH_DATASETS.keys())}"
            )

        # Load from HuggingFace
        dataset = self._datasets.load_dataset(
            "monash_tsf",
            dataset_name,
            split=split,
            cache_dir=self.cache_dir,
        )

        # Extract time series
        series_list = []
        for item in dataset:
            # Monash datasets have 'target' field with the time series
            if "target" in item:
                series = np.array(item["target"], dtype=np.float32)
                series_list.append(series)

        metadata = MONASH_DATASETS[dataset_name]
        return {
            "data": series_list,
            "frequency": metadata["frequency"],
            "domain": metadata["domain"],
            "name": dataset_name,
        }

    def load_as_tensor(
        self,
        dataset_name: str,
        split: str = "train",
        min_length: int | None = None,
        pad_to_length: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Load dataset as a single tensor.

        Args:
            dataset_name: Name of dataset
            split: Data split
            min_length: Minimum series length (shorter series excluded)
            pad_to_length: Pad all series to this length

        Returns:
            Tuple of (tensor, metadata)
            - tensor: (num_series, 1, seq_len) if univariate
            - metadata: Dataset information
        """
        result = self.load(dataset_name, split)
        series_list = result["data"]

        # Filter by length
        if min_length is not None:
            series_list = [s for s in series_list if len(s) >= min_length]

        if len(series_list) == 0:
            raise ValueError(f"No series with length >= {min_length}")

        # Determine target length
        if pad_to_length is not None:
            target_len = pad_to_length
        else:
            # Use maximum length
            target_len = max(len(s) for s in series_list)

        # Pad/truncate to same length
        processed = []
        for series in series_list:
            if len(series) < target_len:
                # Pad with zeros
                padded = np.zeros(target_len, dtype=np.float32)
                padded[: len(series)] = series
                series = padded
            elif len(series) > target_len:
                # Truncate from end
                series = series[:target_len]
            processed.append(series)

        # Stack into tensor: (num_series, 1, seq_len)
        data = np.stack(processed, axis=0)[:, np.newaxis, :]
        tensor = torch.from_numpy(data)

        return tensor, {
            "frequency": result["frequency"],
            "domain": result["domain"],
            "name": dataset_name,
            "num_series": len(processed),
            "seq_len": target_len,
        }

    def load_multiple(
        self,
        dataset_names: list[str] | None = None,
        min_length: int = 512,
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        """Load multiple datasets for pre-training.

        Args:
            dataset_names: List of datasets to load (default: all suitable)
            min_length: Minimum series length

        Returns:
            Tuple of (list of tensors, list of metadata)
        """
        if dataset_names is None:
            dataset_names = get_pretraining_datasets()

        tensors = []
        metadata_list = []

        for name in dataset_names:
            try:
                tensor, metadata = self.load_as_tensor(name, min_length=min_length)
                tensors.append(tensor)
                metadata_list.append(metadata)
            except (ValueError, Exception) as e:
                print(f"Warning: Could not load {name}: {e}")
                continue

        return tensors, metadata_list


def create_synthetic_dataset(
    num_series: int = 100,
    seq_len: int = 1024,
    num_channels: int = 1,
    trend: bool = True,
    seasonality: bool = True,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Create synthetic time series dataset.

    Useful for testing when Monash data is not available.

    Args:
        num_series: Number of series to generate
        seq_len: Length of each series
        num_channels: Number of channels per series
        trend: Whether to include linear trend
        seasonality: Whether to include seasonal patterns
        noise_std: Standard deviation of noise

    Returns:
        Tensor of shape (num_series, num_channels, seq_len)
    """
    t = np.linspace(0, 8 * np.pi, seq_len)
    data = np.zeros((num_series, num_channels, seq_len), dtype=np.float32)

    for i in range(num_series):
        for c in range(num_channels):
            series = np.zeros(seq_len)

            if seasonality:
                # Multiple seasonal components
                amp1 = np.random.uniform(0.5, 2.0)
                amp2 = np.random.uniform(0.2, 1.0)
                phase1 = np.random.uniform(0, 2 * np.pi)
                phase2 = np.random.uniform(0, 2 * np.pi)
                series += amp1 * np.sin(t + phase1)
                series += amp2 * np.sin(2 * t + phase2)

            if trend:
                slope = np.random.uniform(-0.1, 0.1)
                series += slope * t

            # Add noise
            series += np.random.randn(seq_len) * noise_std

            data[i, c] = series

    return torch.from_numpy(data)
