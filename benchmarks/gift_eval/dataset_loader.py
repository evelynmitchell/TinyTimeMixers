"""Dataset loader for GIFT-Eval benchmark.

This module provides utilities for downloading and loading
GIFT-Eval datasets from HuggingFace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from benchmarks.gift_eval.config import DatasetConfig

try:
    from gluonts.dataset.common import ListDataset

    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False
    ListDataset = None

try:
    from datasets import load_dataset as hf_load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    hf_load_dataset = None


logger = logging.getLogger(__name__)


def _check_dependencies():
    """Check if required dependencies are available."""
    if not GLUONTS_AVAILABLE:
        raise ImportError(
            "GluonTS is required for dataset loading. "
            "Install with: pip install gluonts"
        )
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets is required for GIFT-Eval loading. "
            "Install with: pip install datasets"
        )


class GIFTEvalDatasetLoader:
    """Loader for GIFT-Eval datasets from HuggingFace.

    Downloads and caches datasets locally, converts to GluonTS format.
    """

    HF_REPO = "Salesforce/GiftEval"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        download: bool = True,
    ):
        """Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets.
                       Defaults to ~/.cache/gift_eval
            download: Whether to download datasets if not cached
        """
        _check_dependencies()

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "gift_eval"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download = download

        self._metadata_cache: dict = {}

    def load_dataset(
        self,
        config: DatasetConfig,
    ) -> tuple[ListDataset, ListDataset]:
        """Load train and test datasets for a configuration.

        Args:
            config: Dataset configuration

        Returns:
            Tuple of (train_dataset, test_dataset) in GluonTS format
        """
        dataset_name = config.name
        freq = config.freq

        # Try to load from HuggingFace
        try:
            hf_dataset = hf_load_dataset(
                self.HF_REPO,
                name=dataset_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} from HuggingFace: {e}")
            # Fall back to synthetic data for testing
            return self._create_synthetic_dataset(config)

        # Convert to GluonTS format
        train_entries = []
        test_entries = []

        # Process training data
        if "train" in hf_dataset:
            for item in hf_dataset["train"]:
                entry = self._convert_entry(item, freq)
                if entry is not None:
                    train_entries.append(entry)

        # Process test data
        if "test" in hf_dataset:
            for item in hf_dataset["test"]:
                entry = self._convert_entry(item, freq)
                if entry is not None:
                    test_entries.append(entry)

        train_dataset = ListDataset(train_entries, freq=freq)
        test_dataset = ListDataset(test_entries, freq=freq)

        return train_dataset, test_dataset

    def _convert_entry(
        self,
        item: dict,
        freq: str,
    ) -> dict | None:
        """Convert HuggingFace entry to GluonTS format.

        Args:
            item: HuggingFace dataset entry
            freq: Frequency string

        Returns:
            GluonTS-compatible entry dict
        """
        try:
            import pandas as pd

            # Extract target values
            if "target" in item:
                target = np.asarray(item["target"])
            elif "values" in item:
                target = np.asarray(item["values"])
            else:
                return None

            # Extract start date
            if "start" in item:
                start = pd.Period(item["start"], freq=freq)
            elif "timestamp" in item:
                start = pd.Period(item["timestamp"][0], freq=freq)
            else:
                start = pd.Period("2020-01-01", freq=freq)

            # Build entry
            entry = {
                "target": target,
                "start": start,
            }

            # Optional fields
            if "item_id" in item:
                entry["item_id"] = item["item_id"]
            if "feat_static_cat" in item:
                entry["feat_static_cat"] = item["feat_static_cat"]
            if "feat_dynamic_real" in item:
                entry["feat_dynamic_real"] = item["feat_dynamic_real"]

            return entry

        except Exception as e:
            logger.warning(f"Failed to convert entry: {e}")
            return None

    def _create_synthetic_dataset(
        self,
        config: DatasetConfig,
    ) -> tuple[ListDataset, ListDataset]:
        """Create synthetic dataset for testing.

        Args:
            config: Dataset configuration

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import pandas as pd

        num_series = 10
        train_length = config.context_length * 2
        test_length = config.context_length + config.prediction_length

        train_entries = []
        test_entries = []

        for i in range(num_series):
            # Generate synthetic time series
            t = np.arange(train_length)
            # Add trend, seasonality, and noise
            trend = 0.01 * t
            seasonality = np.sin(2 * np.pi * t / config.seasonality)
            noise = np.random.randn(train_length) * 0.1
            train_target = trend + seasonality + noise

            train_entries.append(
                {
                    "target": train_target.astype(np.float32),
                    "start": pd.Period("2020-01-01", freq=config.freq),
                    "item_id": f"series_{i}",
                }
            )

            # Test series (same pattern)
            t = np.arange(test_length)
            trend = 0.01 * t
            seasonality = np.sin(2 * np.pi * t / config.seasonality)
            noise = np.random.randn(test_length) * 0.1
            test_target = trend + seasonality + noise

            test_entries.append(
                {
                    "target": test_target.astype(np.float32),
                    "start": pd.Period("2020-01-01", freq=config.freq),
                    "item_id": f"series_{i}",
                }
            )

        return (
            ListDataset(train_entries, freq=config.freq),
            ListDataset(test_entries, freq=config.freq),
        )

    def download_all(
        self,
        configs: list[DatasetConfig] | None = None,
    ):
        """Pre-download all datasets.

        Args:
            configs: List of configs to download. If None, downloads all.
        """
        from benchmarks.gift_eval.config import GIFT_EVAL_DATASETS

        if configs is None:
            configs = GIFT_EVAL_DATASETS

        # Get unique dataset names
        dataset_names = set(c.name for c in configs)

        for name in dataset_names:
            logger.info(f"Downloading {name}...")
            try:
                hf_load_dataset(
                    self.HF_REPO,
                    name=name,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Failed to download {name}: {e}")

    def get_metadata(
        self,
        config: DatasetConfig,
    ) -> dict:
        """Get metadata for a dataset configuration.

        Args:
            config: Dataset configuration

        Returns:
            Metadata dictionary
        """
        cache_key = config.config_name
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        metadata = {
            "name": config.name,
            "domain": config.domain,
            "freq": config.freq,
            "prediction_length": config.prediction_length,
            "context_length": config.context_length,
            "num_variates": config.num_variates,
            "seasonality": config.seasonality,
            "term": config.term,
        }

        self._metadata_cache[cache_key] = metadata
        return metadata

    def clear_cache(self):
        """Clear the local cache directory."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


def create_gluonts_dataset_from_numpy(
    data: np.ndarray,
    freq: str,
    start: str = "2020-01-01",
) -> ListDataset:
    """Create GluonTS ListDataset from numpy array.

    Args:
        data: Time series data (num_series, seq_len) or (num_series, num_variates, seq_len)
        freq: Frequency string
        start: Start date string

    Returns:
        GluonTS ListDataset
    """
    _check_dependencies()
    import pandas as pd

    if data.ndim == 1:
        data = data.reshape(1, -1)

    entries = []
    for i in range(len(data)):
        target = data[i]
        entries.append(
            {
                "target": target.astype(np.float32),
                "start": pd.Period(start, freq=freq),
                "item_id": f"series_{i}",
            }
        )

    return ListDataset(entries, freq=freq)


def split_dataset_for_evaluation(
    dataset: ListDataset,
    prediction_length: int,
) -> tuple[ListDataset, ListDataset]:
    """Split dataset into context and targets for evaluation.

    Args:
        dataset: Full time series dataset
        prediction_length: Prediction horizon

    Returns:
        Tuple of (context_dataset, target arrays)
    """
    _check_dependencies()

    context_entries = []
    targets = []

    for entry in dataset:
        target = np.asarray(entry["target"])

        # Extract context and target
        if target.ndim == 1:
            context_target = target[:-prediction_length]
            eval_target = target[-prediction_length:]
        else:
            context_target = target[:, :-prediction_length]
            eval_target = target[:, -prediction_length:]

        context_entries.append(
            {
                "target": context_target,
                "start": entry["start"],
                "item_id": entry.get("item_id"),
            }
        )
        targets.append(eval_target)

    freq = dataset.list_data[0]["start"].freq.name if dataset.list_data else "D"
    context_dataset = ListDataset(context_entries, freq=freq)

    return context_dataset, targets
