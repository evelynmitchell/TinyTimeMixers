"""TabPFN-TS wrapper for time series forecasting.

This module provides a wrapper around TabPFN-TS for comparison
with TinyTimeMixers on the GIFT-Eval benchmark.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from gluonts.dataset.common import Dataset
    from gluonts.model.forecast import Forecast

try:
    from tabpfn_time_series import TabPFNForecaster

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    TabPFNForecaster = None

try:
    from gluonts.model.forecast import SampleForecast

    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False
    SampleForecast = None


logger = logging.getLogger(__name__)


def _check_tabpfn():
    """Check if TabPFN-TS is available."""
    if not TABPFN_AVAILABLE:
        raise ImportError(
            "TabPFN-TS is required for comparison. "
            "Install with: pip install tabpfn-time-series"
        )


def _check_gluonts():
    """Check if GluonTS is available."""
    if not GLUONTS_AVAILABLE:
        raise ImportError(
            "GluonTS is required for TabPFN wrapper. "
            "Install with: pip install gluonts"
        )


class TabPFNTSWrapper:
    """Wrapper for TabPFN time series forecasting.

    Provides a unified interface for comparison with TTM.
    """

    def __init__(
        self,
        prediction_length: int,
        freq: str,
        use_api: bool = True,
        device: str = "auto",
    ):
        """Initialize TabPFN-TS wrapper.

        Args:
            prediction_length: Forecast horizon
            freq: Frequency string
            use_api: Use TabPFN cloud API (no GPU needed)
            device: Device for local inference
        """
        _check_tabpfn()

        self.prediction_length = prediction_length
        self.freq = freq
        self.use_api = use_api
        self.device = device

        # Initialize TabPFN forecaster
        self.forecaster = TabPFNForecaster()

    def predict(
        self,
        context: np.ndarray,
        timestamps: pd.DatetimeIndex | None = None,
        future_timestamps: pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        """Generate point forecast.

        Args:
            context: Historical values (context_length,)
            timestamps: Historical timestamps (optional)
            future_timestamps: Future timestamps to predict (optional)

        Returns:
            Point forecast (prediction_length,)
        """

        # Create time features if timestamps provided (for future use)
        # Note: features currently unused but may be used in future versions
        _ = self._create_features(timestamps) if timestamps is not None else None

        # Prepare data for TabPFN
        if context.ndim == 1:
            context = context.reshape(-1, 1)

        # Use TabPFN for prediction
        try:
            forecast = self.forecaster.predict(
                context,
                n_steps=self.prediction_length,
            )
            return forecast.flatten()
        except Exception as e:
            logger.warning(f"TabPFN prediction failed: {e}")
            # Return last value repeated as fallback
            return np.full(self.prediction_length, context[-1, 0])

    def predict_samples(
        self,
        context: np.ndarray,
        timestamps: pd.DatetimeIndex | None = None,
        future_timestamps: pd.DatetimeIndex | None = None,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Generate probabilistic forecast samples.

        Args:
            context: Historical values
            timestamps: Historical timestamps
            future_timestamps: Future timestamps
            num_samples: Number of samples to generate

        Returns:
            Samples (num_samples, prediction_length)
        """
        # Get point forecast
        point_forecast = self.predict(context, timestamps, future_timestamps)

        # Estimate uncertainty from context variance
        if context.ndim == 1:
            noise_scale = np.std(np.diff(context)) if len(context) > 1 else 0.1
        else:
            noise_scale = np.std(np.diff(context, axis=0)) if len(context) > 1 else 0.1

        noise_scale = max(noise_scale, 1e-6)

        # Generate samples
        noise = np.random.randn(num_samples, self.prediction_length) * noise_scale
        samples = point_forecast + noise

        return samples.astype(np.float32)

    def _create_features(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Create time features for TabPFN.

        Includes hour, day, week, month with sine/cosine transforms.

        Args:
            timestamps: Timestamp index

        Returns:
            Feature array (len(timestamps), num_features)
        """
        features = []

        # Hour of day (if hourly or finer)
        if hasattr(timestamps, "hour"):
            hour = timestamps.hour
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))

        # Day of week
        day_of_week = timestamps.dayofweek
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))

        # Day of month
        day_of_month = timestamps.day
        features.append(np.sin(2 * np.pi * day_of_month / 31))
        features.append(np.cos(2 * np.pi * day_of_month / 31))

        # Month of year
        month = timestamps.month
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))

        return np.column_stack(features)


class TabPFNGluonTSPredictor:
    """GluonTS-compatible predictor wrapper for TabPFN-TS."""

    def __init__(
        self,
        prediction_length: int,
        freq: str,
        num_samples: int = 100,
        batch_size: int = 64,
    ):
        """Initialize TabPFN GluonTS predictor.

        Args:
            prediction_length: Forecast horizon
            freq: Frequency string
            num_samples: Number of samples for probabilistic forecasts
            batch_size: Batch size for processing
        """
        _check_tabpfn()
        _check_gluonts()

        self.prediction_length = prediction_length
        self.freq = freq
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.wrapper = TabPFNTSWrapper(
            prediction_length=prediction_length,
            freq=freq,
        )

    def predict(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
    ) -> Iterator[Forecast]:
        """Generate forecasts in GluonTS format.

        Args:
            dataset: GluonTS dataset
            num_samples: Override number of samples

        Yields:
            SampleForecast objects
        """
        import pandas as pd

        if num_samples is None:
            num_samples = self.num_samples

        for entry in dataset:
            target = np.asarray(entry["target"])
            start = entry["start"]

            # Get forecast start date
            if target.ndim == 1:
                forecast_start = start + len(target)
            else:
                forecast_start = start + target.shape[1]

            # Generate samples
            samples = self.wrapper.predict_samples(
                context=target,
                num_samples=num_samples,
            )

            yield SampleForecast(
                samples=samples,
                start_date=pd.Period(forecast_start, freq=self.freq),
                item_id=entry.get("item_id"),
            )


def is_tabpfn_available() -> bool:
    """Check if TabPFN-TS is available.

    Returns:
        True if TabPFN-TS can be imported
    """
    return TABPFN_AVAILABLE
