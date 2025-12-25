"""GluonTS-compatible predictor adapters for TinyTimeMixers.

This module provides predictor classes that wrap TTM models to be
compatible with GluonTS evaluation framework and GIFT-Eval benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from gluonts.dataset.common import Dataset
    from gluonts.model.forecast import Forecast

try:
    from gluonts.model.forecast import SampleForecast
    from gluonts.model.predictor import RepresentablePredictor

    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False
    RepresentablePredictor = object
    SampleForecast = None


def _check_gluonts():
    """Check if GluonTS is available."""
    if not GLUONTS_AVAILABLE:
        raise ImportError(
            "GluonTS is required for benchmark adapters. "
            "Install with: pip install gluonts"
        )


class TTMGluonTSPredictor(RepresentablePredictor):
    """GluonTS-compatible predictor wrapper for TTM models.

    Wraps a TinyTimeMixers model to be compatible with GluonTS
    evaluation framework and GIFT-Eval benchmark requirements.

    The predictor converts GluonTS dataset format to TTM tensor format,
    runs inference, and returns SampleForecast objects.
    """

    def __init__(
        self,
        model: nn.Module,
        prediction_length: int,
        freq: str,
        context_length: int = 512,
        num_samples: int = 100,
        device: str = "auto",
        batch_size: int = 64,
        lead_time: int = 0,
    ):
        """Initialize TTM GluonTS predictor.

        Args:
            model: Pre-trained TTM model
            prediction_length: Forecast horizon
            freq: Frequency string (e.g., "H", "D", "W")
            context_length: Input context length
            num_samples: Number of samples for probabilistic forecasts
            device: Device for inference ("auto", "cpu", "cuda")
            batch_size: Batch size for parallel prediction
            lead_time: Lead time for forecasts
        """
        _check_gluonts()
        super().__init__(prediction_length=prediction_length, lead_time=lead_time)

        self.freq = freq
        self.context_length = context_length
        self.num_samples = num_samples
        self.batch_size = batch_size

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

    def predict(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
    ) -> Iterator[Forecast]:
        """Generate forecasts for all time series in dataset.

        Args:
            dataset: GluonTS dataset with time series
            num_samples: Override number of samples (default: use init value)

        Yields:
            SampleForecast objects for each time series
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Collect series for batched processing
        entries = list(dataset)

        for i in range(0, len(entries), self.batch_size):
            batch_entries = entries[i: i + self.batch_size]

            # Prepare batch contexts
            contexts = []
            start_dates = []
            item_ids = []

            for entry in batch_entries:
                target = np.asarray(entry["target"])
                context = self._prepare_context(target)
                contexts.append(context)

                # Get forecast start date
                start = entry["start"]
                forecast_start = start + len(target)
                start_dates.append(forecast_start)

                item_ids.append(entry.get("item_id", None))

            # Stack into batch tensor
            batch_context = torch.stack(contexts, dim=0).to(self.device)

            # Generate point forecasts
            with torch.no_grad():
                point_forecasts = self.model(batch_context).cpu().numpy()

            # Generate samples for each series
            for j, (point_forecast, context, start_date) in enumerate(
                zip(point_forecasts, contexts, start_dates)
            ):
                samples = self._generate_samples(
                    point_forecast,
                    context.numpy(),
                    num_samples,
                )

                yield SampleForecast(
                    samples=samples,
                    start_date=pd.Period(start_date, freq=self.freq),
                    item_id=item_ids[j],
                )

    def _prepare_context(
        self,
        target: np.ndarray,
    ) -> torch.Tensor:
        """Prepare context window from time series.

        Handles padding for short series and truncation for long series.

        Args:
            target: Time series values (seq_len,) or (num_variates, seq_len)

        Returns:
            Context tensor (channels, context_length)
        """
        # Handle multivariate vs univariate
        if target.ndim == 1:
            target = target.reshape(1, -1)  # (1, seq_len)

        num_channels, seq_len = target.shape

        if seq_len >= self.context_length:
            # Truncate to context length (take last context_length values)
            context = target[:, -self.context_length:]
        else:
            # Pad with zeros at the beginning
            padding = np.zeros((num_channels, self.context_length - seq_len))
            context = np.concatenate([padding, target], axis=1)

        return torch.from_numpy(context).float()

    def _generate_samples(
        self,
        point_forecast: np.ndarray,
        context: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """Generate probabilistic samples from point forecast.

        Uses residual-based sampling: estimate variance from context
        and add noise to point forecast.

        Args:
            point_forecast: Point predictions (channels, prediction_length)
            context: Context window (channels, context_length)
            num_samples: Number of samples to generate

        Returns:
            Samples (num_samples, prediction_length) for univariate
            or (num_samples, num_variates, prediction_length) for multivariate
        """
        # Estimate noise scale from context residuals
        # Use simple differencing to estimate local variance
        if context.shape[1] > 1:
            diffs = np.diff(context, axis=1)
            noise_scale = np.std(diffs, axis=1, keepdims=True)
        else:
            noise_scale = np.abs(context).mean(axis=1, keepdims=True) * 0.1

        # Ensure minimum noise scale
        noise_scale = np.maximum(noise_scale, 1e-6)

        # Generate samples by adding noise to point forecast
        if point_forecast.ndim == 1:
            point_forecast = point_forecast.reshape(1, -1)

        num_channels, pred_len = point_forecast.shape

        if num_channels == 1:
            # Univariate: return (num_samples, prediction_length)
            noise = np.random.randn(num_samples, pred_len) * noise_scale[0, 0]
            samples = point_forecast[0] + noise
        else:
            # Multivariate: return (num_samples, num_variates, prediction_length)
            noise = (
                np.random.randn(num_samples, num_channels, pred_len)
                * noise_scale[:, np.newaxis]
            )
            samples = point_forecast + noise

        return samples.astype(np.float32)


class TTMZeroShotPredictor(TTMGluonTSPredictor):
    """Zero-shot predictor using pre-trained TTM without fine-tuning.

    Uses instance normalization for better zero-shot generalization.
    """

    def __init__(
        self,
        model: nn.Module,
        prediction_length: int,
        freq: str,
        context_length: int = 512,
        num_samples: int = 100,
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        """Initialize zero-shot predictor.

        Args:
            model: Pre-trained TTM model
            prediction_length: Forecast horizon
            freq: Frequency string
            context_length: Input context length
            num_samples: Number of samples for probabilistic forecasts
            device: Device for inference
            batch_size: Batch size for prediction
            normalize: Whether to apply instance normalization
        """
        super().__init__(
            model=model,
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            num_samples=num_samples,
            device=device,
            batch_size=batch_size,
        )
        self.normalize = normalize

    def predict(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
    ) -> Iterator[Forecast]:
        """Generate zero-shot forecasts with optional normalization.

        Args:
            dataset: GluonTS dataset
            num_samples: Override number of samples

        Yields:
            SampleForecast objects
        """
        if num_samples is None:
            num_samples = self.num_samples

        entries = list(dataset)

        for i in range(0, len(entries), self.batch_size):
            batch_entries = entries[i: i + self.batch_size]

            contexts = []
            start_dates = []
            item_ids = []
            means = []
            stds = []

            for entry in batch_entries:
                target = np.asarray(entry["target"])
                context = self._prepare_context(target)
                contexts.append(context)

                start = entry["start"]
                forecast_start = start + len(target)
                start_dates.append(forecast_start)
                item_ids.append(entry.get("item_id", None))

            batch_context = torch.stack(contexts, dim=0).to(self.device)

            # Apply instance normalization if enabled
            if self.normalize:
                mean = batch_context.mean(dim=-1, keepdim=True)
                std = batch_context.std(dim=-1, keepdim=True) + 1e-8
                batch_context_norm = (batch_context - mean) / std
                means = mean.cpu().numpy()
                stds = std.cpu().numpy()
            else:
                batch_context_norm = batch_context
                means = [None] * len(batch_entries)
                stds = [None] * len(batch_entries)

            with torch.no_grad():
                point_forecasts = self.model(batch_context_norm).cpu().numpy()

            # Denormalize if needed
            if self.normalize:
                point_forecasts = point_forecasts * stds + means

            for j, (point_forecast, context, start_date) in enumerate(
                zip(point_forecasts, contexts, start_dates)
            ):
                samples = self._generate_samples(
                    point_forecast,
                    context.numpy(),
                    num_samples,
                )

                yield SampleForecast(
                    samples=samples,
                    start_date=pd.Period(start_date, freq=self.freq),
                    item_id=item_ids[j],
                )


class TTMFewShotPredictor(TTMGluonTSPredictor):
    """Few-shot predictor with lightweight head adaptation.

    Fine-tunes only the forecast head on training data before prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        prediction_length: int,
        freq: str,
        context_length: int = 512,
        num_samples: int = 100,
        device: str = "auto",
        batch_size: int = 64,
    ):
        """Initialize few-shot predictor.

        Args:
            model: Pre-trained TTM model
            prediction_length: Forecast horizon
            freq: Frequency string
            context_length: Input context length
            num_samples: Number of samples for probabilistic forecasts
            device: Device for inference
            batch_size: Batch size for prediction
        """
        super().__init__(
            model=model,
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            num_samples=num_samples,
            device=device,
            batch_size=batch_size,
        )
        self._original_state = None

    def adapt(
        self,
        train_dataset: Dataset,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
    ):
        """Adapt model to training data before prediction.

        Only fine-tunes the forecast head, keeping backbone frozen.

        Args:
            train_dataset: GluonTS training dataset
            num_epochs: Number of adaptation epochs
            learning_rate: Learning rate for adaptation
        """
        # Save original state
        self._original_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }

        # Freeze backbone
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Prepare training data
        train_contexts = []
        train_targets = []

        for entry in train_dataset:
            target = np.asarray(entry["target"])
            if target.ndim == 1:
                target = target.reshape(1, -1)

            # Create context-target pairs using sliding window
            seq_len = target.shape[1]
            total_len = self.context_length + self.prediction_length

            if seq_len >= total_len:
                for start in range(0, seq_len - total_len + 1, self.prediction_length):
                    context = target[:, start: start + self.context_length]
                    tgt = target[
                        :,
                        start + self.context_length: start
                        + self.context_length
                        + self.prediction_length,
                    ]
                    train_contexts.append(torch.from_numpy(context).float())
                    train_targets.append(torch.from_numpy(tgt).float())

        if len(train_contexts) == 0:
            return  # No training data available

        train_contexts = torch.stack(train_contexts)
        train_targets = torch.stack(train_targets)

        # Training loop
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        criterion = torch.nn.MSELoss()

        dataset_size = len(train_contexts)
        indices = np.arange(dataset_size)

        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx: start_idx + self.batch_size]
                batch_context = train_contexts[batch_indices].to(self.device)
                batch_target = train_targets[batch_indices].to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_context)
                loss = criterion(pred, batch_target)
                loss.backward()
                optimizer.step()

        self.model.eval()

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

    def reset(self):
        """Reset model to original pre-trained state."""
        if self._original_state is not None:
            self.model.load_state_dict(self._original_state)
            self._original_state = None


def load_predictor_from_path(
    model_path: str,
    prediction_length: int,
    freq: str,
    context_length: int = 512,
    device: str = "auto",
    predictor_type: str = "zero_shot",
    **kwargs,
) -> TTMGluonTSPredictor:
    """Load TTM model and create predictor.

    Args:
        model_path: Path to saved TTM model
        prediction_length: Forecast horizon
        freq: Frequency string
        context_length: Input context length
        device: Device for inference
        predictor_type: Type of predictor ("base", "zero_shot", "few_shot")
        **kwargs: Additional predictor arguments

    Returns:
        TTMGluonTSPredictor instance
    """
    from tinytimemixers.models.ttm import TTM

    model = TTM.load(model_path, map_location=device)

    if predictor_type == "zero_shot":
        return TTMZeroShotPredictor(
            model=model,
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            device=device,
            **kwargs,
        )
    elif predictor_type == "few_shot":
        return TTMFewShotPredictor(
            model=model,
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            device=device,
            **kwargs,
        )
    else:
        return TTMGluonTSPredictor(
            model=model,
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            device=device,
            **kwargs,
        )
