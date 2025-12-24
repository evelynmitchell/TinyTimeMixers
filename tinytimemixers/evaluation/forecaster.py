"""Forecaster classes for time series prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinytimemixers.data.dataset import TimeSeriesDataset
from tinytimemixers.evaluation.metrics import MetricTracker, compute_all_metrics


class Forecaster:
    """Wrapper for making forecasts with TTM models."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        batch_size: int = 64,
    ):
        """Initialize Forecaster.

        Args:
            model: TTM model
            device: Device for inference
            batch_size: Batch size for evaluation
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Make point predictions.

        Args:
            context: Context window (batch, channels, context_length)

        Returns:
            Predictions (batch, channels, prediction_length)
        """
        context = context.to(self.device)
        return self.model(context)

    @torch.no_grad()
    def predict_dataset(
        self,
        dataset: TimeSeriesDataset,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict on entire dataset.

        Args:
            dataset: Time series dataset

        Returns:
            Tuple of (contexts, predictions, targets)
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_contexts = []
        all_preds = []
        all_targets = []

        for context, target in loader:
            pred = self.predict(context)
            all_contexts.append(context)
            all_preds.append(pred.cpu())
            all_targets.append(target)

        return (
            torch.cat(all_contexts, dim=0),
            torch.cat(all_preds, dim=0),
            torch.cat(all_targets, dim=0),
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataset: TimeSeriesDataset,
        seasonality: int = 1,
    ) -> dict[str, float]:
        """Evaluate model on dataset.

        Args:
            dataset: Time series dataset
            seasonality: Seasonal period for MASE

        Returns:
            Dictionary of metrics
        """
        contexts, preds, targets = self.predict_dataset(dataset)

        return compute_all_metrics(
            preds,
            targets,
            context=contexts,
            seasonality=seasonality,
        )

    @torch.no_grad()
    def evaluate_rolling(
        self,
        data: torch.Tensor,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        seasonality: int = 1,
    ) -> dict[str, float]:
        """Rolling evaluation over time series.

        Args:
            data: Full time series (channels, seq_len)
            context_length: Context window length
            prediction_length: Prediction horizon
            stride: Step between evaluation windows
            seasonality: Seasonal period for MASE

        Returns:
            Dictionary of average metrics
        """
        if data.dim() == 2:
            data = data.unsqueeze(0)  # Add batch dimension

        dataset = TimeSeriesDataset(
            data,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
        )

        return self.evaluate(dataset, seasonality=seasonality)


class ZeroShotForecaster(Forecaster):
    """Zero-shot forecaster for unseen datasets.

    Uses pre-trained model without any fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        """Initialize ZeroShotForecaster.

        Args:
            model: Pre-trained TTM model
            device: Device for inference
            batch_size: Batch size for evaluation
            normalize: Whether to apply instance normalization
        """
        super().__init__(model, device, batch_size)
        self.normalize = normalize

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Make zero-shot predictions.

        Optionally applies instance normalization.

        Args:
            context: Context window (batch, channels, context_length)

        Returns:
            Predictions (batch, channels, prediction_length)
        """
        context = context.to(self.device)

        if self.normalize:
            # Instance normalization (RevIN-style)
            mean = context.mean(dim=-1, keepdim=True)
            std = context.std(dim=-1, keepdim=True) + 1e-8
            context_norm = (context - mean) / std

            pred_norm = self.model(context_norm)

            # Denormalize predictions
            pred = pred_norm * std + mean
        else:
            pred = self.model(context)

        return pred


class FewShotForecaster(Forecaster):
    """Few-shot forecaster with lightweight adaptation.

    Fine-tunes only the forecast head on a small number of samples.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        batch_size: int = 64,
        head_dropout: float = 0.7,
    ):
        """Initialize FewShotForecaster.

        Args:
            model: Pre-trained TTM model
            device: Device for inference
            batch_size: Batch size
            head_dropout: Dropout for forecast head
        """
        super().__init__(model, device, batch_size)
        self.head_dropout = head_dropout
        self._original_state = None

    def adapt(
        self,
        train_data: torch.Tensor,
        context_length: int,
        prediction_length: int,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
    ):
        """Adapt model to new dataset.

        Only fine-tunes the forecast head.

        Args:
            train_data: Training data (num_series, channels, seq_len)
            context_length: Context window length
            prediction_length: Prediction horizon
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

        # Create dataset
        dataset = TimeSeriesDataset(
            train_data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Train forecast head
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        criterion = torch.nn.MSELoss()

        for _ in range(num_epochs):
            for context, target in loader:
                context = context.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                pred = self.model(context)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

        # Switch back to eval mode
        self.model.eval()

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

    def reset(self):
        """Reset model to original pre-trained state."""
        if self._original_state is not None:
            self.model.load_state_dict(self._original_state)
            self._original_state = None


class EnsembleForecaster:
    """Ensemble of multiple forecasters.

    Combines predictions from multiple models.
    """

    def __init__(
        self,
        forecasters: list[Forecaster],
        aggregation: str = "mean",
    ):
        """Initialize EnsembleForecaster.

        Args:
            forecasters: List of forecasters
            aggregation: How to combine predictions ("mean", "median")
        """
        self.forecasters = forecasters
        self.aggregation = aggregation

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Make ensemble predictions.

        Args:
            context: Context window

        Returns:
            Aggregated predictions
        """
        predictions = [f.predict(context) for f in self.forecasters]
        stacked = torch.stack(predictions, dim=0)

        if self.aggregation == "mean":
            return stacked.mean(dim=0)
        elif self.aggregation == "median":
            return stacked.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty estimate.

        Args:
            context: Context window

        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        predictions = [f.predict(context) for f in self.forecasters]
        stacked = torch.stack(predictions, dim=0)

        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        return mean, std

    def evaluate(
        self,
        dataset: TimeSeriesDataset,
        seasonality: int = 1,
    ) -> dict[str, float]:
        """Evaluate ensemble on dataset.

        Args:
            dataset: Time series dataset
            seasonality: Seasonal period for MASE

        Returns:
            Dictionary of metrics
        """
        loader = DataLoader(
            dataset,
            batch_size=self.forecasters[0].batch_size,
            shuffle=False,
            num_workers=0,
        )

        tracker = MetricTracker(["MSE", "MAE", "RMSE", "MAPE", "SMAPE", "MASE"])

        for context, target in loader:
            pred = self.predict(context)
            tracker.update(
                pred,
                target,
                context=context,
                seasonality=seasonality,
            )

        return tracker.compute()


def evaluate_model(
    model: nn.Module,
    test_data: torch.Tensor,
    context_length: int,
    prediction_length: int,
    batch_size: int = 64,
    device: str = "auto",
    seasonality: int = 1,
) -> dict[str, float]:
    """Convenience function to evaluate a model.

    Args:
        model: TTM model
        test_data: Test data (num_series, channels, seq_len)
        context_length: Context window length
        prediction_length: Prediction horizon
        batch_size: Batch size
        device: Device to use
        seasonality: Seasonal period for MASE

    Returns:
        Dictionary of metrics
    """
    forecaster = Forecaster(model, device=device, batch_size=batch_size)

    dataset = TimeSeriesDataset(
        test_data,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    return forecaster.evaluate(dataset, seasonality=seasonality)
