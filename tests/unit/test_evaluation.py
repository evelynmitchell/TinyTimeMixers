"""Unit tests for evaluation module."""

import numpy as np
import pytest
import torch

from tinytimemixers.config import TTMConfig
from tinytimemixers.data.dataset import TimeSeriesDataset
from tinytimemixers.evaluation.forecaster import (
    EnsembleForecaster,
    FewShotForecaster,
    Forecaster,
    ZeroShotForecaster,
    evaluate_model,
)
from tinytimemixers.evaluation.metrics import (
    CRPS,
    MAE,
    MAPE,
    MASE,
    MSE,
    RMSE,
    SMAPE,
    MetricTracker,
    compute_all_metrics,
    coverage,
    interval_width,
)
from tinytimemixers.models.ttm import TTM


class TestMetrics:
    """Test evaluation metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample prediction and target data."""
        np.random.seed(42)
        pred = np.random.randn(32, 3, 96)
        target = np.random.randn(32, 3, 96)
        return pred, target

    def test_mse(self, sample_data):
        """Test MSE computation."""
        pred, target = sample_data
        mse = MSE(pred, target)
        assert isinstance(mse, float)
        assert mse > 0

        # Perfect prediction should have MSE = 0
        assert MSE(pred, pred) == 0.0

    def test_mae(self, sample_data):
        """Test MAE computation."""
        pred, target = sample_data
        mae = MAE(pred, target)
        assert isinstance(mae, float)
        assert mae > 0

        # Perfect prediction should have MAE = 0
        assert MAE(pred, pred) == 0.0

    def test_rmse(self, sample_data):
        """Test RMSE computation."""
        pred, target = sample_data
        rmse = RMSE(pred, target)
        mse = MSE(pred, target)

        assert isinstance(rmse, float)
        assert rmse == pytest.approx(np.sqrt(mse))

    def test_mape(self, sample_data):
        """Test MAPE computation."""
        pred, target = sample_data
        mape = MAPE(pred, target)
        assert isinstance(mape, float)
        assert mape >= 0

    def test_smape(self, sample_data):
        """Test SMAPE computation."""
        pred, target = sample_data
        smape = SMAPE(pred, target)
        assert isinstance(smape, float)
        assert 0 <= smape <= 200  # SMAPE is bounded

    def test_mase(self, sample_data):
        """Test MASE computation."""
        pred, target = sample_data
        context = np.random.randn(32, 3, 512)

        mase = MASE(pred, target, context=context, seasonality=1)
        assert isinstance(mase, float)
        assert mase > 0

    def test_mase_without_context(self, sample_data):
        """Test MASE without context (uses target)."""
        pred, target = sample_data
        mase = MASE(pred, target, seasonality=1)
        assert isinstance(mase, float)
        assert mase > 0

    def test_crps(self, sample_data):
        """Test CRPS computation."""
        _, target = sample_data
        # Generate samples
        pred_samples = np.random.randn(10, 32, 3, 96)

        crps = CRPS(pred_samples, target)
        assert isinstance(crps, float)
        assert crps >= 0

    def test_coverage(self):
        """Test prediction interval coverage."""
        target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pred_lower = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
        pred_upper = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        cov = coverage(pred_lower, pred_upper, target)
        assert cov == 1.0  # All targets within bounds

        # Partial coverage
        pred_lower_partial = np.array([0.5, 0.5, 1.5, 2.5, 3.5])
        cov_partial = coverage(pred_lower_partial, pred_upper, target)
        assert cov_partial == 0.8  # 4 out of 5

    def test_interval_width(self):
        """Test prediction interval width."""
        pred_lower = np.array([0.0, 1.0, 2.0])
        pred_upper = np.array([1.0, 2.0, 3.0])

        width = interval_width(pred_lower, pred_upper)
        assert width == 1.0

    def test_reduction_modes(self, sample_data):
        """Test different reduction modes."""
        pred, target = sample_data

        # Mean reduction
        mse_mean = MSE(pred, target, reduction="mean")
        assert isinstance(mse_mean, float)

        # No reduction
        mse_none = MSE(pred, target, reduction="none")
        assert isinstance(mse_none, np.ndarray)
        assert mse_none.shape[0] == pred.shape[0]

    def test_torch_tensor_input(self, sample_data):
        """Test metrics work with torch tensors."""
        pred_np, target_np = sample_data
        pred = torch.from_numpy(pred_np)
        target = torch.from_numpy(target_np)

        mse = MSE(pred, target)
        mae = MAE(pred, target)

        assert isinstance(mse, float)
        assert isinstance(mae, float)

    def test_compute_all_metrics(self, sample_data):
        """Test computing all metrics at once."""
        pred, target = sample_data
        context = np.random.randn(32, 3, 512)

        metrics = compute_all_metrics(pred, target, context=context, seasonality=1)

        assert "MSE" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert "SMAPE" in metrics
        assert "MASE" in metrics


class TestMetricTracker:
    """Test metric tracker."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = MetricTracker(["MSE", "MAE"])
        assert tracker.count == 0
        assert "MSE" in tracker.values
        assert "MAE" in tracker.values

    def test_tracker_update(self):
        """Test tracker update."""
        tracker = MetricTracker()
        pred = np.random.randn(10, 3, 96)
        target = np.random.randn(10, 3, 96)

        tracker.update(pred, target)
        assert tracker.count == 1

        tracker.update(pred, target)
        assert tracker.count == 2

    def test_tracker_compute(self):
        """Test tracker compute average."""
        tracker = MetricTracker(["MSE"])
        pred = np.random.randn(10, 3, 96)
        target = np.random.randn(10, 3, 96)

        tracker.update(pred, target)
        metrics = tracker.compute()

        assert "MSE" in metrics
        assert isinstance(metrics["MSE"], float)

    def test_tracker_reset(self):
        """Test tracker reset."""
        tracker = MetricTracker()
        pred = np.random.randn(10, 3, 96)
        target = np.random.randn(10, 3, 96)

        tracker.update(pred, target)
        tracker.reset()

        assert tracker.count == 0
        assert len(tracker.values["MSE"]) == 0


class TestForecaster:
    """Test forecaster classes."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and test data."""
        config = TTMConfig(
            context_length=64,
            prediction_length=16,
            patch_length=8,
            patch_stride=8,
            num_backbone_levels=2,
            blocks_per_level=1,
            feature_scaler=2,
        )
        model = TTM(config=config, num_channels=1)

        # Create synthetic data
        data = torch.randn(10, 1, 128)
        dataset = TimeSeriesDataset(
            data,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )

        return model, dataset, config

    def test_forecaster_predict(self, model_and_data):
        """Test forecaster prediction."""
        model, dataset, config = model_and_data
        forecaster = Forecaster(model, device="cpu")

        context, _ = dataset[0]
        context = context.unsqueeze(0)  # Add batch dim

        pred = forecaster.predict(context)
        assert pred.shape == (1, 1, config.prediction_length)

    def test_forecaster_predict_dataset(self, model_and_data):
        """Test forecaster on full dataset."""
        model, dataset, _ = model_and_data
        forecaster = Forecaster(model, device="cpu", batch_size=4)

        contexts, preds, targets = forecaster.predict_dataset(dataset)

        assert contexts.shape[0] == len(dataset)
        assert preds.shape[0] == len(dataset)
        assert targets.shape[0] == len(dataset)

    def test_forecaster_evaluate(self, model_and_data):
        """Test forecaster evaluation."""
        model, dataset, _ = model_and_data
        forecaster = Forecaster(model, device="cpu")

        metrics = forecaster.evaluate(dataset)

        assert "MSE" in metrics
        assert "MAE" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_zero_shot_forecaster(self, model_and_data):
        """Test zero-shot forecaster."""
        model, dataset, config = model_and_data
        forecaster = ZeroShotForecaster(model, device="cpu", normalize=True)

        context, _ = dataset[0]
        context = context.unsqueeze(0)

        pred = forecaster.predict(context)
        assert pred.shape == (1, 1, config.prediction_length)

    def test_few_shot_forecaster(self, model_and_data):
        """Test few-shot forecaster adaptation."""
        model, dataset, config = model_and_data
        forecaster = FewShotForecaster(model, device="cpu")

        # Adapt to new data
        data = torch.randn(5, 1, 128)
        forecaster.adapt(
            data,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
            num_epochs=2,
        )

        # Make prediction
        context, _ = dataset[0]
        context = context.unsqueeze(0)

        pred = forecaster.predict(context)
        assert pred.shape == (1, 1, config.prediction_length)

        # Reset to original
        forecaster.reset()

    def test_ensemble_forecaster(self, model_and_data):
        """Test ensemble forecaster."""
        model, dataset, config = model_and_data

        # Create multiple forecasters
        forecasters = [
            Forecaster(model, device="cpu"),
            Forecaster(model, device="cpu"),
        ]
        ensemble = EnsembleForecaster(forecasters, aggregation="mean")

        context, _ = dataset[0]
        context = context.unsqueeze(0)

        pred = ensemble.predict(context)
        assert pred.shape == (1, 1, config.prediction_length)

    def test_ensemble_with_uncertainty(self, model_and_data):
        """Test ensemble uncertainty estimation."""
        model, dataset, _ = model_and_data

        forecasters = [
            Forecaster(model, device="cpu"),
            Forecaster(model, device="cpu"),
        ]
        ensemble = EnsembleForecaster(forecasters)

        context, _ = dataset[0]
        context = context.unsqueeze(0)

        mean, std = ensemble.predict_with_uncertainty(context)
        assert mean.shape == std.shape

    def test_evaluate_model_convenience(self, model_and_data):
        """Test convenience evaluation function."""
        model, _, config = model_and_data
        test_data = torch.randn(5, 1, 128)

        metrics = evaluate_model(
            model,
            test_data,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
            device="cpu",
        )

        assert "MSE" in metrics
        assert "MAE" in metrics


class TestForecasterEdgeCases:
    """Test forecaster edge cases."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model."""
        config = TTMConfig(
            context_length=32,
            prediction_length=8,
            patch_length=8,
            patch_stride=8,
            num_backbone_levels=1,
            blocks_per_level=1,
            feature_scaler=2,
        )
        return TTM(config=config, num_channels=1)

    def test_rolling_evaluation(self, simple_model):
        """Test rolling evaluation."""
        forecaster = Forecaster(simple_model, device="cpu")

        # Single long series
        data = torch.randn(1, 64)

        metrics = forecaster.evaluate_rolling(
            data,
            context_length=32,
            prediction_length=8,
            stride=4,
        )

        assert "MSE" in metrics
        assert isinstance(metrics["MSE"], float)

    def test_multivariate_forecast(self, simple_model):
        """Test multivariate forecasting."""
        # Create multivariate model
        config = TTMConfig(
            context_length=32,
            prediction_length=8,
            patch_length=8,
            patch_stride=8,
            num_backbone_levels=1,
            blocks_per_level=1,
            feature_scaler=2,
        )
        model = TTM(config=config, num_channels=3)
        forecaster = Forecaster(model, device="cpu")

        context = torch.randn(4, 3, 32)
        pred = forecaster.predict(context)

        assert pred.shape == (4, 3, 8)
