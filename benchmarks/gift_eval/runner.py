"""Benchmark runner for GIFT-Eval.

This module provides the main runner class for executing
the GIFT-Eval benchmark across all dataset configurations.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from benchmarks.gift_eval.adapter import TTMGluonTSPredictor
    from benchmarks.gift_eval.config import DatasetConfig

from benchmarks.gift_eval.config import GIFT_EVAL_DATASETS
from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader
from benchmarks.gift_eval.results import (
    BenchmarkResult,
    ModelMetadata,
    ResultsAggregator,
)

try:
    import gluonts  # noqa: F401

    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False


logger = logging.getLogger(__name__)


class GIFTEvalRunner:
    """Runner for GIFT-Eval benchmark across all configurations.

    Provides sequential execution with checkpointing and resume capability.
    """

    def __init__(
        self,
        predictor_factory: Callable[[DatasetConfig], TTMGluonTSPredictor],
        output_dir: str | Path = "results/ttm",
        checkpoint_interval: int = 10,
        resume: bool = True,
        device: str = "auto",
    ):
        """Initialize benchmark runner.

        Args:
            predictor_factory: Factory function to create predictor for each config.
                               Takes DatasetConfig and returns TTMGluonTSPredictor.
            output_dir: Directory to save results and checkpoints
            checkpoint_interval: Save progress every N datasets
            resume: Resume from previous checkpoint if available
            device: Device for inference
        """
        self.predictor_factory = predictor_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.resume = resume
        self.device = device

        self.dataset_loader = GIFTEvalDatasetLoader()
        self._completed_configs: set[str] = set()
        self._results: list[BenchmarkResult] = []

        if resume:
            self._load_checkpoint()

    def run_all(
        self,
        configs: list[DatasetConfig] | None = None,
        domains: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmark on all or selected configurations.

        Args:
            configs: Specific configs to run (None for all 98)
            domains: Filter by domains (e.g., ["Energy", "Web"])

        Returns:
            List of benchmark results
        """
        if configs is None:
            configs = GIFT_EVAL_DATASETS

        if domains is not None:
            configs = [c for c in configs if c.domain in domains]

        logger.info(f"Running benchmark on {len(configs)} configurations")

        for i, config in enumerate(configs):
            # Skip if already completed (resume mode)
            if config.config_name in self._completed_configs:
                logger.info(f"Skipping {config.config_name} (already completed)")
                continue

            logger.info(f"[{i + 1}/{len(configs)}] Running {config.config_name}...")

            try:
                result = self.run_single(config)
                self._results.append(result)
                self._completed_configs.add(config.config_name)

                if result.success:
                    logger.info(
                        f"  Completed in {result.runtime_seconds:.2f}s - "
                        f"MSE: {result.metrics.get('MSE', 'N/A'):.4f}"
                    )
                else:
                    logger.warning(f"  Failed: {result.error}")

            except Exception as e:
                logger.error(f"  Error: {e}")
                result = BenchmarkResult(
                    config_name=config.config_name,
                    dataset_name=config.name,
                    domain=config.domain,
                    num_variates=config.num_variates,
                    error=str(e),
                )
                self._results.append(result)
                self._completed_configs.add(config.config_name)

            # Checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()

        return self._results

    def run_single(
        self,
        config: DatasetConfig,
    ) -> BenchmarkResult:
        """Run benchmark on a single configuration.

        Args:
            config: Dataset configuration

        Returns:
            BenchmarkResult with metrics or error
        """
        start_time = time.time()

        try:
            # Load dataset
            train_dataset, test_dataset = self.dataset_loader.load_dataset(config)

            # Create predictor
            predictor = self.predictor_factory(config)

            # Run evaluation
            metrics = self._evaluate_predictor(
                predictor,
                test_dataset,
                config.prediction_length,
                config.seasonality,
            )

            runtime = time.time() - start_time

            return BenchmarkResult(
                config_name=config.config_name,
                dataset_name=config.name,
                domain=config.domain,
                num_variates=config.num_variates,
                metrics=metrics,
                runtime_seconds=runtime,
            )

        except Exception as e:
            runtime = time.time() - start_time
            return BenchmarkResult(
                config_name=config.config_name,
                dataset_name=config.name,
                domain=config.domain,
                num_variates=config.num_variates,
                runtime_seconds=runtime,
                error=str(e),
            )

    def _evaluate_predictor(
        self,
        predictor: TTMGluonTSPredictor,
        test_dataset,
        prediction_length: int,
        seasonality: int,
    ) -> dict[str, float]:
        """Evaluate predictor on test dataset.

        Args:
            predictor: TTM GluonTS predictor
            test_dataset: GluonTS test dataset
            prediction_length: Prediction horizon
            seasonality: Seasonal period

        Returns:
            Dictionary of metrics
        """
        # Collect forecasts and actuals
        forecasts = list(predictor.predict(test_dataset))

        # Extract targets from test dataset
        targets = []
        for entry in test_dataset:
            target = np.asarray(entry["target"])
            if target.ndim == 1:
                # Take last prediction_length values as target
                targets.append(target[-prediction_length:])
            else:
                targets.append(target[:, -prediction_length:])

        # Compute metrics
        metrics = {}

        for forecast, target in zip(forecasts, targets):
            # Get point forecast (median)
            if hasattr(forecast, "median"):
                pred = forecast.median
            elif hasattr(forecast, "mean"):
                pred = forecast.mean
            else:
                pred = forecast.samples.mean(axis=0)

            # Ensure shapes match
            if pred.shape != target.shape:
                min_len = min(len(pred), len(target))
                pred = pred[:min_len]
                target = target[:min_len]

            # Compute individual metrics
            mse = np.mean((pred - target) ** 2)
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(mse)

            # MAPE (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                mape = np.mean(
                    np.abs((target - pred) / np.where(target != 0, target, 1))
                )
                mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)

            # SMAPE
            smape = np.mean(
                2 * np.abs(pred - target) / (np.abs(pred) + np.abs(target) + 1e-8)
            )

            # Accumulate
            for name, value in [
                ("MSE", mse),
                ("MAE", mae),
                ("RMSE", rmse),
                ("MAPE", mape),
                ("SMAPE", smape),
            ]:
                if name not in metrics:
                    metrics[name] = []
                metrics[name].append(value)

        # Average across all series
        return {name: np.mean(values) for name, values in metrics.items()}

    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint_path = self.output_dir / "checkpoint.json"

        checkpoint = {
            "completed_configs": list(self._completed_configs),
            "results": [r.to_dict() for r in self._results],
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint ({len(self._completed_configs)} completed)")

    def _load_checkpoint(self):
        """Load previous checkpoint if available."""
        checkpoint_path = self.output_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return

        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)

            self._completed_configs = set(checkpoint.get("completed_configs", []))
            self._results = [
                BenchmarkResult(**r) for r in checkpoint.get("results", [])
            ]

            logger.info(f"Loaded checkpoint ({len(self._completed_configs)} completed)")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def get_results(self) -> list[BenchmarkResult]:
        """Get current results.

        Returns:
            List of completed BenchmarkResult objects
        """
        return self._results

    def get_aggregator(
        self,
        model_metadata: ModelMetadata,
    ) -> ResultsAggregator:
        """Create results aggregator with current results.

        Args:
            model_metadata: Model metadata for submission

        Returns:
            ResultsAggregator with all results
        """
        aggregator = ResultsAggregator(model_metadata)
        aggregator.add_results(self._results)
        return aggregator

    def save_results(
        self,
        model_metadata: ModelMetadata,
    ):
        """Save final results in GIFT-Eval format.

        Args:
            model_metadata: Model metadata for submission
        """
        aggregator = self.get_aggregator(model_metadata)
        aggregator.prepare_submission(self.output_dir)

    def clear_checkpoint(self):
        """Clear checkpoint and reset progress."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        self._completed_configs = set()
        self._results = []
        logger.info("Cleared checkpoint")


def run_gift_eval(
    model_path: str,
    output_dir: str = "results/ttm",
    domains: list[str] | None = None,
    device: str = "auto",
    context_length: int = 512,
    num_samples: int = 100,
    batch_size: int = 64,
    resume: bool = True,
) -> ResultsAggregator:
    """Convenience function to run GIFT-Eval benchmark.

    Args:
        model_path: Path to saved TTM model
        output_dir: Output directory for results
        domains: Filter by domains (None for all)
        device: Device for inference
        context_length: Context length for predictions
        num_samples: Number of samples for probabilistic forecasts
        batch_size: Batch size for inference
        resume: Resume from checkpoint

    Returns:
        ResultsAggregator with all results
    """
    from benchmarks.gift_eval.adapter import load_predictor_from_path
    from tinytimemixers.models.ttm import TTM

    # Load model to get metadata
    model = TTM.load(model_path, map_location=device)

    model_metadata = ModelMetadata(
        model_name="TTM",
        model_version="1.0.0",
        num_parameters=model.get_num_parameters(),
        context_length=context_length,
    )

    def predictor_factory(config: DatasetConfig):
        return load_predictor_from_path(
            model_path=model_path,
            prediction_length=config.prediction_length,
            freq=config.freq,
            context_length=min(context_length, config.context_length),
            device=device,
            predictor_type="zero_shot",
            num_samples=num_samples,
            batch_size=batch_size,
        )

    runner = GIFTEvalRunner(
        predictor_factory=predictor_factory,
        output_dir=output_dir,
        resume=resume,
        device=device,
    )

    runner.run_all(domains=domains)
    runner.save_results(model_metadata)

    return runner.get_aggregator(model_metadata)
