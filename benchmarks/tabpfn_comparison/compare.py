"""Head-to-head comparison between TTM and TabPFN-TS.

This module provides utilities for comparing TinyTimeMixers
with TabPFN-TS on the GIFT-Eval benchmark.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from benchmarks.gift_eval.config import DatasetConfig

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None


logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from head-to-head comparison on a single dataset.

    Attributes:
        config_name: Dataset configuration name
        domain: Domain category
        ttm_metrics: Metrics for TTM model
        tabpfn_metrics: Metrics for TabPFN model
        ttm_runtime: TTM evaluation runtime in seconds
        tabpfn_runtime: TabPFN evaluation runtime in seconds
        winner: Winner for primary metric ("TTM", "TabPFN", "Tie")
    """

    config_name: str
    domain: str
    ttm_metrics: dict[str, float] = field(default_factory=dict)
    tabpfn_metrics: dict[str, float] = field(default_factory=dict)
    ttm_runtime: float = 0.0
    tabpfn_runtime: float = 0.0
    winner: str = "Tie"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "config_name": self.config_name,
            "domain": self.domain,
            "ttm_metrics": self.ttm_metrics,
            "tabpfn_metrics": self.tabpfn_metrics,
            "ttm_runtime": self.ttm_runtime,
            "tabpfn_runtime": self.tabpfn_runtime,
            "winner": self.winner,
        }


class ModelComparator:
    """Head-to-head comparison between TTM and TabPFN-TS."""

    def __init__(
        self,
        ttm_predictor_factory,
        tabpfn_predictor_factory=None,
        output_dir: str | Path = "results/comparison",
        primary_metric: str = "MASE",
    ):
        """Initialize comparator.

        Args:
            ttm_predictor_factory: Factory to create TTM predictor for each config
            tabpfn_predictor_factory: Factory to create TabPFN predictor (optional)
            output_dir: Output directory for comparison results
            primary_metric: Metric to use for determining winner
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for comparison")

        self.ttm_predictor_factory = ttm_predictor_factory
        self.tabpfn_predictor_factory = tabpfn_predictor_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.primary_metric = primary_metric

        self.results: list[ComparisonResult] = []

    def compare_single(
        self,
        config: DatasetConfig,
        test_dataset,
    ) -> ComparisonResult:
        """Compare models on a single dataset configuration.

        Args:
            config: Dataset configuration
            test_dataset: GluonTS test dataset

        Returns:
            ComparisonResult with metrics for both models
        """
        ttm_metrics = {}
        tabpfn_metrics = {}
        ttm_runtime = 0.0
        tabpfn_runtime = 0.0

        # Evaluate TTM
        try:
            ttm_predictor = self.ttm_predictor_factory(config)
            start_time = time.time()
            ttm_metrics = self._evaluate_predictor(
                ttm_predictor,
                test_dataset,
                config.prediction_length,
            )
            ttm_runtime = time.time() - start_time
        except Exception as e:
            logger.warning(f"TTM evaluation failed: {e}")

        # Evaluate TabPFN
        if self.tabpfn_predictor_factory is not None:
            try:
                tabpfn_predictor = self.tabpfn_predictor_factory(config)
                start_time = time.time()
                tabpfn_metrics = self._evaluate_predictor(
                    tabpfn_predictor,
                    test_dataset,
                    config.prediction_length,
                )
                tabpfn_runtime = time.time() - start_time
            except Exception as e:
                logger.warning(f"TabPFN evaluation failed: {e}")

        # Determine winner
        winner = self._determine_winner(
            ttm_metrics,
            tabpfn_metrics,
            self.primary_metric,
        )

        result = ComparisonResult(
            config_name=config.config_name,
            domain=config.domain,
            ttm_metrics=ttm_metrics,
            tabpfn_metrics=tabpfn_metrics,
            ttm_runtime=ttm_runtime,
            tabpfn_runtime=tabpfn_runtime,
            winner=winner,
        )

        self.results.append(result)
        return result

    def compare_all(
        self,
        configs: list[DatasetConfig],
        dataset_loader,
    ) -> list[ComparisonResult]:
        """Compare models on all configurations.

        Args:
            configs: List of dataset configurations
            dataset_loader: Dataset loader instance

        Returns:
            List of ComparisonResult objects
        """
        for i, config in enumerate(configs):
            logger.info(
                f"[{i + 1}/{len(configs)}] Comparing on {config.config_name}..."
            )

            try:
                _, test_dataset = dataset_loader.load_dataset(config)
                result = self.compare_single(config, test_dataset)
                logger.info(f"  Winner: {result.winner}")
            except Exception as e:
                logger.error(f"  Comparison failed: {e}")

        return self.results

    def _evaluate_predictor(
        self,
        predictor,
        test_dataset,
        prediction_length: int,
    ) -> dict[str, float]:
        """Evaluate a predictor on test dataset.

        Args:
            predictor: GluonTS-compatible predictor
            test_dataset: Test dataset
            prediction_length: Prediction horizon

        Returns:
            Dictionary of metrics
        """
        forecasts = list(predictor.predict(test_dataset))

        # Extract targets
        targets = []
        for entry in test_dataset:
            target = np.asarray(entry["target"])
            if target.ndim == 1:
                targets.append(target[-prediction_length:])
            else:
                targets.append(target[:, -prediction_length:])

        # Compute metrics
        metrics = {}
        for forecast, target in zip(forecasts, targets):
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

            mse = np.mean((pred - target) ** 2)
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(mse)

            for name, value in [("MSE", mse), ("MAE", mae), ("RMSE", rmse)]:
                if name not in metrics:
                    metrics[name] = []
                metrics[name].append(value)

        return {name: np.mean(values) for name, values in metrics.items()}

    def _determine_winner(
        self,
        ttm_metrics: dict[str, float],
        tabpfn_metrics: dict[str, float],
        metric: str,
    ) -> str:
        """Determine winner based on metric comparison.

        Args:
            ttm_metrics: TTM metric values
            tabpfn_metrics: TabPFN metric values
            metric: Metric to compare

        Returns:
            "TTM", "TabPFN", or "Tie"
        """
        ttm_value = ttm_metrics.get(metric)
        tabpfn_value = tabpfn_metrics.get(metric)

        if ttm_value is None and tabpfn_value is None:
            return "Tie"
        if ttm_value is None:
            return "TabPFN"
        if tabpfn_value is None:
            return "TTM"

        # Lower is better for error metrics
        if ttm_value < tabpfn_value * 0.99:  # 1% tolerance
            return "TTM"
        elif tabpfn_value < ttm_value * 0.99:
            return "TabPFN"
        else:
            return "Tie"

    def generate_report(self) -> pd.DataFrame:
        """Generate comparison summary report.

        Returns:
            DataFrame with per-dataset and aggregate comparisons
        """
        rows = []
        for result in self.results:
            row = {
                "dataset": result.config_name,
                "domain": result.domain,
                "winner": result.winner,
                "ttm_runtime": result.ttm_runtime,
                "tabpfn_runtime": result.tabpfn_runtime,
            }

            # Add metrics
            for metric in ["MSE", "MAE", "RMSE", "MASE"]:
                row[f"ttm_{metric}"] = result.ttm_metrics.get(metric)
                row[f"tabpfn_{metric}"] = result.tabpfn_metrics.get(metric)

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_statistics(self) -> dict:
        """Compute aggregate statistics.

        Returns:
            Dictionary with win/loss counts and averages
        """
        ttm_wins = sum(1 for r in self.results if r.winner == "TTM")
        tabpfn_wins = sum(1 for r in self.results if r.winner == "TabPFN")
        ties = sum(1 for r in self.results if r.winner == "Tie")

        stats_dict = {
            "total_comparisons": len(self.results),
            "ttm_wins": ttm_wins,
            "tabpfn_wins": tabpfn_wins,
            "ties": ties,
            "ttm_win_rate": ttm_wins / len(self.results) if self.results else 0,
        }

        # Average metrics
        for metric in ["MSE", "MAE", "RMSE"]:
            ttm_values = [
                r.ttm_metrics.get(metric)
                for r in self.results
                if r.ttm_metrics.get(metric) is not None
            ]
            tabpfn_values = [
                r.tabpfn_metrics.get(metric)
                for r in self.results
                if r.tabpfn_metrics.get(metric) is not None
            ]

            if ttm_values:
                stats_dict[f"ttm_avg_{metric}"] = np.mean(ttm_values)
            if tabpfn_values:
                stats_dict[f"tabpfn_avg_{metric}"] = np.mean(tabpfn_values)

        return stats_dict

    def compute_statistical_significance(
        self,
        metric: str = "MSE",
    ) -> dict:
        """Compute statistical significance of differences.

        Uses paired t-test or Wilcoxon signed-rank test.

        Args:
            metric: Metric to test

        Returns:
            Dictionary with test results
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy required for statistical tests"}

        ttm_values = []
        tabpfn_values = []

        for result in self.results:
            ttm_val = result.ttm_metrics.get(metric)
            tabpfn_val = result.tabpfn_metrics.get(metric)

            if ttm_val is not None and tabpfn_val is not None:
                ttm_values.append(ttm_val)
                tabpfn_values.append(tabpfn_val)

        if len(ttm_values) < 3:
            return {"error": "Not enough paired samples for statistical test"}

        ttm_arr = np.array(ttm_values)
        tabpfn_arr = np.array(tabpfn_values)

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(ttm_arr, tabpfn_arr)

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = stats.wilcoxon(ttm_arr, tabpfn_arr)
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan

        return {
            "metric": metric,
            "n_samples": len(ttm_values),
            "ttm_mean": np.mean(ttm_arr),
            "tabpfn_mean": np.mean(tabpfn_arr),
            "paired_ttest_statistic": t_stat,
            "paired_ttest_pvalue": t_pvalue,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_pvalue,
            "significant_at_0.05": t_pvalue < 0.05,
        }

    def save_results(self):
        """Save comparison results to output directory."""
        # Save report CSV
        report = self.generate_report()
        report.to_csv(self.output_dir / "comparison_report.csv", index=False)

        # Save statistics
        import json

        stats_dict = self.compute_statistics()
        with open(self.output_dir / "comparison_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=2)

        # Save significance tests
        for metric in ["MSE", "MAE", "RMSE"]:
            sig = self.compute_statistical_significance(metric)
            if "error" not in sig:
                with open(self.output_dir / f"significance_{metric}.json", "w") as f:
                    json.dump(sig, f, indent=2)

        logger.info(f"Saved comparison results to {self.output_dir}")
