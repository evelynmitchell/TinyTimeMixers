"""Results aggregation for GIFT-Eval benchmark.

This module provides utilities for aggregating benchmark results
into the GIFT-Eval CSV format for leaderboard submission.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for model submission to GIFT-Eval leaderboard.

    Attributes:
        model_name: Model identifier (e.g., "TTM")
        model_version: Version string
        model_type: Type (pretrained, fine-tuned, zero-shot, etc.)
        num_parameters: Total number of parameters
        context_length: Maximum context length
        prediction_length_max: Maximum supported prediction length
        training_data: Description of training data
        github_url: Repository URL
        paper_url: Paper URL (optional)
        model_dtype: Data type (e.g., "float32")
        testdata_leakage: Whether test data was used in training
        replication_code_available: Whether code is available
    """

    model_name: str
    model_version: str = "1.0.0"
    model_type: str = "pretrained"
    num_parameters: int = 0
    context_length: int = 512
    prediction_length_max: int = 720
    training_data: str = "Monash Time Series Repository"
    github_url: str = ""
    paper_url: str | None = None
    model_dtype: str = "float32"
    testdata_leakage: bool = False
    replication_code_available: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "num_parameters": self.num_parameters,
            "context_length": self.context_length,
            "prediction_length_max": self.prediction_length_max,
            "training_data": self.training_data,
            "github_url": self.github_url,
            "paper_url": self.paper_url,
            "model_dtype": self.model_dtype,
            "testdata_leakage": "No" if not self.testdata_leakage else "Yes",
            "replication_code_available": (
                "Yes" if self.replication_code_available else "No"
            ),
            "submission_date": datetime.now().isoformat(),
        }


@dataclass
class BenchmarkResult:
    """Result from evaluating a single dataset configuration.

    Attributes:
        config_name: Unique dataset config name
        dataset_name: Dataset name
        domain: Domain category
        num_variates: Number of variates
        metrics: Dictionary of metric values
        runtime_seconds: Evaluation runtime
        error: Error message if evaluation failed
    """

    config_name: str
    dataset_name: str
    domain: str
    num_variates: int
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether evaluation completed successfully."""
        return self.error is None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "config_name": self.config_name,
            "dataset_name": self.dataset_name,
            "domain": self.domain,
            "num_variates": self.num_variates,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "error": self.error,
        }


class ResultsAggregator:
    """Aggregates benchmark results into GIFT-Eval CSV format.

    The output CSV follows GIFT-Eval requirements with 15 columns:
    - dataset, model, domain, num_variates
    - 11 evaluation metrics
    """

    # GIFT-Eval required metric columns
    METRIC_COLUMNS = [
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
    ]

    # Mapping from our metric names to GIFT-Eval column names
    METRIC_MAPPING = {
        "MSE": "eval_metrics/MSE[mean]",
        "MAE": "eval_metrics/MAE[0.5]",
        "MASE": "eval_metrics/MASE[0.5]",
        "MAPE": "eval_metrics/MAPE[0.5]",
        "SMAPE": "eval_metrics/sMAPE[0.5]",
        "RMSE": "eval_metrics/RMSE[mean]",
        "CRPS": "eval_metrics/mean_weighted_sum_quantile_loss",
    }

    def __init__(
        self,
        model_metadata: ModelMetadata,
    ):
        """Initialize results aggregator.

        Args:
            model_metadata: Model metadata for config.json
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for results aggregation")

        self.model_metadata = model_metadata
        self.results: list[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result.

        Args:
            result: BenchmarkResult to add
        """
        self.results.append(result)

    def add_results(self, results: list[BenchmarkResult]):
        """Add multiple benchmark results.

        Args:
            results: List of BenchmarkResult objects
        """
        self.results.extend(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame with GIFT-Eval format.

        Returns:
            DataFrame with correct column structure
        """
        rows = []

        for result in self.results:
            row = {
                "dataset": result.config_name,
                "model": self.model_metadata.model_name,
                "domain": result.domain,
                "num_variates": result.num_variates,
            }

            # Map metrics to GIFT-Eval columns
            for our_name, gift_name in self.METRIC_MAPPING.items():
                if our_name in result.metrics:
                    row[gift_name] = result.metrics[our_name]
                else:
                    row[gift_name] = np.nan

            # Fill in remaining required columns
            for col in self.METRIC_COLUMNS:
                if col not in row:
                    row[col] = np.nan

            rows.append(row)

        # Create DataFrame with correct column order
        columns = ["dataset", "model", "domain", "num_variates"] + self.METRIC_COLUMNS
        df = pd.DataFrame(rows, columns=columns)

        return df

    def save_csv(self, path: str | Path):
        """Save results to all_results.csv.

        Args:
            path: Output path for CSV file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved results to {path}")

    def save_config_json(self, path: str | Path):
        """Save model metadata to config.json.

        Args:
            path: Output path for JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_metadata.to_dict(), f, indent=2)
        logger.info(f"Saved config to {path}")

    def prepare_submission(self, output_dir: str | Path):
        """Prepare complete submission package.

        Creates:
            output_dir/
            ├── all_results.csv
            └── config.json

        Args:
            output_dir: Output directory for submission files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.save_csv(output_dir / "all_results.csv")
        self.save_config_json(output_dir / "config.json")

        logger.info(f"Prepared submission package in {output_dir}")

    def compute_summary_statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics across all datasets.

        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()

        summary = {
            "num_datasets": len(self.results),
            "num_successful": sum(1 for r in self.results if r.success),
            "num_failed": sum(1 for r in self.results if not r.success),
            "total_runtime_seconds": sum(r.runtime_seconds for r in self.results),
            "domains": {},
        }

        # Per-domain statistics
        for domain in df["domain"].unique():
            domain_df = df[df["domain"] == domain]
            summary["domains"][domain] = {
                "num_datasets": len(domain_df),
                "mean_MSE": domain_df["eval_metrics/MSE[mean]"].mean(),
                "mean_MAE": domain_df["eval_metrics/MAE[0.5]"].mean(),
                "mean_MASE": domain_df["eval_metrics/MASE[0.5]"].mean(),
            }

        # Overall averages
        for col in self.METRIC_COLUMNS:
            summary[f"mean_{col}"] = df[col].mean()

        return summary

    def compute_rank(
        self,
        baseline_results: pd.DataFrame | None = None,
        metric: str = "eval_metrics/MASE[0.5]",
    ) -> float:
        """Compute average rank metric for leaderboard.

        Args:
            baseline_results: DataFrame with baseline model results
            metric: Metric column to use for ranking

        Returns:
            Average rank (1.0 = best possible)
        """
        df = self.to_dataframe()

        if baseline_results is None:
            # Without baseline, return mean metric value
            return df[metric].mean()

        # Compute rank against baseline for each dataset
        ranks = []
        for _, row in df.iterrows():
            dataset = row["dataset"]
            our_value = row[metric]

            baseline_row = baseline_results[baseline_results["dataset"] == dataset]
            if len(baseline_row) == 0:
                continue

            baseline_value = baseline_row[metric].iloc[0]

            # Lower is better, rank 1 if we're better
            if our_value < baseline_value:
                ranks.append(1)
            elif our_value > baseline_value:
                ranks.append(2)
            else:
                ranks.append(1.5)  # Tie

        return np.mean(ranks) if ranks else float("nan")


def validate_results_csv(path: str | Path) -> tuple[bool, list[str]]:
    """Validate results CSV against GIFT-Eval requirements.

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    if not PANDAS_AVAILABLE:
        return False, ["pandas is required for validation"]

    errors = []
    path = Path(path)

    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, [f"Failed to read CSV: {e}"]

    # Check required columns
    required_columns = [
        "dataset",
        "model",
        "domain",
        "num_variates",
    ] + ResultsAggregator.METRIC_COLUMNS

    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Check for empty values in key columns
    if df["dataset"].isna().any():
        errors.append("Missing values in 'dataset' column")
    if df["model"].isna().any():
        errors.append("Missing values in 'model' column")
    if df["domain"].isna().any():
        errors.append("Missing values in 'domain' column")

    # Check metric values are numeric
    for col in ResultsAggregator.METRIC_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Non-numeric values in column: {col}")

    is_valid = len(errors) == 0
    return is_valid, errors


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """Load results CSV file.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with results
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required")

    return pd.read_csv(path)


def merge_results(
    results_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge multiple results DataFrames.

    Args:
        results_list: List of DataFrames to merge

    Returns:
        Merged DataFrame
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required")

    return pd.concat(results_list, ignore_index=True)
