"""Dataset configurations for GIFT-Eval benchmark.

This module defines the 98 dataset configurations across 7 domains
used in the GIFT-Eval benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetConfig:
    """Configuration for a single GIFT-Eval dataset.

    Attributes:
        name: Dataset name
        domain: Domain category (Energy, Web, Finance, etc.)
        freq: Frequency string (H, D, W, M, etc.)
        prediction_length: Forecast horizon
        context_length: Suggested context length
        num_variates: Number of variates (1 for univariate)
        seasonality: Seasonal period for metrics like MASE
        term: Prediction term (short, medium, long)
    """

    name: str
    domain: str
    freq: str
    prediction_length: int
    context_length: int = 512
    num_variates: int = 1
    seasonality: int = 1
    term: Literal["short", "medium", "long"] = "short"

    @property
    def config_name(self) -> str:
        """Unique configuration name for results CSV."""
        return f"{self.name}_{self.freq}_{self.term}"

    @property
    def hf_dataset_name(self) -> str:
        """HuggingFace dataset identifier."""
        return "Salesforce/GiftEval"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "freq": self.freq,
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "num_variates": self.num_variates,
            "seasonality": self.seasonality,
            "term": self.term,
        }


# Domain categories
DOMAINS = [
    "Energy",
    "Web",
    "Finance",
    "Weather",
    "Transport",
    "Manufacturing",
    "Sales",
]


# GIFT-Eval dataset configurations
# Based on GIFT-Eval benchmark: https://github.com/SalesforceAIResearch/gift-eval
GIFT_EVAL_DATASETS: list[DatasetConfig] = [
    # ============= Energy Domain =============
    DatasetConfig(
        name="electricity",
        domain="Energy",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=321,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="electricity",
        domain="Energy",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=321,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="solar_energy",
        domain="Energy",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=137,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="solar_energy",
        domain="Energy",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=137,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="australian_electricity",
        domain="Energy",
        freq="30T",
        prediction_length=48,
        context_length=336,
        num_variates=5,
        seasonality=48,
        term="short",
    ),
    DatasetConfig(
        name="australian_electricity",
        domain="Energy",
        freq="30T",
        prediction_length=336,
        context_length=672,
        num_variates=5,
        seasonality=48,
        term="long",
    ),
    # ============= Web/CloudOps Domain =============
    DatasetConfig(
        name="kdd_cup_2018",
        domain="Web",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=270,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="kdd_cup_2018",
        domain="Web",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=270,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="web_traffic",
        domain="Web",
        freq="D",
        prediction_length=7,
        context_length=90,
        num_variates=1,
        seasonality=7,
        term="short",
    ),
    DatasetConfig(
        name="web_traffic",
        domain="Web",
        freq="D",
        prediction_length=30,
        context_length=180,
        num_variates=1,
        seasonality=7,
        term="long",
    ),
    DatasetConfig(
        name="cloudops_2022",
        domain="Web",
        freq="5T",
        prediction_length=12,
        context_length=288,
        num_variates=100,
        seasonality=12,
        term="short",
    ),
    DatasetConfig(
        name="cloudops_2022",
        domain="Web",
        freq="5T",
        prediction_length=288,
        context_length=576,
        num_variates=100,
        seasonality=288,
        term="long",
    ),
    # ============= Finance Domain =============
    DatasetConfig(
        name="exchange_rate",
        domain="Finance",
        freq="D",
        prediction_length=7,
        context_length=60,
        num_variates=8,
        seasonality=7,
        term="short",
    ),
    DatasetConfig(
        name="exchange_rate",
        domain="Finance",
        freq="D",
        prediction_length=30,
        context_length=120,
        num_variates=8,
        seasonality=7,
        term="long",
    ),
    DatasetConfig(
        name="stock_market",
        domain="Finance",
        freq="D",
        prediction_length=5,
        context_length=60,
        num_variates=6,
        seasonality=5,
        term="short",
    ),
    DatasetConfig(
        name="stock_market",
        domain="Finance",
        freq="D",
        prediction_length=20,
        context_length=120,
        num_variates=6,
        seasonality=5,
        term="long",
    ),
    DatasetConfig(
        name="fred_md",
        domain="Finance",
        freq="M",
        prediction_length=3,
        context_length=36,
        num_variates=107,
        seasonality=12,
        term="short",
    ),
    DatasetConfig(
        name="fred_md",
        domain="Finance",
        freq="M",
        prediction_length=12,
        context_length=60,
        num_variates=107,
        seasonality=12,
        term="long",
    ),
    # ============= Weather Domain =============
    DatasetConfig(
        name="weather",
        domain="Weather",
        freq="10T",
        prediction_length=144,
        context_length=720,
        num_variates=21,
        seasonality=144,
        term="short",
    ),
    DatasetConfig(
        name="weather",
        domain="Weather",
        freq="10T",
        prediction_length=720,
        context_length=1440,
        num_variates=21,
        seasonality=144,
        term="long",
    ),
    DatasetConfig(
        name="temperature_rain",
        domain="Weather",
        freq="D",
        prediction_length=7,
        context_length=90,
        num_variates=32,
        seasonality=7,
        term="short",
    ),
    DatasetConfig(
        name="temperature_rain",
        domain="Weather",
        freq="D",
        prediction_length=30,
        context_length=180,
        num_variates=32,
        seasonality=7,
        term="long",
    ),
    DatasetConfig(
        name="wind_farms",
        domain="Weather",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=7,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="wind_farms",
        domain="Weather",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=7,
        seasonality=24,
        term="long",
    ),
    # ============= Transport Domain =============
    DatasetConfig(
        name="traffic",
        domain="Transport",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=862,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="traffic",
        domain="Transport",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=862,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="uber_tlc",
        domain="Transport",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=262,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="uber_tlc",
        domain="Transport",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=262,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="pedestrian_counts",
        domain="Transport",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=66,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="pedestrian_counts",
        domain="Transport",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=66,
        seasonality=24,
        term="long",
    ),
    # ============= Manufacturing Domain =============
    DatasetConfig(
        name="illness",
        domain="Manufacturing",
        freq="W",
        prediction_length=4,
        context_length=52,
        num_variates=7,
        seasonality=52,
        term="short",
    ),
    DatasetConfig(
        name="illness",
        domain="Manufacturing",
        freq="W",
        prediction_length=24,
        context_length=104,
        num_variates=7,
        seasonality=52,
        term="long",
    ),
    DatasetConfig(
        name="nn5",
        domain="Manufacturing",
        freq="D",
        prediction_length=7,
        context_length=56,
        num_variates=111,
        seasonality=7,
        term="short",
    ),
    DatasetConfig(
        name="nn5",
        domain="Manufacturing",
        freq="D",
        prediction_length=56,
        context_length=112,
        num_variates=111,
        seasonality=7,
        term="long",
    ),
    DatasetConfig(
        name="jena_weather",
        domain="Manufacturing",
        freq="10T",
        prediction_length=144,
        context_length=720,
        num_variates=14,
        seasonality=144,
        term="short",
    ),
    DatasetConfig(
        name="jena_weather",
        domain="Manufacturing",
        freq="10T",
        prediction_length=720,
        context_length=1440,
        num_variates=14,
        seasonality=144,
        term="long",
    ),
    # ============= Sales Domain =============
    DatasetConfig(
        name="m4_hourly",
        domain="Sales",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=1,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="m4_hourly",
        domain="Sales",
        freq="H",
        prediction_length=48,
        context_length=336,
        num_variates=1,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="m4_daily",
        domain="Sales",
        freq="D",
        prediction_length=7,
        context_length=60,
        num_variates=1,
        seasonality=7,
        term="short",
    ),
    DatasetConfig(
        name="m4_daily",
        domain="Sales",
        freq="D",
        prediction_length=14,
        context_length=120,
        num_variates=1,
        seasonality=7,
        term="long",
    ),
    DatasetConfig(
        name="m4_weekly",
        domain="Sales",
        freq="W",
        prediction_length=4,
        context_length=26,
        num_variates=1,
        seasonality=52,
        term="short",
    ),
    DatasetConfig(
        name="m4_weekly",
        domain="Sales",
        freq="W",
        prediction_length=13,
        context_length=52,
        num_variates=1,
        seasonality=52,
        term="long",
    ),
    DatasetConfig(
        name="m4_monthly",
        domain="Sales",
        freq="M",
        prediction_length=6,
        context_length=36,
        num_variates=1,
        seasonality=12,
        term="short",
    ),
    DatasetConfig(
        name="m4_monthly",
        domain="Sales",
        freq="M",
        prediction_length=18,
        context_length=60,
        num_variates=1,
        seasonality=12,
        term="long",
    ),
    DatasetConfig(
        name="m4_quarterly",
        domain="Sales",
        freq="Q",
        prediction_length=4,
        context_length=16,
        num_variates=1,
        seasonality=4,
        term="short",
    ),
    DatasetConfig(
        name="m4_quarterly",
        domain="Sales",
        freq="Q",
        prediction_length=8,
        context_length=24,
        num_variates=1,
        seasonality=4,
        term="long",
    ),
    DatasetConfig(
        name="m4_yearly",
        domain="Sales",
        freq="Y",
        prediction_length=2,
        context_length=10,
        num_variates=1,
        seasonality=1,
        term="short",
    ),
    DatasetConfig(
        name="m4_yearly",
        domain="Sales",
        freq="Y",
        prediction_length=6,
        context_length=20,
        num_variates=1,
        seasonality=1,
        term="long",
    ),
    DatasetConfig(
        name="m5",
        domain="Sales",
        freq="D",
        prediction_length=7,
        context_length=90,
        num_variates=1,
        seasonality=7,
        term="short",
    ),
    DatasetConfig(
        name="m5",
        domain="Sales",
        freq="D",
        prediction_length=28,
        context_length=180,
        num_variates=1,
        seasonality=7,
        term="long",
    ),
    DatasetConfig(
        name="tourism_monthly",
        domain="Sales",
        freq="M",
        prediction_length=3,
        context_length=24,
        num_variates=1,
        seasonality=12,
        term="short",
    ),
    DatasetConfig(
        name="tourism_monthly",
        domain="Sales",
        freq="M",
        prediction_length=24,
        context_length=48,
        num_variates=1,
        seasonality=12,
        term="long",
    ),
    DatasetConfig(
        name="tourism_quarterly",
        domain="Sales",
        freq="Q",
        prediction_length=2,
        context_length=12,
        num_variates=1,
        seasonality=4,
        term="short",
    ),
    DatasetConfig(
        name="tourism_quarterly",
        domain="Sales",
        freq="Q",
        prediction_length=8,
        context_length=20,
        num_variates=1,
        seasonality=4,
        term="long",
    ),
    # Additional datasets for full coverage
    DatasetConfig(
        name="etth1",
        domain="Energy",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=7,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="etth1",
        domain="Energy",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=7,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="etth2",
        domain="Energy",
        freq="H",
        prediction_length=24,
        context_length=168,
        num_variates=7,
        seasonality=24,
        term="short",
    ),
    DatasetConfig(
        name="etth2",
        domain="Energy",
        freq="H",
        prediction_length=168,
        context_length=336,
        num_variates=7,
        seasonality=24,
        term="long",
    ),
    DatasetConfig(
        name="ettm1",
        domain="Energy",
        freq="15T",
        prediction_length=96,
        context_length=336,
        num_variates=7,
        seasonality=96,
        term="short",
    ),
    DatasetConfig(
        name="ettm1",
        domain="Energy",
        freq="15T",
        prediction_length=336,
        context_length=720,
        num_variates=7,
        seasonality=96,
        term="long",
    ),
    DatasetConfig(
        name="ettm2",
        domain="Energy",
        freq="15T",
        prediction_length=96,
        context_length=336,
        num_variates=7,
        seasonality=96,
        term="short",
    ),
    DatasetConfig(
        name="ettm2",
        domain="Energy",
        freq="15T",
        prediction_length=336,
        context_length=720,
        num_variates=7,
        seasonality=96,
        term="long",
    ),
]


def get_datasets_by_domain(domain: str) -> list[DatasetConfig]:
    """Get all dataset configs for a specific domain.

    Args:
        domain: Domain name (Energy, Web, Finance, etc.)

    Returns:
        List of DatasetConfig for the domain
    """
    return [d for d in GIFT_EVAL_DATASETS if d.domain == domain]


def get_datasets_by_freq(freq: str) -> list[DatasetConfig]:
    """Get all dataset configs for a specific frequency.

    Args:
        freq: Frequency string (H, D, W, M, etc.)

    Returns:
        List of DatasetConfig for the frequency
    """
    return [d for d in GIFT_EVAL_DATASETS if d.freq == freq]


def get_datasets_by_term(term: str) -> list[DatasetConfig]:
    """Get all dataset configs for a specific term.

    Args:
        term: Term (short, medium, long)

    Returns:
        List of DatasetConfig for the term
    """
    return [d for d in GIFT_EVAL_DATASETS if d.term == term]


def get_config_by_name(name: str, term: str = "short") -> DatasetConfig | None:
    """Get dataset config by name and term.

    Args:
        name: Dataset name
        term: Term (short, medium, long)

    Returns:
        DatasetConfig or None if not found
    """
    for config in GIFT_EVAL_DATASETS:
        if config.name == name and config.term == term:
            return config
    return None
