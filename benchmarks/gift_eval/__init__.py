"""GIFT-Eval benchmark integration for TinyTimeMixers."""

from benchmarks.gift_eval.adapter import (
    TTMFewShotPredictor,
    TTMGluonTSPredictor,
    TTMZeroShotPredictor,
)
from benchmarks.gift_eval.config import GIFT_EVAL_DATASETS, DatasetConfig
from benchmarks.gift_eval.results import ModelMetadata, ResultsAggregator
from benchmarks.gift_eval.runner import BenchmarkResult, GIFTEvalRunner

__all__ = [
    "TTMGluonTSPredictor",
    "TTMZeroShotPredictor",
    "TTMFewShotPredictor",
    "DatasetConfig",
    "GIFT_EVAL_DATASETS",
    "GIFTEvalRunner",
    "BenchmarkResult",
    "ResultsAggregator",
    "ModelMetadata",
]
