"""TinyTimeMixers - Lightweight Time Series Foundation Model.

A compact pre-trained model for multivariate time series forecasting
based on the TSMixer architecture with adaptive patching.

Reference: arXiv 2401.03955
"""

from tinytimemixers.config import TrainingConfig, TTMConfig
from tinytimemixers.models.ttm import TTM, TTMForFinetune, TTMForPretrain

__version__ = "0.1.0"
__all__ = [
    "TTM",
    "TTMConfig",
    "TrainingConfig",
    "TTMForPretrain",
    "TTMForFinetune",
]
