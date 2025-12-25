# TinyTimeMixers Documentation

TinyTimeMixers (TTM) is a lightweight time series foundation model implementation in PyTorch, based on the paper [arXiv 2401.03955](https://arxiv.org/abs/2401.03955).

## Overview

TTM is designed for efficient time series forecasting with:

- **~2.2M parameters** - Lightweight yet powerful
- **Zero-shot capability** - Works on unseen datasets without fine-tuning
- **Few-shot adaptation** - Quick adaptation with minimal data
- **Multi-resolution support** - Handles various frequencies and horizons

## Quick Links

- [Architecture](architecture.md) - Model architecture and components
- [Metrics](metric.md) - Evaluation metrics
- [Benchmarking](benchmarking.md) - GIFT-Eval and TabPFN-TS comparison
- [Examples](examples/index.md) - Code examples

## Installation

```bash
# Clone repository
git clone https://github.com/evelynmitchell/TinyTimeMixers
cd TinyTimeMixers

# Install with uv
uv sync --all-groups
```

## Quick Start

```python
from tinytimemixers import TTM, TTMConfig
import torch

# Create model
config = TTMConfig(context_length=512, prediction_length=96)
model = TTM(config)

# Make prediction
context = torch.randn(1, 1, 512)
forecast = model(context)
```

## Key Features

### TSMixer Architecture

The model uses a hierarchical TSMixer backbone with:

- Multi-level processing (L=6 levels)
- Time, feature, and channel mixing at each level
- Adaptive patching for multi-resolution

### RevIN Normalization

Reversible instance normalization enables:

- Domain-agnostic inference
- Better zero-shot generalization
- Stable training dynamics

### Two-Stage Training

1. **Pre-training**: Full model on large-scale time series
2. **Fine-tuning**: Frozen backbone, trainable decoder/head

## Project Structure

```
tinytimemixers/
├── config.py           # Configuration dataclasses
├── layers/             # Neural network layers
├── models/             # TTM model components
├── training/           # Training infrastructure
├── evaluation/         # Metrics and forecasters
├── data/               # Data loading and preprocessing
└── utils/              # Utilities

benchmarks/
├── gift_eval/          # GIFT-Eval integration
└── tabpfn_comparison/  # TabPFN-TS comparison
```

## References

- [TTM Paper](https://arxiv.org/abs/2401.03955) - TinyTimeMixers: Tiny Foundation Models for Time Series
- [TSMixer Paper](https://arxiv.org/abs/2303.06053) - TSMixer: An All-MLP Architecture for Time Series Forecasting
- [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) - Benchmark suite
