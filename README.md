# TinyTimeMixers

A lightweight time series foundation model implementation in PyTorch, based on the TTM paper ([arXiv 2401.03955](https://arxiv.org/abs/2401.03955)).

## Features

- **TSMixer Architecture**: Multi-level backbone with time, feature, and channel mixing
- **Adaptive Patching**: Resolution-aware patch extraction
- **RevIN Normalization**: Reversible instance normalization for domain adaptation
- **Zero-Shot & Few-Shot**: Pre-trained model with optional head fine-tuning
- **GIFT-Eval Benchmark**: Integration with 98 dataset configurations across 7 domains
- **TabPFN-TS Comparison**: Head-to-head comparison with statistical testing

## Installation

```bash
# Clone repository
git clone https://github.com/evelynmitchell/TinyTimeMixers
cd TinyTimeMixers

# Install with uv (recommended)
uv sync --all-groups

# Or with pip
pip install -e .
```

### Optional Dependencies

```bash
# Benchmark dependencies (GIFT-Eval)
uv sync --group benchmark

# TabPFN comparison (requires API key)
uv sync --group tabpfn
```

## Quick Start

### Basic Forecasting

```python
from tinytimemixers import TTM, TTMConfig
import torch

# Create model
config = TTMConfig(
    context_length=512,
    prediction_length=96,
    num_channels=1,
)
model = TTM(config)

# Make prediction
context = torch.randn(1, 1, 512)  # (batch, channels, context_length)
forecast = model(context)  # (batch, channels, prediction_length)
```

### Zero-Shot Evaluation

```python
from tinytimemixers.evaluation import ZeroShotForecaster
from tinytimemixers.data import TimeSeriesDataset

# Load pre-trained model
model = TTM.load("models/ttm.pt")
forecaster = ZeroShotForecaster(model)

# Evaluate on dataset
dataset = TimeSeriesDataset(data, context_length=512, prediction_length=96)
metrics = forecaster.evaluate(dataset, seasonality=24)
print(f"MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}")
```

### Training

```python
from tinytimemixers.training import Trainer, TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    num_epochs=100,
    batch_size=64,
)

trainer = Trainer(model, train_loader, val_loader, config)
metrics = trainer.train()
```

## Benchmarking

### Run GIFT-Eval Benchmark

```bash
# Run on all datasets
python scripts/benchmark_gift.py --model-path models/ttm.pt

# Run on specific domains
python scripts/benchmark_gift.py --model-path models/ttm.pt --domains Energy Web

# Resume interrupted run
python scripts/benchmark_gift.py --model-path models/ttm.pt --resume

# List available datasets
python scripts/benchmark_gift.py --list-datasets
```

### Compare with TabPFN-TS

```bash
python scripts/compare_tabpfn.py --ttm-path models/ttm.pt
```

See [docs/benchmarking.md](docs/benchmarking.md) for detailed benchmarking guide.

## Model Architecture

```
TTM
├── RevIN (Reversible Instance Normalization)
├── Patch Embedding (64 → 192 features)
├── Backbone (L=6 levels)
│   └── TSMixer Blocks (M=2 per level)
│       ├── Time Mixing MLP
│       ├── Feature Mixing MLP
│       └── Channel Mixing MLP
├── Decoder (2 layers)
└── Forecast Head (Linear projection)
```

**Model Stats:**
- Parameters: ~2.2M
- Context Length: 512
- Prediction Length: 96 (configurable)

## Project Structure

```
tinytimemixers/
├── config.py           # TTMConfig, TrainingConfig
├── layers/             # RevIN, TSMixer blocks, patching
├── models/             # TTM, backbone, decoder, forecast head
├── training/           # Trainer, losses, optimizers
├── evaluation/         # Metrics, forecasters
├── data/               # Dataset, preprocessing, augmentation
└── utils/              # Checkpointing, device management

benchmarks/
├── gift_eval/          # GIFT-Eval benchmark integration
└── tabpfn_comparison/  # TabPFN-TS comparison

scripts/
├── benchmark_gift.py   # Run GIFT-Eval benchmark
└── compare_tabpfn.py   # Compare with TabPFN-TS
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suite
uv run pytest tests/unit/test_ttm.py -v
uv run pytest tests/unit/test_benchmarks.py -v
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run linting
ruff check .

# Run formatting
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## References

- [TTM Paper (arXiv 2401.03955)](https://arxiv.org/abs/2401.03955)
- [TSMixer Paper (arXiv 2303.06053)](https://arxiv.org/abs/2303.06053)
- [GIFT-Eval Benchmark](https://github.com/SalesforceAIResearch/gift-eval)
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series)

## License

MIT
