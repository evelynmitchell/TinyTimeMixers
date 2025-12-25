# Benchmarking Guide

This guide covers how to run TinyTimeMixers on the GIFT-Eval benchmark and compare with TabPFN-TS.

## Overview

The benchmarking module provides:

- **GIFT-Eval Integration**: Run TTM on 98 dataset configurations across 7 domains
- **TabPFN-TS Comparison**: Head-to-head comparison with statistical significance testing
- **GluonTS Compatibility**: Adapters for GluonTS evaluation framework
- **Checkpointing**: Resume interrupted benchmark runs

## Installation

Install benchmark dependencies:

```bash
# Core benchmark dependencies
uv sync --group benchmark

# Optional: TabPFN-TS for comparison (requires API key)
uv sync --group tabpfn
```

## Quick Start

### Running GIFT-Eval Benchmark

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

### Comparing with TabPFN-TS

```bash
# Full comparison
python scripts/compare_tabpfn.py --ttm-path models/ttm.pt

# Compare on subset of datasets
python scripts/compare_tabpfn.py --ttm-path models/ttm.pt --datasets 10

# Compare using specific metric
python scripts/compare_tabpfn.py --ttm-path models/ttm.pt --metric MASE
```

## GIFT-Eval Datasets

The benchmark covers 7 domains:

| Domain | Datasets | Description |
|--------|----------|-------------|
| Energy | electricity, solar, ETT | Power consumption, solar generation |
| Web | kdd_cup, web_traffic, cloudops | Web traffic, cloud metrics |
| Finance | exchange_rate, stock_market, fred_md | Financial time series |
| Weather | weather, temperature_rain, wind_farms | Meteorological data |
| Transport | traffic, uber_tlc, pedestrian | Transportation flows |
| Manufacturing | illness, nn5, jena_weather | Industrial metrics |
| Sales | m4, m5, tourism | Retail and tourism forecasting |

Each dataset has short-term and long-term prediction configurations.

## Output Format

### Results CSV

The benchmark outputs `all_results.csv` in GIFT-Eval format:

```
dataset,model,domain,num_variates,eval_metrics/MSE[mean],...
electricity_H_short,TTM,Energy,321,0.0123,...
```

### Config JSON

Model metadata for leaderboard submission:

```json
{
  "model": "TTM",
  "model_type": "pretrained",
  "num_parameters": 2200000,
  "context_length": 512,
  "testdata_leakage": "No"
}
```

## Python API

### Using the GluonTS Adapter

```python
from benchmarks.gift_eval.adapter import TTMZeroShotPredictor
from tinytimemixers.models.ttm import TTM

# Load model
model = TTM.load("models/ttm.pt")

# Create GluonTS-compatible predictor
predictor = TTMZeroShotPredictor(
    model=model,
    prediction_length=24,
    freq="H",
    context_length=512,
    num_samples=100,
)

# Generate forecasts
for forecast in predictor.predict(test_dataset):
    print(forecast.median)  # Point forecast
    print(forecast.samples)  # Probabilistic samples
```

### Running Benchmarks Programmatically

```python
from benchmarks.gift_eval.runner import GIFTEvalRunner
from benchmarks.gift_eval.results import ModelMetadata
from benchmarks.gift_eval.adapter import load_predictor_from_path

def predictor_factory(config):
    return load_predictor_from_path(
        model_path="models/ttm.pt",
        prediction_length=config.prediction_length,
        freq=config.freq,
        predictor_type="zero_shot",
    )

runner = GIFTEvalRunner(
    predictor_factory=predictor_factory,
    output_dir="results/ttm",
    resume=True,
)

# Run all or filtered datasets
results = runner.run_all(domains=["Energy", "Web"])

# Save in GIFT-Eval format
metadata = ModelMetadata(model_name="TTM", num_parameters=2200000)
runner.save_results(metadata)
```

### Model Comparison

```python
from benchmarks.tabpfn_comparison.compare import ModelComparator
from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader

comparator = ModelComparator(
    ttm_predictor_factory=ttm_factory,
    tabpfn_predictor_factory=tabpfn_factory,
    primary_metric="MASE",
)

loader = GIFTEvalDatasetLoader()
results = comparator.compare_all(configs, loader)

# Get statistics
stats = comparator.compute_statistics()
print(f"TTM wins: {stats['ttm_wins']}")
print(f"TabPFN wins: {stats['tabpfn_wins']}")

# Statistical significance
sig = comparator.compute_statistical_significance("MSE")
print(f"p-value: {sig['paired_ttest_pvalue']}")
```

## CLI Options

### benchmark_gift.py

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | required | Path to TTM model |
| `--output` | results/ttm | Output directory |
| `--domains` | all | Filter by domains |
| `--device` | auto | Device for inference |
| `--context-length` | 512 | Context length |
| `--num-samples` | 100 | Probabilistic samples |
| `--batch-size` | 64 | Batch size |
| `--checkpoint-interval` | 10 | Save every N datasets |
| `--resume` | false | Resume from checkpoint |

### compare_tabpfn.py

| Option | Default | Description |
|--------|---------|-------------|
| `--ttm-path` | required | Path to TTM model |
| `--output` | results/comparison | Output directory |
| `--datasets` | all | Number of datasets |
| `--domains` | all | Filter by domains |
| `--metric` | MSE | Primary comparison metric |

## Troubleshooting

### Missing GluonTS

```
ImportError: GluonTS is required for benchmark adapters
```

Solution: `uv sync --group benchmark`

### TabPFN Not Available

```
TabPFN-TS is not installed
```

Solution: `uv sync --group tabpfn`

### Checkpoint Corruption

If checkpoint is corrupted, clear and restart:

```python
runner.clear_checkpoint()
```

Or delete `results/ttm/checkpoint.json` manually.
