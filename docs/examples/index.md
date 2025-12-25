# Examples

This section provides practical examples for using TinyTimeMixers.

## Basic Forecasting

### Create and Run Model

```python
import torch
from tinytimemixers import TTM, TTMConfig

# Create model with default config
config = TTMConfig(
    context_length=512,
    prediction_length=96,
)
model = TTM(config, num_channels=1)

# Generate sample input
batch_size = 4
context = torch.randn(batch_size, 1, 512)

# Make prediction
with torch.no_grad():
    forecast = model(context)

print(f"Input shape: {context.shape}")
print(f"Output shape: {forecast.shape}")
# Input shape: torch.Size([4, 1, 512])
# Output shape: torch.Size([4, 1, 96])
```

### Save and Load Model

```python
# Save model
model.save("models/ttm.pt")

# Load model
loaded_model = TTM.load("models/ttm.pt")
```

## Training

### Basic Training Loop

```python
from tinytimemixers.training import Trainer, TrainingConfig
from tinytimemixers.data import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np

# Create synthetic data
data = np.random.randn(100, 1, 1000).astype(np.float32)

# Create datasets
train_dataset = TimeSeriesDataset(
    data[:80],
    context_length=512,
    prediction_length=96,
)
val_dataset = TimeSeriesDataset(
    data[80:],
    context_length=512,
    prediction_length=96,
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Create model and trainer
model = TTM(TTMConfig(), num_channels=1)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    training_config=TrainingConfig(
        learning_rate=1e-4,
        num_epochs=10,
    ),
)

# Train
metrics = trainer.train()
print(f"Final train loss: {metrics.train_loss:.4f}")
print(f"Final val loss: {metrics.val_loss:.4f}")
```

## Evaluation

### Zero-Shot Evaluation

```python
from tinytimemixers.evaluation import ZeroShotForecaster
from tinytimemixers.data import TimeSeriesDataset

# Load pre-trained model
model = TTM.load("models/ttm.pt")

# Create forecaster with instance normalization
forecaster = ZeroShotForecaster(model, normalize=True)

# Evaluate on dataset
test_data = np.random.randn(20, 1, 700).astype(np.float32)
dataset = TimeSeriesDataset(test_data, context_length=512, prediction_length=96)

metrics = forecaster.evaluate(dataset, seasonality=24)
print(f"MSE: {metrics['MSE']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
print(f"MASE: {metrics['MASE']:.4f}")
```

### Few-Shot Adaptation

```python
from tinytimemixers.evaluation import FewShotForecaster

# Create few-shot forecaster
forecaster = FewShotForecaster(model)

# Adapt to new dataset
train_data = torch.randn(10, 1, 700)
forecaster.adapt(
    train_data=train_data,
    context_length=512,
    prediction_length=96,
    num_epochs=5,
    learning_rate=1e-3,
)

# Evaluate
metrics = forecaster.evaluate(test_dataset, seasonality=24)

# Reset to original weights
forecaster.reset()
```

## Benchmarking

### Run GIFT-Eval

```python
from benchmarks.gift_eval.runner import run_gift_eval

# Run benchmark on all datasets
results = run_gift_eval(
    model_path="models/ttm.pt",
    output_dir="results/ttm",
    device="cuda",
)

# Get summary
summary = results.compute_summary_statistics()
print(f"Datasets evaluated: {summary['num_datasets']}")
print(f"Mean MSE: {summary['mean_eval_metrics/MSE[mean]']:.4f}")
```

### Compare with TabPFN

```python
from benchmarks.tabpfn_comparison.compare import ModelComparator
from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader
from benchmarks.gift_eval.config import GIFT_EVAL_DATASETS

# Setup comparison
comparator = ModelComparator(
    ttm_predictor_factory=ttm_factory,
    tabpfn_predictor_factory=tabpfn_factory,
)

loader = GIFTEvalDatasetLoader()
configs = GIFT_EVAL_DATASETS[:10]  # First 10 datasets

# Run comparison
comparator.compare_all(configs, loader)

# Get results
stats = comparator.compute_statistics()
print(f"TTM wins: {stats['ttm_wins']}")
print(f"TabPFN wins: {stats['tabpfn_wins']}")
```

## Data Loading

### Load from Monash Repository

```python
from tinytimemixers.data import MonashLoader

loader = MonashLoader()

# List available datasets
datasets = loader.list_datasets()
print(datasets)

# Load specific dataset
data = loader.load("electricity_hourly")
print(f"Shape: {data.shape}")
```

### Create Custom Dataset

```python
from tinytimemixers.data import TimeSeriesDataset
import torch

# Your time series data: (num_series, channels, seq_length)
data = torch.randn(50, 1, 1000)

dataset = TimeSeriesDataset(
    data,
    context_length=512,
    prediction_length=96,
    stride=48,  # Overlapping windows
)

print(f"Number of samples: {len(dataset)}")

# Get a sample
context, target = dataset[0]
print(f"Context shape: {context.shape}")  # (1, 512)
print(f"Target shape: {target.shape}")    # (1, 96)
```
