# Metrics

TinyTimeMixers provides comprehensive evaluation metrics for time series forecasting.

## Point Forecast Metrics

### MSE (Mean Squared Error)

```python
MSE = mean((pred - target)^2)
```

Measures average squared difference. Sensitive to outliers.

### MAE (Mean Absolute Error)

```python
MAE = mean(|pred - target|)
```

Measures average absolute difference. More robust to outliers than MSE.

### RMSE (Root Mean Squared Error)

```python
RMSE = sqrt(MSE)
```

Same units as the target variable. Standard benchmark metric.

### MAPE (Mean Absolute Percentage Error)

```python
MAPE = mean(|pred - target| / |target|) * 100
```

Scale-independent percentage error. Undefined when target is zero.

### SMAPE (Symmetric MAPE)

```python
SMAPE = mean(2 * |pred - target| / (|pred| + |target|)) * 100
```

Symmetric version of MAPE. Bounded between 0% and 200%.

## Scale-Independent Metrics

### MASE (Mean Absolute Scaled Error)

```python
naive_mae = mean(|target[t] - target[t-seasonality]|)
MASE = MAE / naive_mae
```

Compares error against seasonal naive baseline.

| MASE Value | Interpretation |
|------------|----------------|
| < 1.0 | Better than naive |
| = 1.0 | Same as naive |
| > 1.0 | Worse than naive |

**Example:**
```python
from tinytimemixers.evaluation.metrics import MASE

mase = MASE(pred, target, context, seasonality=24)
```

## Probabilistic Metrics

### CRPS (Continuous Ranked Probability Score)

```python
CRPS = E[|X - target|] - 0.5 * E[|X - X'|]
```

Measures calibration of probabilistic forecasts. Lower is better.

**Example:**
```python
from tinytimemixers.evaluation.metrics import CRPS

# samples: (num_samples, prediction_length)
crps = CRPS(samples, target)
```

### Coverage

```python
coverage = mean(lower <= target <= upper)
```

Fraction of targets within prediction interval. Should match confidence level.

**Example:**
```python
from tinytimemixers.evaluation.metrics import coverage

# Check 90% interval coverage
cov = coverage(pred, target, lower_quantile=0.05, upper_quantile=0.95)
# Expected: ~0.90
```

### Interval Width

```python
width = mean(upper - lower)
```

Average width of prediction intervals. Narrower is better (given good coverage).

## Usage

### Individual Metrics

```python
from tinytimemixers.evaluation.metrics import MSE, MAE, RMSE, MAPE, SMAPE, MASE

# Point metrics
mse = MSE(pred, target)
mae = MAE(pred, target)
rmse = RMSE(pred, target)
mape = MAPE(pred, target)
smape = SMAPE(pred, target)

# Scale-independent
mase = MASE(pred, target, context, seasonality=24)
```

### Compute All Metrics

```python
from tinytimemixers.evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(
    pred=predictions,
    target=targets,
    context=context,
    seasonality=24,
)
# Returns: {'MSE': ..., 'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'SMAPE': ..., 'MASE': ...}
```

### MetricTracker

For accumulating metrics across batches:

```python
from tinytimemixers.evaluation.metrics import MetricTracker

tracker = MetricTracker(["MSE", "MAE", "MASE"])

for batch in dataloader:
    pred = model(batch.context)
    tracker.update(pred, batch.target, context=batch.context, seasonality=24)

final_metrics = tracker.compute()
```

## GIFT-Eval Metrics

The GIFT-Eval benchmark uses these specific metrics:

| Column Name | Metric |
|-------------|--------|
| eval_metrics/MSE[mean] | Mean MSE |
| eval_metrics/MSE[0.5] | Median MSE |
| eval_metrics/MAE[0.5] | Median MAE |
| eval_metrics/MASE[0.5] | Median MASE |
| eval_metrics/MAPE[0.5] | Median MAPE |
| eval_metrics/sMAPE[0.5] | Median SMAPE |
| eval_metrics/MSIS | Mean Scaled Interval Score |
| eval_metrics/RMSE[mean] | Mean RMSE |
| eval_metrics/NRMSE[mean] | Normalized RMSE |
| eval_metrics/ND[0.5] | Normalized Deviation |
| eval_metrics/mean_weighted_sum_quantile_loss | Quantile Loss |

## Recommendations

| Use Case | Recommended Metric |
|----------|-------------------|
| General benchmarking | MASE, RMSE |
| Business forecasting | MAPE, SMAPE |
| Scale comparison | MASE |
| Probabilistic evaluation | CRPS |
| Interval forecasting | Coverage + Width |
