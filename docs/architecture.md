# Architecture

TinyTimeMixers uses a hierarchical TSMixer architecture optimized for time series forecasting.

## Model Overview

```
Input: (batch, channels, context_length)
         │
         ▼
┌─────────────────────┐
│   RevIN Normalize   │  Reversible instance normalization
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Patch Embedding   │  (context_length) → (num_patches, hidden_features)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     Backbone        │  L=6 levels, M=2 TSMixer blocks per level
│   (TTMBackbone)     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│      Decoder        │  2 layers
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Forecast Head     │  Linear projection
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  RevIN Denormalize  │
└─────────────────────┘
         │
         ▼
Output: (batch, channels, prediction_length)
```

## Components

### RevIN (Reversible Instance Normalization)

Normalizes input to zero mean and unit variance, then denormalizes output.

```python
# Normalize
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + eps)

# Denormalize (after model)
x_out = x_pred * std + mean
```

**Benefits:**
- Domain-agnostic inference
- Better zero-shot generalization
- Handles varying scales across datasets

### Patch Embedding

Converts time series into patches for efficient processing.

| Parameter | Value |
|-----------|-------|
| Patch Length | 64 |
| Patch Stride | 64 (non-overlapping) |
| Hidden Features | 192 (3 × patch_length) |

```python
# Input: (batch, channels, 512)
# After patching: (batch, channels, 8, 64)  # 8 patches
# After embedding: (batch, channels, 8, 192)
```

### TSMixer Block

Each block contains three mixing operations:

```
Input
  │
  ├──► Time Mixing MLP ──► LayerNorm ──► Dropout
  │         │
  │    (mix across time dimension)
  │         │
  ├──► Feature Mixing MLP ──► LayerNorm ──► Dropout
  │         │
  │    (mix across feature dimension)
  │         │
  └──► Channel Mixing MLP ──► LayerNorm ──► Dropout
            │
       (mix across channels)
            │
         Output
```

**MLP Structure:**
```python
MLP(
    Linear(in_features, expansion * in_features),
    GELU(),
    Dropout(dropout),
    Linear(expansion * in_features, out_features),
    Dropout(dropout),
)
```

### Backbone

Multi-level architecture with adaptive patching:

| Level | Patches | TSMixer Blocks |
|-------|---------|----------------|
| 1 | 8 | 2 |
| 2 | 8 | 2 |
| 3 | 8 | 2 |
| 4 | 8 | 2 |
| 5 | 8 | 2 |
| 6 | 8 | 2 |

**Total: 12 TSMixer blocks**

### Decoder

Lightweight decoder for task-specific processing:

- 2 TSMixer blocks
- Same structure as backbone blocks
- 10-20% of backbone parameters

### Forecast Head

Linear projection to prediction length:

```python
# Flatten: (batch, channels, num_patches, hidden) → (batch, channels, num_patches * hidden)
# Project: (batch, channels, num_patches * hidden) → (batch, channels, prediction_length)
```

## Configuration

```python
from tinytimemixers import TTMConfig

config = TTMConfig(
    # Input/Output
    context_length=512,
    prediction_length=96,

    # Patching
    patch_length=64,
    patch_stride=64,

    # Architecture
    num_backbone_levels=6,
    blocks_per_level=2,
    decoder_layers=2,

    # Features
    hidden_features=192,
    expansion_factor=2,

    # Regularization
    dropout=0.2,
    head_dropout=0.7,

    # Normalization
    use_revin=True,
)
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Patch Embedding | ~12K |
| Backbone (6 levels) | ~1.8M |
| Decoder | ~300K |
| Forecast Head | ~150K |
| **Total** | **~2.2M** |

## Variants

### TTMForPretrain

Channel-independent pre-training:

```python
model = TTMForPretrain(config)
# Forces num_channels=1 for channel-independent learning
```

### TTMForFinetune

Fine-tuning with frozen backbone:

```python
model = TTMForFinetune(config, pretrained_path="model.pt")
# Backbone frozen, only decoder/head trainable
```
