# ADR-001: TinyTimeMixers Implementation

**Status:** Proposed
**Date:** 2024-12-24
**Authors:** Claude Code session

## Context

We need to implement TinyTimeMixers (TTM) from paper [arXiv 2401.03955](https://arxiv.org/abs/2401.03955), a lightweight time series foundation model accepted at NeurIPS 2024. The goal is to:

1. Build TTM from scratch in PyTorch
2. Benchmark on GIFT-Eval (97 tasks, 7 domains)
3. Compare against TabPFN-TS

**Constraints:**
- ~1M parameters (as per paper)
- CPU-capable with optional GPU acceleration
- Compatible with GluonTS evaluation framework

## Decision

### Architecture

Implement the full TTM architecture following the paper specifications:

| Component | Specification |
|-----------|--------------|
| Backbone | L=6 levels, M=2 TSMixer blocks per level |
| Decoder | 2 layers (10-20% of backbone size) |
| Context Length | 512 timesteps |
| Patch Length | 64 (non-overlapping) |
| Hidden Features | 192 (= 3 × patch_length) |
| Dropout | 0.2 (model), 0.7 (head for small datasets) |

**Key Innovations to Implement:**
- **Adaptive Patching:** Ki = 2^(L-i) reshaping per level
- **Resolution Prefix Tuning:** Learnable embeddings for multi-resolution
- **RevIN:** Reversible instance normalization
- **Two-stage Training:** Pre-train full model, fine-tune with frozen backbone

### File Structure

```
tinytimemixers/
├── config.py                    # TTMConfig dataclass
├── models/
│   ├── ttm.py                   # Main model
│   ├── backbone.py              # L=6 level backbone
│   ├── decoder.py               # Fine-tunable decoder
│   ├── forecast_head.py         # Linear projection
│   └── exogenous_mixer.py       # Exogenous variable handling
├── layers/
│   ├── tsmixer_block.py         # Core TSMixer block
│   ├── patch_embedding.py       # Patching + projection
│   ├── patch_partition.py       # Adaptive reshape
│   ├── resolution_prefix.py     # Resolution embeddings
│   ├── mixer_mlp.py             # Time/Feature/Channel MLPs
│   └── normalization.py         # RevIN implementation
├── data/
│   ├── dataset.py               # PyTorch Dataset
│   ├── preprocessing.py         # Normalization, windowing
│   ├── augmentation.py          # Downsampling
│   └── monash_loader.py         # HuggingFace Monash
├── training/
│   ├── trainer.py               # Training loop
│   ├── losses.py                # MSE, MAE
│   └── optimizer.py             # AdamW factory
├── evaluation/
│   ├── metrics.py               # MSE, MAE, MASE, CRPS
│   └── forecaster.py            # Zero/few-shot evaluation
└── utils/
    ├── device.py                # CPU/GPU management
    ├── checkpoint.py            # Save/load
    └── logging.py               # Progress tracking

benchmarks/
├── gift_eval/
│   ├── adapter.py               # GluonTS interface
│   ├── runner.py                # 97-task runner
│   └── results.py               # Aggregation
└── tabpfn_comparison/
    ├── wrapper.py               # TabPFN-TS wrapper
    └── compare.py               # Head-to-head

scripts/
├── pretrain.py
├── finetune.py
├── evaluate.py
└── benchmark_gift.py
```

### Dependencies

```toml
[dependency-groups]
core = [
    "torch>=2.1.0",
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "einops>=0.7.0",
    "tqdm>=4.66.0",
    "pydantic>=2.5.0",
]
data = [
    "datasets>=2.16.0",
    "gluonts>=0.14.0",
]
benchmark = [
    "tabpfn-time-series>=1.0.0",
    "scipy>=1.11.0",
]
```

### Implementation Phases

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Core Architecture | Layers, models, forward pass |
| 2 | Data Pipeline | Dataset, preprocessing, Monash loader |
| 3 | Training | Trainer, losses, checkpointing |
| 4 | Evaluation | Metrics, forecaster |
| 5 | Benchmarking | GIFT-Eval, TabPFN-TS comparison |

### Testing Strategy

- **Unit tests:** Individual layer shapes and behaviors
- **Integration tests:** End-to-end forward pass, gradient flow
- **Performance tests:** <1M params verification, CPU inference speed

## Alternatives Considered

### 1. Use IBM's Pre-trained Weights
**Rejected:** User explicitly chose full implementation for deeper understanding.

### 2. Minimal Implementation for Benchmarking Only
**Rejected:** User wants the complete architecture, not just inference wrappers.

### 3. TensorFlow/JAX Implementation
**Rejected:** PyTorch is the de facto standard for research and matches the original TSMixer codebase.

## Consequences

**Positive:**
- Full control over architecture for experimentation
- Deep understanding of TTM internals
- Clean, modular codebase for future extensions
- Direct comparison with TabPFN-TS on same benchmark

**Negative:**
- Longer implementation timeline vs. using pre-trained weights
- Risk of implementation bugs affecting benchmark results
- Need to validate against paper's reported numbers

**Risks & Mitigations:**
| Risk | Mitigation |
|------|------------|
| Parameter count exceeds 1M | Unit test enforcing <1M params |
| Architecture mismatch vs paper | Cross-reference IBM's implementation |
| GIFT-Eval integration issues | Use established GluonTS adapter pattern |

## References

1. [TTM Paper (arXiv 2401.03955)](https://arxiv.org/abs/2401.03955)
2. [TSMixer Paper (arXiv 2303.06053)](https://arxiv.org/abs/2303.06053)
3. [GIFT-Eval Benchmark](https://github.com/SalesforceAIResearch/gift-eval)
4. [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series)
5. [IBM Granite TSFM](https://github.com/ibm-granite/granite-tsfm)
6. [Monash Time Series Repository](https://huggingface.co/datasets/Monash-University/monash_tsf)
