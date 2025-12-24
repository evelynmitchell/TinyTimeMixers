"""TTM Configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TTMConfig:
    """Configuration for TinyTimeMixers model.

    Based on arXiv 2401.03955 specifications.
    Default values target ~1M parameters.
    """

    # Sequence lengths
    context_length: int = 512
    prediction_length: int = 96

    # Patching
    patch_length: int = 64
    patch_stride: int = 64  # Non-overlapping by default

    # Backbone architecture
    num_backbone_levels: int = 6  # L=6 in paper
    blocks_per_level: int = 2  # M=2 in paper

    # Decoder architecture
    num_decoder_layers: int = 2

    # Feature dimensions
    feature_scaler: int = 3  # fs=3 in paper
    # hidden_features = feature_scaler * patch_length = 192
    # expansion_features = hidden_features * 2 = 384

    # Regularization
    dropout: float = 0.2
    head_dropout: float = 0.7  # Higher for small datasets

    # Resolution prefix tuning
    use_resolution_prefix: bool = False  # Disabled for now, TODO: fix prefix handling
    num_resolutions: int = 10  # Discrete resolution buckets

    # Channel handling
    channel_independent_backbone: bool = True
    channel_mixing_decoder: bool = False

    # Exogenous variables
    use_exogenous_mixer: bool = False

    # Training
    loss_type: Literal["mse", "mae"] = "mse"

    # Computed properties
    @property
    def hidden_features(self) -> int:
        """Hidden feature dimension (hf = fs * pl)."""
        return self.feature_scaler * self.patch_length

    @property
    def expansion_features(self) -> int:
        """Expansion feature dimension for MLPs (ef = hf * 2)."""
        return self.hidden_features * 2

    @property
    def num_patches(self) -> int:
        """Number of patches from context length."""
        return self.context_length // self.patch_stride

    def __post_init__(self):
        """Validate configuration."""
        if self.context_length % self.patch_stride != 0:
            raise ValueError(
                f"context_length ({self.context_length}) must be "
                f"divisible by patch_stride ({self.patch_stride})"
            )

        if self.patch_length > self.context_length:
            raise ValueError(
                f"patch_length ({self.patch_length}) must be <= "
                f"context_length ({self.context_length})"
            )


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    # Training loop
    batch_size: int = 64
    num_epochs: int = 100
    warmup_steps: int = 1000

    # Checkpointing
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "./checkpoints"

    # Fine-tuning
    freeze_backbone: bool = False
    finetune_head_dropout: float = 0.7

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
