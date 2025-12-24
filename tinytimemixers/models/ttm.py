"""Main TinyTimeMixers model."""

import torch
import torch.nn as nn

from tinytimemixers.config import TTMConfig
from tinytimemixers.layers.normalization import RevIN
from tinytimemixers.layers.patch_embedding import PatchEmbedding
from tinytimemixers.layers.resolution_prefix import ResolutionPrefix
from tinytimemixers.models.backbone import TTMBackboneLight
from tinytimemixers.models.decoder import TTMDecoder
from tinytimemixers.models.forecast_head import ForecastHead


class TTM(nn.Module):
    """TinyTimeMixers - Lightweight Time Series Foundation Model.

    A compact pre-trained model for multivariate time series forecasting
    based on the TSMixer architecture with adaptive patching.

    Key features:
    - ~1M parameters for efficient CPU inference
    - Multi-level backbone with adaptive patching
    - RevIN for scale normalization
    - Resolution prefix tuning for multi-resolution data
    - Frozen backbone fine-tuning for transfer learning

    Reference: arXiv 2401.03955

    Architecture:
        Input (batch, channels, context_length)
            -> RevIN normalization
            -> Patch embedding
            -> Resolution prefix (optional)
            -> Backbone (L levels, M blocks each)
            -> Decoder (fine-tunable)
            -> Forecast head
            -> RevIN denormalization
        Output (batch, channels, prediction_length)
    """

    def __init__(
        self,
        config: TTMConfig | None = None,
        num_channels: int = 1,
    ):
        """Initialize TTM.

        Args:
            config: Model configuration (uses defaults if None)
            num_channels: Number of input channels
        """
        super().__init__()
        self.config = config or TTMConfig()
        self.num_channels = num_channels

        # Normalization
        self.revin = RevIN(eps=1e-5, affine=False)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_length=self.config.patch_length,
            patch_stride=self.config.patch_stride,
            hidden_features=self.config.hidden_features,
        )

        # Resolution prefix (optional)
        self.use_resolution_prefix = self.config.use_resolution_prefix
        if self.use_resolution_prefix:
            self.resolution_prefix = ResolutionPrefix(
                hidden_features=self.config.hidden_features,
                num_resolutions=self.config.num_resolutions,
            )
            # Adjust num_patches for prefix
            self._num_patches_with_prefix = self.config.num_patches + 1
        else:
            self._num_patches_with_prefix = self.config.num_patches

        # Backbone (use light version without adaptive patching for now)
        # TODO: Fix adaptive patching for full backbone
        self.backbone = TTMBackboneLight(self.config)

        # Decoder
        self.decoder = TTMDecoder(
            config=self.config,
            num_channels=num_channels if num_channels > 1 else None,
        )

        # Forecast head
        self.forecast_head = ForecastHead(config=self.config)

    def forward(
        self,
        x: torch.Tensor,
        resolution_idx: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Generate forecast.

        Args:
            x: Input time series (batch, channels, context_length)
            resolution_idx: Resolution bucket index for prefix tuning

        Returns:
            Forecast (batch, channels, prediction_length)
        """
        # Normalize input
        x, stats = self.revin(x, mode="normalize")

        # Extract and embed patches
        # (batch, channels, num_patches, hidden_features)
        x = self.patch_embedding(x)

        # Add resolution prefix if enabled
        if self.use_resolution_prefix:
            x = self.resolution_prefix(x, resolution_idx)

        # Apply backbone
        x = self.backbone(x)

        # Remove resolution prefix if it was added
        if self.use_resolution_prefix:
            x = self.resolution_prefix.remove_prefix(x)

        # Apply decoder
        x = self.decoder(x)

        # Generate forecast
        # (batch, channels, prediction_length)
        forecast = self.forecast_head(x)

        # Denormalize output
        forecast = self.revin(forecast, mode="denormalize", stats=stats)

        return forecast

    def freeze_backbone(self):
        """Freeze backbone for fine-tuning."""
        self.backbone.freeze()

    def unfreeze_backbone(self):
        """Unfreeze backbone."""
        self.backbone.unfreeze()

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Get parameter count per component.

        Returns:
            Dictionary mapping component name to parameter count
        """
        return {
            "patch_embedding": sum(
                p.numel() for p in self.patch_embedding.parameters()
            ),
            "resolution_prefix": (
                sum(p.numel() for p in self.resolution_prefix.parameters())
                if self.use_resolution_prefix
                else 0
            ),
            "backbone": self.backbone.get_num_parameters(),
            "decoder": self.decoder.get_num_parameters(),
            "forecast_head": sum(p.numel() for p in self.forecast_head.parameters()),
            "total": self.get_num_parameters(),
        }

    @classmethod
    def from_config(
        cls,
        config: TTMConfig,
        num_channels: int = 1,
    ) -> "TTM":
        """Create model from configuration.

        Args:
            config: Model configuration
            num_channels: Number of input channels

        Returns:
            TTM model instance
        """
        return cls(config=config, num_channels=num_channels)

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model
        """
        torch.save(
            {
                "config": self.config,
                "num_channels": self.num_channels,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "TTM":
        """Load model from file.

        Args:
            path: Path to saved model
            map_location: Device to load model to

        Returns:
            Loaded TTM model
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            config=checkpoint["config"],
            num_channels=checkpoint["num_channels"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


class TTMForPretrain(TTM):
    """TTM configured for pre-training.

    Same architecture as TTM but with channel-independent processing
    and appropriate loss function.
    """

    def __init__(
        self,
        config: TTMConfig | None = None,
    ):
        """Initialize TTM for pre-training.

        Args:
            config: Model configuration
        """
        # Force channel-independent settings
        if config is None:
            config = TTMConfig()

        config.channel_independent_backbone = True
        config.channel_mixing_decoder = False

        super().__init__(config=config, num_channels=1)


class TTMForFinetune(TTM):
    """TTM configured for fine-tuning.

    Backbone is frozen, only decoder and head are trainable.
    """

    def __init__(
        self,
        config: TTMConfig | None = None,
        num_channels: int = 1,
        pretrained_path: str | None = None,
    ):
        """Initialize TTM for fine-tuning.

        Args:
            config: Model configuration
            num_channels: Number of input channels
            pretrained_path: Path to pretrained weights
        """
        super().__init__(config=config, num_channels=num_channels)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            pretrained = torch.load(pretrained_path, map_location="cpu")
            # Load backbone weights only
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in pretrained["state_dict"].items()
                if k.startswith("backbone.")
            }
            self.backbone.load_state_dict(backbone_state)

        # Freeze backbone
        self.freeze_backbone()
