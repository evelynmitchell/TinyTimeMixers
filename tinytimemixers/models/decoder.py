"""TTM Decoder module."""

import torch
import torch.nn as nn

from tinytimemixers.config import TTMConfig
from tinytimemixers.layers.tsmixer_block import TSMixerLevel


class TTMDecoder(nn.Module):
    """TTM Decoder for task-specific adaptation.

    The decoder is smaller than the backbone (10-20% of size) and
    can optionally include channel mixing for multivariate forecasting.

    Unlike the backbone, the decoder is trained during fine-tuning
    while the backbone is frozen.

    Architecture:
        Input -> [Decoder Layer 1] -> [Decoder Layer 2] -> Output

    Each decoder layer is a TSMixerLevel with optional channel mixing.
    """

    def __init__(
        self,
        config: TTMConfig,
        num_channels: int | None = None,
    ):
        """Initialize TTMDecoder.

        Args:
            config: Model configuration
            num_channels: Number of channels (for channel mixing)
        """
        super().__init__()
        self.config = config
        self.num_channels = num_channels

        # Determine if channel mixing is enabled
        channel_mixing = config.channel_mixing_decoder and (
            num_channels is not None and num_channels > 1
        )

        # Create decoder layers
        self.layers = nn.ModuleList(
            [
                TSMixerLevel(
                    num_blocks=1,  # Single block per decoder layer
                    num_patches=config.num_patches,
                    hidden_features=config.hidden_features,
                    expansion_features=config.expansion_features,
                    num_channels=num_channels if channel_mixing else None,
                    channel_mixing=channel_mixing,
                    dropout=config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decoder.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Output (batch, channels, num_patches, hidden_features)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def get_num_parameters(self) -> int:
        """Get total number of parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())


class TTMDecoderWithCrossChannel(nn.Module):
    """TTM Decoder with explicit cross-channel attention.

    Alternative decoder design that uses attention mechanism
    for channel interactions instead of MLP-based mixing.
    """

    def __init__(
        self,
        config: TTMConfig,
        num_channels: int,
        num_heads: int = 4,
    ):
        """Initialize TTMDecoderWithCrossChannel.

        Args:
            config: Model configuration
            num_channels: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.config = config

        # Standard TSMixer layers
        self.layers = nn.ModuleList(
            [
                TSMixerLevel(
                    num_blocks=1,
                    num_patches=config.num_patches,
                    hidden_features=config.hidden_features,
                    expansion_features=config.expansion_features,
                    num_channels=None,
                    channel_mixing=False,
                    dropout=config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

        # Cross-channel attention after TSMixer layers
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_features,
            num_heads=num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(config.hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decoder with cross-channel attention.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Output (batch, channels, num_patches, hidden_features)
        """
        batch_size, num_channels, num_patches, hidden_features = x.shape

        # Apply TSMixer layers
        for layer in self.layers:
            x = layer(x)

        # Apply cross-channel attention per patch position
        # Reshape: (batch, channels, patches, features)
        #       -> (batch * patches, channels, features)
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_patches, num_channels, hidden_features
        )

        # Self-attention across channels
        residual = x
        x = self.channel_norm(x)
        x, _ = self.channel_attention(x, x, x)
        x = x + residual

        # Reshape back
        x = x.reshape(batch_size, num_patches, num_channels, hidden_features).permute(
            0, 2, 1, 3
        )

        return x
