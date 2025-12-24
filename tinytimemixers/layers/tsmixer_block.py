"""TSMixer block for TinyTimeMixers."""

import torch
import torch.nn as nn

from tinytimemixers.layers.mixer_mlp import (
    ChannelMixerMLP,
    FeatureMixerMLP,
    TimeMixerMLP,
)


class TSMixerBlock(nn.Module):
    """Single TSMixer block.

    Applies time-mixing and feature-mixing with optional channel-mixing.
    Each mixing operation is preceded by layer normalization and
    followed by a residual connection.

    Architecture:
        x -> LN -> TimeMixer -> + (residual)
                                |
        x -> LN -> FeatureMixer -> + (residual)
                                   |
        x -> LN -> ChannelMixer (optional) -> + (residual)
    """

    def __init__(
        self,
        num_patches: int,
        hidden_features: int,
        expansion_features: int,
        num_channels: int | None = None,
        channel_mixing: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize TSMixerBlock.

        Args:
            num_patches: Number of patches (time dimension)
            hidden_features: Feature dimension
            expansion_features: Expanded feature dimension for MLPs
            num_channels: Number of channels (required if channel_mixing)
            channel_mixing: Whether to include channel mixing
            dropout: Dropout probability
        """
        super().__init__()

        # Time mixing
        self.norm1 = nn.LayerNorm(hidden_features)
        self.time_mixer = TimeMixerMLP(
            num_patches=num_patches,
            expansion_factor=2,
            dropout=dropout,
        )

        # Feature mixing
        self.norm2 = nn.LayerNorm(hidden_features)
        self.feature_mixer = FeatureMixerMLP(
            num_features=hidden_features,
            expansion_features=expansion_features,
            dropout=dropout,
        )

        # Optional channel mixing
        self.channel_mixing = channel_mixing
        if channel_mixing:
            if num_channels is None:
                raise ValueError("num_channels required when channel_mixing=True")
            self.norm3 = nn.LayerNorm(hidden_features)
            self.channel_mixer = ChannelMixerMLP(
                num_channels=num_channels,
                expansion_factor=2,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TSMixer block.

        Args:
            x: Input tensor (batch, channels, patches, features)

        Returns:
            Output tensor (batch, channels, patches, features)
        """
        # Time mixing with residual
        residual = x
        x = self.norm1(x)
        x = self.time_mixer(x)
        x = x + residual

        # Feature mixing with residual
        residual = x
        x = self.norm2(x)
        x = self.feature_mixer(x)
        x = x + residual

        # Optional channel mixing with residual
        if self.channel_mixing:
            residual = x
            x = self.norm3(x)
            x = self.channel_mixer(x)
            x = x + residual

        return x


class TSMixerLevel(nn.Module):
    """A level in the TTM backbone containing multiple TSMixer blocks.

    Each level has M TSMixer blocks operating at the same resolution.
    """

    def __init__(
        self,
        num_blocks: int,
        num_patches: int,
        hidden_features: int,
        expansion_features: int,
        num_channels: int | None = None,
        channel_mixing: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize TSMixerLevel.

        Args:
            num_blocks: Number of TSMixer blocks (M in paper)
            num_patches: Number of patches at this level
            hidden_features: Feature dimension
            expansion_features: Expanded feature dimension
            num_channels: Number of channels
            channel_mixing: Whether to use channel mixing
            dropout: Dropout probability
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TSMixerBlock(
                    num_patches=num_patches,
                    hidden_features=hidden_features,
                    expansion_features=expansion_features,
                    num_channels=num_channels,
                    channel_mixing=channel_mixing,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all blocks in the level.

        Args:
            x: Input tensor (batch, channels, patches, features)

        Returns:
            Output tensor (batch, channels, patches, features)
        """
        for block in self.blocks:
            x = block(x)
        return x
