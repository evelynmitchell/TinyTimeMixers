"""Mixer MLP layers for TinyTimeMixers."""

import torch
import torch.nn as nn


class MixerMLP(nn.Module):
    """Base MLP for mixing operations.

    Two-layer MLP with GELU activation used for time, feature,
    and channel mixing in TSMixer blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize MixerMLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (expansion)
            output_dim: Output dimension (defaults to input_dim)
            dropout: Dropout probability
        """
        super().__init__()
        output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Output tensor (..., output_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TimeMixerMLP(nn.Module):
    """Time-mixing MLP.

    Mixes information across the time (patch) dimension.
    Input: (batch, channels, patches, features)
    Mixing dimension: patches
    """

    def __init__(
        self,
        num_patches: int,
        expansion_factor: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize TimeMixerMLP.

        Args:
            num_patches: Number of patches (time dimension)
            expansion_factor: Hidden dimension multiplier
            dropout: Dropout probability
        """
        super().__init__()
        hidden_dim = num_patches * expansion_factor
        self.mlp = MixerMLP(num_patches, hidden_dim, num_patches, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix across time dimension.

        Args:
            x: Input (batch, channels, patches, features)

        Returns:
            Output (batch, channels, patches, features)
        """
        # Transpose to (batch, channels, features, patches)
        x = x.transpose(-1, -2)
        x = self.mlp(x)
        # Transpose back to (batch, channels, patches, features)
        return x.transpose(-1, -2)


class FeatureMixerMLP(nn.Module):
    """Feature-mixing MLP.

    Mixes information across the feature dimension.
    Input: (batch, channels, patches, features)
    Mixing dimension: features
    """

    def __init__(
        self,
        num_features: int,
        expansion_features: int,
        dropout: float = 0.0,
    ):
        """Initialize FeatureMixerMLP.

        Args:
            num_features: Number of features (hidden dimension)
            expansion_features: Expanded hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.mlp = MixerMLP(num_features, expansion_features, num_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix across feature dimension.

        Args:
            x: Input (batch, channels, patches, features)

        Returns:
            Output (batch, channels, patches, features)
        """
        return self.mlp(x)


class ChannelMixerMLP(nn.Module):
    """Channel-mixing MLP.

    Mixes information across channels (multivariate).
    Input: (batch, channels, patches, features)
    Mixing dimension: channels
    """

    def __init__(
        self,
        num_channels: int,
        expansion_factor: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize ChannelMixerMLP.

        Args:
            num_channels: Number of channels
            expansion_factor: Hidden dimension multiplier
            dropout: Dropout probability
        """
        super().__init__()
        hidden_dim = num_channels * expansion_factor
        self.mlp = MixerMLP(num_channels, hidden_dim, num_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix across channel dimension.

        Args:
            x: Input (batch, channels, patches, features)

        Returns:
            Output (batch, channels, patches, features)
        """
        # Transpose to (batch, patches, features, channels)
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        # Transpose back to (batch, channels, patches, features)
        return x.permute(0, 3, 1, 2)
