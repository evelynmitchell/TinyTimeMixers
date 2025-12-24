"""Forecast head for TinyTimeMixers."""

import torch
import torch.nn as nn

from tinytimemixers.config import TTMConfig


class ForecastHead(nn.Module):
    """Forecast head for time series prediction.

    Projects from the hidden representation to the prediction horizon.
    Includes head dropout for regularization during fine-tuning.

    Input: (batch, channels, num_patches, hidden_features)
    Output: (batch, channels, prediction_length)
    """

    def __init__(
        self,
        config: TTMConfig,
        head_dropout: float | None = None,
    ):
        """Initialize ForecastHead.

        Args:
            config: Model configuration
            head_dropout: Dropout probability (overrides config if provided)
        """
        super().__init__()
        self.config = config
        self.prediction_length = config.prediction_length

        dropout = head_dropout if head_dropout is not None else config.head_dropout

        # Flatten patches and project to prediction length
        input_dim = config.num_patches * config.hidden_features

        self.flatten = nn.Flatten(start_dim=-2)  # Flatten patches and features
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_dim, config.prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate forecast.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Forecast (batch, channels, prediction_length)
        """
        # Flatten patches and features: (batch, channels, num_patches * hidden_features)
        x = self.flatten(x)

        # Apply dropout
        x = self.dropout(x)

        # Project to prediction length: (batch, channels, prediction_length)
        x = self.projection(x)

        return x


class ForecastHeadPatchwise(nn.Module):
    """Patch-wise forecast head.

    Projects each patch independently, then aggregates.
    May be more parameter-efficient for long prediction horizons.

    Input: (batch, channels, num_patches, hidden_features)
    Output: (batch, channels, prediction_length)
    """

    def __init__(
        self,
        config: TTMConfig,
        head_dropout: float | None = None,
    ):
        """Initialize ForecastHeadPatchwise.

        Args:
            config: Model configuration
            head_dropout: Dropout probability
        """
        super().__init__()
        self.config = config

        dropout = head_dropout if head_dropout is not None else config.head_dropout

        # Output per patch
        output_per_patch = config.prediction_length // config.num_patches
        remainder = config.prediction_length % config.num_patches

        self.output_per_patch = output_per_patch
        self.remainder = remainder

        # Per-patch projection
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(config.hidden_features, output_per_patch)

        # Handle remainder if prediction_length not divisible by num_patches
        if remainder > 0:
            self.remainder_proj = nn.Linear(config.hidden_features, remainder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate forecast.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Forecast (batch, channels, prediction_length)
        """
        batch_size, num_channels, num_patches, _ = x.shape

        # Apply dropout
        x = self.dropout(x)

        # Project each patch: (batch, channels, num_patches, output_per_patch)
        y = self.projection(x)

        # Reshape to (batch, channels, num_patches * output_per_patch)
        y = y.reshape(batch_size, num_channels, -1)

        # Handle remainder
        if self.remainder > 0:
            # Use last patch for remainder
            r = self.remainder_proj(x[:, :, -1, :])
            y = torch.cat([y, r], dim=-1)

        return y


class ForecastHeadMultiHorizon(nn.Module):
    """Multi-horizon forecast head.

    Generates predictions for multiple horizons simultaneously.
    Useful for hierarchical forecasting or when multiple
    prediction lengths are needed.

    Input: (batch, channels, num_patches, hidden_features)
    Output: dict mapping horizon -> (batch, channels, horizon)
    """

    def __init__(
        self,
        config: TTMConfig,
        horizons: list[int],
        head_dropout: float | None = None,
    ):
        """Initialize ForecastHeadMultiHorizon.

        Args:
            config: Model configuration
            horizons: List of prediction horizons
            head_dropout: Dropout probability
        """
        super().__init__()
        self.horizons = horizons

        dropout = head_dropout if head_dropout is not None else config.head_dropout
        input_dim = config.num_patches * config.hidden_features

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)

        # Separate projection for each horizon
        self.projections = nn.ModuleDict(
            {str(h): nn.Linear(input_dim, h) for h in horizons}
        )

    def forward(
        self,
        x: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """Generate forecast(s).

        Args:
            x: Input (batch, channels, num_patches, hidden_features)
            horizon: Specific horizon to predict (returns single tensor)
                If None, returns dict of all horizons

        Returns:
            If horizon specified: (batch, channels, horizon)
            Otherwise: dict[horizon, (batch, channels, horizon)]
        """
        x = self.flatten(x)
        x = self.dropout(x)

        if horizon is not None:
            if horizon not in self.horizons:
                raise ValueError(
                    f"Unknown horizon {horizon}. " f"Available: {self.horizons}"
                )
            return self.projections[str(horizon)](x)

        return {h: self.projections[str(h)](x) for h in self.horizons}
