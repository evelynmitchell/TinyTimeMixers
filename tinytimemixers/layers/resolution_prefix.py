"""Resolution prefix tuning for TinyTimeMixers."""

import torch
import torch.nn as nn


class ResolutionPrefix(nn.Module):
    """Resolution prefix tuning layer.

    Adds a learnable prefix token to the patch sequence that encodes
    the data resolution. This enables the model to adapt to different
    sampling frequencies (e.g., hourly, daily, weekly).

    The resolution is discretized into buckets and each bucket has
    a learnable embedding.

    Input: (batch, channels, num_patches, hidden_features)
    Output: (batch, channels, num_patches + 1, hidden_features)
    """

    def __init__(
        self,
        hidden_features: int,
        num_resolutions: int = 10,
    ):
        """Initialize ResolutionPrefix.

        Args:
            hidden_features: Feature dimension
            num_resolutions: Number of resolution buckets
        """
        super().__init__()
        self.hidden_features = hidden_features
        self.num_resolutions = num_resolutions

        # Learnable resolution embeddings
        self.resolution_embeddings = nn.Embedding(num_resolutions, hidden_features)

    def forward(
        self,
        x: torch.Tensor,
        resolution_idx: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Add resolution prefix to input.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)
            resolution_idx: Resolution bucket index (batch,) or scalar
                If None, uses default resolution (0)

        Returns:
            Output with prefix (batch, channels, num_patches + 1, hidden_features)
        """
        batch_size, num_channels, num_patches, _ = x.shape

        # Handle resolution index
        if resolution_idx is None:
            resolution_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        elif isinstance(resolution_idx, int):
            resolution_idx = torch.full(
                (batch_size,), resolution_idx, dtype=torch.long, device=x.device
            )

        # Get resolution embeddings: (batch, hidden_features)
        prefix = self.resolution_embeddings(resolution_idx)

        # Expand for channels: (batch, channels, 1, hidden_features)
        prefix = (
            prefix.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, num_channels, 1, self.hidden_features)
        )

        # Concatenate prefix to patches
        # (batch, channels, num_patches + 1, hidden_features)
        return torch.cat([prefix, x], dim=2)

    def remove_prefix(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the resolution prefix.

        Args:
            x: Input with prefix (batch, channels, num_patches + 1, hidden_features)

        Returns:
            Output without prefix (batch, channels, num_patches, hidden_features)
        """
        return x[:, :, 1:, :]


class ResolutionEncoder(nn.Module):
    """Encode time series metadata into resolution index.

    Maps sampling frequency information to discrete resolution buckets.
    """

    # Common time series frequencies in seconds
    FREQUENCY_MAP = {
        "S": 1,  # Secondly
        "T": 60,  # Minutely
        "H": 3600,  # Hourly
        "D": 86400,  # Daily
        "W": 604800,  # Weekly
        "M": 2592000,  # Monthly (30 days)
        "Q": 7776000,  # Quarterly (90 days)
        "Y": 31536000,  # Yearly
    }

    def __init__(self, num_resolutions: int = 10):
        """Initialize ResolutionEncoder.

        Args:
            num_resolutions: Number of resolution buckets
        """
        super().__init__()
        self.num_resolutions = num_resolutions

        # Create bucket boundaries (log-spaced)
        min_freq = 1  # 1 second
        max_freq = 31536000  # 1 year
        self.register_buffer(
            "bucket_boundaries",
            torch.logspace(
                torch.log10(torch.tensor(min_freq)),
                torch.log10(torch.tensor(max_freq)),
                num_resolutions + 1,
            ),
        )

    def forward(
        self,
        frequency: str | float | torch.Tensor,
    ) -> torch.Tensor:
        """Encode frequency to resolution index.

        Args:
            frequency: Sampling frequency as string ("H", "D", etc.),
                seconds (float), or tensor of seconds

        Returns:
            Resolution bucket index
        """
        # Convert string frequency to seconds
        if isinstance(frequency, str):
            if frequency not in self.FREQUENCY_MAP:
                raise ValueError(f"Unknown frequency: {frequency}")
            freq_seconds = self.FREQUENCY_MAP[frequency]
        else:
            freq_seconds = frequency

        # Convert to tensor
        if not isinstance(freq_seconds, torch.Tensor):
            freq_seconds = torch.tensor(
                freq_seconds, device=self.bucket_boundaries.device
            )

        # Find bucket (using searchsorted)
        bucket_idx = torch.searchsorted(self.bucket_boundaries, freq_seconds)

        # Clamp to valid range
        return bucket_idx.clamp(0, self.num_resolutions - 1)
