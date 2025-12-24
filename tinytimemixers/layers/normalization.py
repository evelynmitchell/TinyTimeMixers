"""Normalization layers for TinyTimeMixers."""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Per-instance, per-channel normalization that can be reversed
    to restore the original scale. Used for time series to handle
    varying scales across different series.

    Reference: https://arxiv.org/abs/2110.07610
    """

    def __init__(
        self,
        num_features: int | None = None,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        """Initialize RevIN.

        Args:
            num_features: Number of features/channels (optional, for affine)
            eps: Small constant for numerical stability
            affine: Whether to use learnable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            if num_features is None:
                raise ValueError("num_features required when affine=True")
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "normalize",
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply or reverse normalization.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            mode: "normalize" or "denormalize"
            stats: (mean, std) tuple for denormalization

        Returns:
            If mode="normalize": (normalized_x, (mean, std))
            If mode="denormalize": denormalized_x
        """
        if mode == "normalize":
            return self._normalize(x)
        elif mode == "denormalize":
            if stats is None:
                raise ValueError("stats required for denormalization")
            return self._denormalize(x, stats)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _normalize(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalize input and return statistics.

        Args:
            x: Input tensor (batch, channels, seq_len)

        Returns:
            Tuple of (normalized_x, (mean, std))
        """
        # Compute per-instance, per-channel statistics
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps

        # Normalize
        x_norm = (x - mean) / std

        # Apply affine if enabled
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm, (mean, std)

    def _denormalize(
        self,
        x: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Reverse normalization using stored statistics.

        Args:
            x: Normalized tensor (batch, channels, seq_len)
            stats: (mean, std) from normalization

        Returns:
            Denormalized tensor
        """
        mean, std = stats

        # Remove affine if enabled
        if self.affine:
            x = (x - self.beta) / self.gamma

        # Reverse normalization
        return x * std + mean


class BatchNorm1d(nn.Module):
    """Batch normalization for time series.

    Wrapper around nn.BatchNorm1d with proper dimension handling.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        """Initialize BatchNorm1d.

        Args:
            num_features: Number of features to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization.

        Args:
            x: Input tensor (batch, features, seq_len) or
               (batch, channels, patches, features)

        Returns:
            Normalized tensor
        """
        if x.dim() == 3:
            return self.norm(x)
        elif x.dim() == 4:
            # (batch, channels, patches, features)
            b, c, p, f = x.shape
            x = x.reshape(b * c, p, f).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).reshape(b, c, p, f)
            return x
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
