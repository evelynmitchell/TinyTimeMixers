"""Patch embedding layer for TinyTimeMixers."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Patch embedding layer.

    Extracts non-overlapping patches from input time series
    and projects them to the hidden dimension.

    Input: (batch, channels, seq_len)
    Output: (batch, channels, num_patches, hidden_features)
    """

    def __init__(
        self,
        patch_length: int,
        patch_stride: int,
        hidden_features: int,
    ):
        """Initialize PatchEmbedding.

        Args:
            patch_length: Length of each patch
            patch_stride: Stride between patches
            hidden_features: Output feature dimension
        """
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.hidden_features = hidden_features

        # Linear projection from patch to hidden dimension
        self.projection = nn.Linear(patch_length, hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and embed patches.

        Args:
            x: Input tensor (batch, channels, seq_len)

        Returns:
            Embedded patches (batch, channels, num_patches, hidden_features)
        """
        batch_size, num_channels, seq_len = x.shape

        # Extract patches using unfold
        # (batch, channels, num_patches, patch_length)
        patches = x.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.patch_stride,
        )

        # Project to hidden dimension
        # (batch, channels, num_patches, hidden_features)
        embedded = self.projection(patches)

        return embedded

    def compute_num_patches(self, seq_len: int) -> int:
        """Compute number of patches for a given sequence length.

        Args:
            seq_len: Input sequence length

        Returns:
            Number of patches
        """
        return (seq_len - self.patch_length) // self.patch_stride + 1


class PatchUnembedding(nn.Module):
    """Reverse patch embedding for reconstruction.

    Projects from hidden dimension back to patch space and
    reconstructs the time series.

    Input: (batch, channels, num_patches, hidden_features)
    Output: (batch, channels, seq_len)
    """

    def __init__(
        self,
        patch_length: int,
        patch_stride: int,
        hidden_features: int,
    ):
        """Initialize PatchUnembedding.

        Args:
            patch_length: Length of each patch
            patch_stride: Stride between patches
            hidden_features: Input feature dimension
        """
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.hidden_features = hidden_features

        # Linear projection from hidden dimension to patch
        self.projection = nn.Linear(hidden_features, patch_length)

    def forward(
        self,
        x: torch.Tensor,
        target_len: int | None = None,
    ) -> torch.Tensor:
        """Unembed patches to time series.

        Args:
            x: Embedded patches (batch, channels, num_patches, hidden_features)
            target_len: Target sequence length (optional)

        Returns:
            Reconstructed time series (batch, channels, seq_len)
        """
        batch_size, num_channels, num_patches, _ = x.shape

        # Project back to patch space
        # (batch, channels, num_patches, patch_length)
        patches = self.projection(x)

        # For non-overlapping patches, simply reshape
        if self.patch_stride == self.patch_length:
            # (batch, channels, num_patches * patch_length)
            return patches.reshape(batch_size, num_channels, -1)

        # For overlapping patches, use fold operation
        # This is more complex and handles averaging of overlapping regions
        seq_len = target_len or (
            (num_patches - 1) * self.patch_stride + self.patch_length
        )

        # Reshape for fold: (batch * channels, patch_length, num_patches)
        patches = patches.reshape(
            batch_size * num_channels, num_patches, self.patch_length
        ).transpose(1, 2)

        # Use fold to reconstruct
        output = torch.nn.functional.fold(
            patches,
            output_size=(1, seq_len),
            kernel_size=(1, self.patch_length),
            stride=(1, self.patch_stride),
        )

        # Average overlapping regions
        ones = torch.ones_like(patches)
        divisor = torch.nn.functional.fold(
            ones,
            output_size=(1, seq_len),
            kernel_size=(1, self.patch_length),
            stride=(1, self.patch_stride),
        )

        output = output / divisor.clamp(min=1)

        # Reshape back: (batch, channels, seq_len)
        return output.reshape(batch_size, num_channels, seq_len)
