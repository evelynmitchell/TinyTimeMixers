"""Adaptive patch partition for TinyTimeMixers."""

import torch
import torch.nn as nn


class PatchPartition(nn.Module):
    """Adaptive patch partition/merge for multi-level modeling.

    In TTM, each backbone level i has a different effective resolution.
    The partition factor Ki = 2^(L-i) where L is the total number of levels.

    Partition: Reshape from (c, n, hf) to (c, n*Ki, hf/Ki)
    Merge: Reshape from (c, n*Ki, hf/Ki) back to (c, n, hf)

    This allows different levels to operate at different resolutions
    while maintaining the same total capacity.
    """

    def __init__(self, level: int, num_levels: int):
        """Initialize PatchPartition.

        Args:
            level: Current level index (0 to num_levels-1)
            num_levels: Total number of levels (L)
        """
        super().__init__()
        self.level = level
        self.num_levels = num_levels

        # Ki = 2^(L - 1 - i) for level i (0-indexed)
        # Level 0 (first) has highest K, level L-1 (last) has K=1
        self.partition_factor = 2 ** (num_levels - 1 - level)

    def partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition patches to higher resolution.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Partitioned (batch, channels, num_patches*K, hidden_features/K)
        """
        if self.partition_factor == 1:
            return x

        batch, channels, num_patches, hidden_features = x.shape
        k = self.partition_factor

        if hidden_features % k != 0:
            raise ValueError(
                f"hidden_features ({hidden_features}) must be divisible "
                f"by partition_factor ({k})"
            )

        # Reshape: (b, c, n, hf) -> (b, c, n, k, hf/k) -> (b, c, n*k, hf/k)
        x = x.reshape(batch, channels, num_patches, k, hidden_features // k)
        x = x.reshape(batch, channels, num_patches * k, hidden_features // k)

        return x

    def merge(self, x: torch.Tensor, original_patches: int) -> torch.Tensor:
        """Merge patches back to original resolution.

        Args:
            x: Partitioned (batch, channels, num_patches*K, hidden_features/K)
            original_patches: Original number of patches before partition

        Returns:
            Merged (batch, channels, num_patches, hidden_features)
        """
        if self.partition_factor == 1:
            return x

        batch, channels, _, features_per_k = x.shape
        k = self.partition_factor

        # Reshape: (b, c, n*k, hf/k) -> (b, c, n, k, hf/k) -> (b, c, n, hf)
        x = x.reshape(batch, channels, original_patches, k, features_per_k)
        x = x.reshape(batch, channels, original_patches, k * features_per_k)

        return x

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "partition",
        original_patches: int | None = None,
    ) -> torch.Tensor:
        """Apply partition or merge.

        Args:
            x: Input tensor
            mode: "partition" or "merge"
            original_patches: Required for merge mode

        Returns:
            Transformed tensor
        """
        if mode == "partition":
            return self.partition(x)
        elif mode == "merge":
            if original_patches is None:
                raise ValueError("original_patches required for merge")
            return self.merge(x, original_patches)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class AdaptivePatchHandler(nn.Module):
    """Handles adaptive patching across all backbone levels.

    Creates partition modules for each level and provides
    convenient methods to transform between levels.
    """

    def __init__(self, num_levels: int):
        """Initialize AdaptivePatchHandler.

        Args:
            num_levels: Total number of levels (L)
        """
        super().__init__()
        self.num_levels = num_levels

        self.partitions = nn.ModuleList(
            [PatchPartition(level=i, num_levels=num_levels) for i in range(num_levels)]
        )

    def get_effective_patches(
        self,
        num_patches: int,
        level: int,
    ) -> int:
        """Get number of patches at a given level.

        Args:
            num_patches: Base number of patches
            level: Level index

        Returns:
            Effective number of patches at this level
        """
        return num_patches * self.partitions[level].partition_factor

    def get_effective_features(
        self,
        hidden_features: int,
        level: int,
    ) -> int:
        """Get feature dimension at a given level.

        Args:
            hidden_features: Base feature dimension
            level: Level index

        Returns:
            Effective feature dimension at this level
        """
        return hidden_features // self.partitions[level].partition_factor

    def partition_at_level(
        self,
        x: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Partition input for a specific level.

        Args:
            x: Input tensor
            level: Target level

        Returns:
            Partitioned tensor
        """
        return self.partitions[level].partition(x)

    def merge_at_level(
        self,
        x: torch.Tensor,
        level: int,
        original_patches: int,
    ) -> torch.Tensor:
        """Merge output from a specific level.

        Args:
            x: Partitioned tensor
            level: Current level
            original_patches: Original number of patches

        Returns:
            Merged tensor
        """
        return self.partitions[level].merge(x, original_patches)
