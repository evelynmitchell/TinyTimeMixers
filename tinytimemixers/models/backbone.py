"""TTM Backbone module."""

import torch
import torch.nn as nn

from tinytimemixers.config import TTMConfig
from tinytimemixers.layers.patch_partition import PatchPartition
from tinytimemixers.layers.tsmixer_block import TSMixerLevel


class TTMBackbone(nn.Module):
    """TTM Backbone with multi-level TSMixer blocks.

    The backbone consists of L levels, each containing M TSMixer blocks.
    Each level operates at a different resolution using adaptive patching:
    - Level 0 (first): Ki = 2^(L-1), highest partition factor
    - Level L-1 (last): Ki = 1, original resolution

    The backbone is channel-independent (no channel mixing) and is
    typically frozen during fine-tuning.

    Architecture:
        Input -> [Level 0: Partition -> M blocks -> Merge] ->
                 [Level 1: Partition -> M blocks -> Merge] ->
                 ... ->
                 [Level L-1: M blocks] -> Output
    """

    def __init__(self, config: TTMConfig):
        """Initialize TTMBackbone.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.num_levels = config.num_backbone_levels
        self.num_patches = config.num_patches

        # Create partition modules for each level
        self.partitions = nn.ModuleList(
            [
                PatchPartition(level=i, num_levels=self.num_levels)
                for i in range(self.num_levels)
            ]
        )

        # Create TSMixer levels
        self.levels = nn.ModuleList()
        for level_idx in range(self.num_levels):
            partition = self.partitions[level_idx]
            k = partition.partition_factor

            # Effective dimensions at this level
            effective_patches = self.num_patches * k
            effective_features = config.hidden_features // k
            effective_expansion = config.expansion_features // k

            level = TSMixerLevel(
                num_blocks=config.blocks_per_level,
                num_patches=effective_patches,
                hidden_features=effective_features,
                expansion_features=effective_expansion,
                num_channels=None,  # Channel-independent
                channel_mixing=False,  # No channel mixing in backbone
                dropout=config.dropout,
            )
            self.levels.append(level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backbone.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Output (batch, channels, num_patches, hidden_features)
        """
        for level_idx, (partition, level) in enumerate(
            zip(self.partitions, self.levels)
        ):
            # Partition to level resolution
            x = partition.partition(x)

            # Apply TSMixer blocks
            x = level(x)

            # Merge back to original resolution
            x = partition.merge(x, self.num_patches)

        return x

    def freeze(self):
        """Freeze all backbone parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> int:
        """Get total number of parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())


class TTMBackboneLight(nn.Module):
    """Lightweight backbone without adaptive patching.

    Simpler version that skips the partition/merge operations.
    Useful for debugging and ablation studies.
    """

    def __init__(self, config: TTMConfig):
        """Initialize TTMBackboneLight.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Stack all levels without adaptive patching
        self.levels = nn.ModuleList(
            [
                TSMixerLevel(
                    num_blocks=config.blocks_per_level,
                    num_patches=config.num_patches,
                    hidden_features=config.hidden_features,
                    expansion_features=config.expansion_features,
                    num_channels=None,
                    channel_mixing=False,
                    dropout=config.dropout,
                )
                for _ in range(config.num_backbone_levels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backbone.

        Args:
            x: Input (batch, channels, num_patches, hidden_features)

        Returns:
            Output (batch, channels, num_patches, hidden_features)
        """
        for level in self.levels:
            x = level(x)
        return x

    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
