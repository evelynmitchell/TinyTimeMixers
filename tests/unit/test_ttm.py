"""Unit tests for TTM model."""

import pytest
import torch

from tinytimemixers import TTM, TTMConfig


class TestTTMConfig:
    """Tests for TTMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TTMConfig()
        assert config.context_length == 512
        assert config.prediction_length == 96
        assert config.patch_length == 64
        assert config.num_backbone_levels == 6
        assert config.blocks_per_level == 2

    def test_computed_properties(self):
        """Test computed properties."""
        config = TTMConfig()
        assert config.hidden_features == 192  # 3 * 64
        assert config.expansion_features == 384  # 192 * 2
        assert config.num_patches == 8  # 512 / 64

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            # context_length not divisible by patch_stride
            TTMConfig(context_length=500, patch_stride=64)


class TestTTM:
    """Tests for TTM model."""

    def test_model_creation(self):
        """Test model can be created."""
        config = TTMConfig()
        model = TTM(config, num_channels=1)
        assert model is not None

    def test_forward_pass_univariate(self):
        """Test forward pass with univariate input."""
        config = TTMConfig()
        model = TTM(config, num_channels=1)

        x = torch.randn(2, 1, config.context_length)
        y = model(x)

        assert y.shape == (2, 1, config.prediction_length)

    def test_forward_pass_multivariate(self):
        """Test forward pass with multivariate input."""
        config = TTMConfig()
        model = TTM(config, num_channels=7)

        x = torch.randn(4, 7, config.context_length)
        y = model(x)

        assert y.shape == (4, 7, config.prediction_length)

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        config = TTMConfig()
        model = TTM(config, num_channels=1)

        x = torch.randn(2, 1, config.context_length, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        config = TTMConfig()
        model = TTM(config, num_channels=1)

        num_params = model.get_num_parameters()
        # Should be under 5M for a lightweight model
        assert num_params < 5_000_000

    def test_freeze_backbone(self):
        """Test freezing backbone."""
        config = TTMConfig()
        model = TTM(config, num_channels=1)

        model.freeze_backbone()

        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_save_load(self, tmp_path):
        """Test model save and load."""
        config = TTMConfig()
        model = TTM(config, num_channels=3)

        # Save
        path = tmp_path / "model.pt"
        model.save(str(path))

        # Load
        loaded = TTM.load(str(path))

        assert loaded.num_channels == model.num_channels
        assert loaded.get_num_parameters() == model.get_num_parameters()


class TestTTMLayers:
    """Tests for individual layers."""

    def test_revin_normalize(self):
        """Test RevIN normalization."""
        from tinytimemixers.layers.normalization import RevIN

        revin = RevIN()
        x = torch.randn(2, 3, 100)

        x_norm, stats = revin(x, mode="normalize")

        # Check normalized values have ~0 mean, ~1 std
        assert torch.abs(x_norm.mean(dim=-1)).max() < 1e-4
        assert torch.abs(x_norm.std(dim=-1) - 1).max() < 1e-4

    def test_revin_denormalize(self):
        """Test RevIN denormalization."""
        from tinytimemixers.layers.normalization import RevIN

        revin = RevIN()
        x = torch.randn(2, 3, 100)

        x_norm, stats = revin(x, mode="normalize")
        x_denorm = revin(x_norm, mode="denormalize", stats=stats)

        assert torch.allclose(x, x_denorm, atol=1e-5)

    def test_patch_embedding(self):
        """Test patch embedding."""
        from tinytimemixers.layers.patch_embedding import PatchEmbedding

        embed = PatchEmbedding(patch_length=64, patch_stride=64, hidden_features=192)
        x = torch.randn(2, 3, 512)

        y = embed(x)

        assert y.shape == (2, 3, 8, 192)  # 8 patches, 192 features

    def test_tsmixer_block(self):
        """Test TSMixer block."""
        from tinytimemixers.layers.tsmixer_block import TSMixerBlock

        block = TSMixerBlock(
            num_patches=8,
            hidden_features=192,
            expansion_features=384,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 8, 192)

        y = block(x)

        assert y.shape == x.shape
