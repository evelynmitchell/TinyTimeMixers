"""Unit tests for training infrastructure."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tinytimemixers.config import TrainingConfig, TTMConfig
from tinytimemixers.data.dataset import TimeSeriesDataset
from tinytimemixers.models.ttm import TTM
from tinytimemixers.training.losses import (
    ForecastLoss,
    HuberLoss,
    MAELoss,
    MASELoss,
    MSELoss,
    QuantileLoss,
    get_loss_fn,
)
from tinytimemixers.training.optimizer import (
    WarmupScheduler,
    create_optimizer,
    create_scheduler,
    get_parameter_count,
    get_parameter_groups_summary,
)
from tinytimemixers.training.trainer import EarlyStopping, Trainer, TrainingMetrics
from tinytimemixers.utils.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
    save_model,
)


class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        loss_fn = MSELoss()
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mae_loss(self):
        """Test MAE loss computation."""
        loss_fn = MAELoss()
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_huber_loss(self):
        """Test Huber loss computation."""
        loss_fn = HuberLoss(delta=1.0)
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_quantile_loss(self):
        """Test quantile loss computation."""
        loss_fn = QuantileLoss(quantile=0.9)
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_mase_loss(self):
        """Test MASE loss computation."""
        loss_fn = MASELoss(seasonality=1)
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)
        context = torch.randn(4, 3, 512)

        loss = loss_fn(pred, target, context=context)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_masked_loss(self):
        """Test loss with mask."""
        loss_fn = MSELoss()
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)
        mask = torch.ones_like(pred)
        mask[:, :, 50:] = 0  # Mask out second half

        loss_masked = loss_fn(pred, target, mask=mask)
        loss_full = loss_fn(pred, target)

        assert loss_masked.shape == ()
        assert loss_masked != loss_full

    def test_get_loss_fn(self):
        """Test loss function factory."""
        for loss_type in ["mse", "mae", "huber", "quantile"]:
            loss_fn = get_loss_fn(loss_type)
            assert isinstance(loss_fn, ForecastLoss)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.randn(4, 3, 96)
        target = torch.randn(4, 3, 96)

        for reduction in ["mean", "sum", "none"]:
            loss_fn = MSELoss(reduction=reduction)
            loss = loss_fn(pred, target)

            if reduction == "none":
                assert loss.shape == pred.shape
            else:
                assert loss.shape == ()


class TestOptimizer:
    """Test optimizer utilities."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def test_create_optimizer_adamw(self, simple_model):
        """Test AdamW optimizer creation."""
        optimizer = create_optimizer(
            simple_model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_optimizer_adam(self, simple_model):
        """Test Adam optimizer creation."""
        optimizer = create_optimizer(
            simple_model,
            optimizer_type="adam",
            learning_rate=1e-4,
        )
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_sgd(self, simple_model):
        """Test SGD optimizer creation."""
        optimizer = create_optimizer(
            simple_model,
            optimizer_type="sgd",
            learning_rate=1e-4,
        )
        assert isinstance(optimizer, torch.optim.SGD)

    def test_create_scheduler_cosine(self, simple_model):
        """Test cosine scheduler creation."""
        optimizer = create_optimizer(simple_model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_epochs=100,
        )
        assert scheduler is not None

    def test_create_scheduler_none(self, simple_model):
        """Test no scheduler."""
        optimizer = create_optimizer(simple_model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="none",
        )
        assert scheduler is None

    def test_warmup_scheduler(self, simple_model):
        """Test warmup scheduler."""
        optimizer = create_optimizer(simple_model, learning_rate=1e-3)
        warmup = WarmupScheduler(
            optimizer,
            warmup_steps=10,
        )

        # After first step, LR should be 1/10 of target
        warmup.step()
        first_step_lr = warmup.get_last_lr()[0]
        assert first_step_lr == pytest.approx(1e-3 / 10)

        # After 10 steps, should reach full LR
        for _ in range(9):
            warmup.step()
        final_lr = warmup.get_last_lr()[0]
        assert final_lr == pytest.approx(1e-3)

    def test_parameter_count(self, simple_model):
        """Test parameter counting."""
        count = get_parameter_count(simple_model)
        assert count > 0

        # Should be 10*20 + 20 + 20*10 + 10 = 430
        assert count == 430

    def test_parameter_groups_summary(self, simple_model):
        """Test parameter groups summary."""
        summary = get_parameter_groups_summary(simple_model)
        assert len(summary) > 0


class TestEarlyStopping:
    """Test early stopping callback."""

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after no improvement."""
        stopper = EarlyStopping(patience=3)

        assert not stopper(1.0)
        assert not stopper(1.0)
        assert not stopper(1.0)
        assert stopper(1.0)  # Should stop after 3 epochs without improvement

    def test_early_stopping_with_improvement(self):
        """Test early stopping resets on improvement."""
        stopper = EarlyStopping(patience=2)

        assert not stopper(1.0)
        assert not stopper(1.0)
        assert not stopper(0.5)  # Improvement resets counter
        assert not stopper(0.5)
        assert stopper(0.5)  # Stops after patience

    def test_early_stopping_min_delta(self):
        """Test minimum delta threshold."""
        stopper = EarlyStopping(patience=2, min_delta=0.1)

        assert not stopper(1.0)
        assert not stopper(0.99)  # Not enough improvement
        assert stopper(0.99)  # Stops


class TestCheckpoint:
    """Test checkpoint utilities."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Linear(10, 5)

    def test_save_load_model(self, simple_model):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"

            # Save without config (simple model doesn't use TTMConfig)
            save_model(simple_model, path, metadata={"test": True})

            # Check files exist
            assert (path.with_suffix(".pt")).exists()
            assert (path.with_suffix(".json")).exists()

            # Load - create new model and load weights directly
            loaded_model = nn.Linear(10, 5)
            state_dict = torch.load(path.with_suffix(".pt"), weights_only=True)
            loaded_model.load_state_dict(state_dict)

            # Check weights match
            for p1, p2 in zip(
                simple_model.parameters(),
                loaded_model.parameters(),
                strict=True,
            ):
                assert torch.allclose(p1, p2)

    def test_save_load_checkpoint(self, simple_model):
        """Test saving and loading training checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            optimizer = torch.optim.Adam(simple_model.parameters())

            # Save
            save_checkpoint(
                path=path,
                model=simple_model,
                optimizer=optimizer,
                epoch=5,
                step=100,
                metrics={"val_loss": 0.5},
            )

            # Create new model and optimizer
            new_model = nn.Linear(10, 5)
            new_optimizer = torch.optim.Adam(new_model.parameters())

            # Load
            checkpoint = load_checkpoint(
                path,
                model=new_model,
                optimizer=new_optimizer,
            )

            assert checkpoint["epoch"] == 5
            assert checkpoint["step"] == 100
            assert checkpoint["metrics"]["val_loss"] == 0.5

    def test_checkpoint_manager(self, simple_model):
        """Test checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                max_checkpoints=2,
                save_best=True,
            )

            # Save multiple checkpoints
            for epoch in range(5):
                manager.save(
                    model=simple_model,
                    epoch=epoch,
                    metric=1.0 / (epoch + 1),  # Improving metric
                )

            # Check only 2 regular checkpoints kept
            checkpoints = list(Path(tmpdir).glob("epoch_*.pt"))
            assert len(checkpoints) == 2

            # Check best checkpoint exists
            assert (Path(tmpdir) / "best.pt").exists()


class TestTrainer:
    """Test trainer class."""

    @pytest.fixture
    def training_setup(self):
        """Create training setup."""
        config = TTMConfig(
            context_length=64,
            prediction_length=16,
            patch_length=8,
            patch_stride=8,
            num_backbone_levels=2,
            blocks_per_level=1,
            feature_scaler=2,
        )
        model = TTM(config=config, num_channels=1)

        # Create synthetic data
        data = torch.randn(10, 1, 128)
        dataset = TimeSeriesDataset(
            data,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        return model, loader, config

    def test_trainer_single_epoch(self, training_setup):
        """Test training for one epoch."""
        model, loader, config = training_setup

        training_config = TrainingConfig(
            learning_rate=1e-3,
            num_epochs=1,
            warmup_steps=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=loader,
                training_config=training_config,
                model_config=config,
                checkpoint_dir=tmpdir,
            )

            loss = trainer.train_epoch()
            assert loss > 0

    def test_trainer_full_training(self, training_setup):
        """Test full training loop."""
        model, loader, config = training_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                checkpoint_dir=tmpdir,
            )

            metrics = trainer.train(
                num_epochs=2,
                early_stopping=False,
                verbose=False,
            )

            assert isinstance(metrics, TrainingMetrics)
            assert len(metrics.history) == 2
            assert (Path(tmpdir) / "final.pt").exists()

    def test_trainer_validation(self, training_setup):
        """Test validation during training."""
        model, loader, config = training_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                checkpoint_dir=tmpdir,
            )

            val_loss = trainer.validate()
            assert val_loss > 0

    def test_trainer_checkpoint_save_load(self, training_setup):
        """Test checkpoint saving and loading."""
        model, loader, config = training_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=loader,
                checkpoint_dir=tmpdir,
            )

            # Train and save
            trainer.train(num_epochs=1, verbose=False)
            trainer.save_checkpoint("test.pt")

            # Load
            trainer.load_checkpoint(Path(tmpdir) / "test.pt")
            assert trainer.current_epoch == 0  # After 1 epoch


class TestTrainingMetrics:
    """Test training metrics dataclass."""

    def test_metrics_initialization(self):
        """Test default metrics values."""
        metrics = TrainingMetrics()

        assert metrics.train_loss == 0.0
        assert metrics.val_loss is None
        assert metrics.best_val_loss == float("inf")
        assert len(metrics.history) == 0

    def test_metrics_history_tracking(self):
        """Test history tracking."""
        metrics = TrainingMetrics()
        metrics.history.append({"epoch": 0, "train_loss": 1.0})
        metrics.history.append({"epoch": 1, "train_loss": 0.5})

        assert len(metrics.history) == 2
        assert metrics.history[-1]["train_loss"] == 0.5
