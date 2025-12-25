"""Unit tests for benchmark infrastructure."""

import pytest

# Skip all tests if benchmark dependencies not available
pytest.importorskip("gluonts")


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_config_creation(self):
        """Test DatasetConfig can be created."""
        from benchmarks.gift_eval.config import DatasetConfig

        config = DatasetConfig(
            name="electricity",
            domain="Energy",
            freq="H",
            prediction_length=24,
            context_length=168,
            num_variates=321,
            seasonality=24,
            term="short",
        )

        assert config.name == "electricity"
        assert config.domain == "Energy"
        assert config.prediction_length == 24

    def test_config_name_property(self):
        """Test config_name property."""
        from benchmarks.gift_eval.config import DatasetConfig

        config = DatasetConfig(
            name="electricity",
            domain="Energy",
            freq="H",
            prediction_length=24,
            term="short",
        )

        assert config.config_name == "electricity_H_short"

    def test_gift_eval_datasets_not_empty(self):
        """Test GIFT_EVAL_DATASETS is populated."""
        from benchmarks.gift_eval.config import GIFT_EVAL_DATASETS

        assert len(GIFT_EVAL_DATASETS) > 0

    def test_get_datasets_by_domain(self):
        """Test filtering by domain."""
        from benchmarks.gift_eval.config import get_datasets_by_domain

        energy_datasets = get_datasets_by_domain("Energy")
        assert len(energy_datasets) > 0
        assert all(d.domain == "Energy" for d in energy_datasets)


class TestResultsAggregator:
    """Tests for ResultsAggregator."""

    def test_aggregator_creation(self):
        """Test ResultsAggregator can be created."""
        from benchmarks.gift_eval.results import ModelMetadata, ResultsAggregator

        metadata = ModelMetadata(
            model_name="TTM",
            model_version="1.0.0",
            num_parameters=1000000,
        )

        aggregator = ResultsAggregator(metadata)
        assert aggregator.model_metadata.model_name == "TTM"

    def test_add_result(self):
        """Test adding a result."""
        from benchmarks.gift_eval.results import (
            BenchmarkResult,
            ModelMetadata,
            ResultsAggregator,
        )

        metadata = ModelMetadata(model_name="TTM")
        aggregator = ResultsAggregator(metadata)

        result = BenchmarkResult(
            config_name="test_H_short",
            dataset_name="test",
            domain="Energy",
            num_variates=1,
            metrics={"MSE": 0.1, "MAE": 0.2},
            runtime_seconds=1.0,
        )

        aggregator.add_result(result)
        assert len(aggregator.results) == 1

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        from benchmarks.gift_eval.results import (
            BenchmarkResult,
            ModelMetadata,
            ResultsAggregator,
        )

        metadata = ModelMetadata(model_name="TTM")
        aggregator = ResultsAggregator(metadata)

        result = BenchmarkResult(
            config_name="test_H_short",
            dataset_name="test",
            domain="Energy",
            num_variates=1,
            metrics={"MSE": 0.1, "MAE": 0.2},
        )
        aggregator.add_result(result)

        df = aggregator.to_dataframe()
        assert len(df) == 1
        assert "dataset" in df.columns
        assert "model" in df.columns
        assert df["model"].iloc[0] == "TTM"

    def test_save_csv(self, tmp_path):
        """Test saving to CSV."""
        from benchmarks.gift_eval.results import (
            BenchmarkResult,
            ModelMetadata,
            ResultsAggregator,
        )

        metadata = ModelMetadata(model_name="TTM")
        aggregator = ResultsAggregator(metadata)

        result = BenchmarkResult(
            config_name="test_H_short",
            dataset_name="test",
            domain="Energy",
            num_variates=1,
            metrics={"MSE": 0.1},
        )
        aggregator.add_result(result)

        csv_path = tmp_path / "results.csv"
        aggregator.save_csv(csv_path)

        assert csv_path.exists()

    def test_save_config_json(self, tmp_path):
        """Test saving config.json."""
        from benchmarks.gift_eval.results import (
            ModelMetadata,
            ResultsAggregator,
        )

        metadata = ModelMetadata(
            model_name="TTM",
            model_version="1.0.0",
            num_parameters=1000000,
        )
        aggregator = ResultsAggregator(metadata)

        json_path = tmp_path / "config.json"
        aggregator.save_config_json(json_path)

        assert json_path.exists()

        import json

        with open(json_path) as f:
            config = json.load(f)
        assert config["model"] == "TTM"


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_result_creation(self):
        """Test BenchmarkResult creation."""
        from benchmarks.gift_eval.results import BenchmarkResult

        result = BenchmarkResult(
            config_name="test_H_short",
            dataset_name="test",
            domain="Energy",
            num_variates=1,
            metrics={"MSE": 0.1},
            runtime_seconds=1.0,
        )

        assert result.success
        assert result.config_name == "test_H_short"

    def test_result_with_error(self):
        """Test BenchmarkResult with error."""
        from benchmarks.gift_eval.results import BenchmarkResult

        result = BenchmarkResult(
            config_name="test_H_short",
            dataset_name="test",
            domain="Energy",
            num_variates=1,
            error="Test error",
        )

        assert not result.success
        assert result.error == "Test error"


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_metadata_creation(self):
        """Test ModelMetadata creation."""
        from benchmarks.gift_eval.results import ModelMetadata

        metadata = ModelMetadata(
            model_name="TTM",
            model_version="1.0.0",
            num_parameters=1000000,
            context_length=512,
        )

        assert metadata.model_name == "TTM"
        assert metadata.num_parameters == 1000000

    def test_metadata_to_dict(self):
        """Test ModelMetadata to_dict."""
        from benchmarks.gift_eval.results import ModelMetadata

        metadata = ModelMetadata(
            model_name="TTM",
            model_version="1.0.0",
        )

        d = metadata.to_dict()
        assert d["model"] == "TTM"
        assert "submission_date" in d


class TestValidation:
    """Tests for results validation."""

    def test_validate_valid_csv(self, tmp_path):
        """Test validation of valid CSV."""
        from benchmarks.gift_eval.results import (
            BenchmarkResult,
            ModelMetadata,
            ResultsAggregator,
            validate_results_csv,
        )

        metadata = ModelMetadata(model_name="TTM")
        aggregator = ResultsAggregator(metadata)

        result = BenchmarkResult(
            config_name="test_H_short",
            dataset_name="test",
            domain="Energy",
            num_variates=1,
            metrics={"MSE": 0.1, "MAE": 0.2},
        )
        aggregator.add_result(result)

        csv_path = tmp_path / "results.csv"
        aggregator.save_csv(csv_path)

        is_valid, errors = validate_results_csv(csv_path)
        # May have some missing columns but core should be valid
        assert "dataset" not in str(errors)

    def test_validate_missing_file(self, tmp_path):
        """Test validation of missing file."""
        from benchmarks.gift_eval.results import validate_results_csv

        csv_path = tmp_path / "nonexistent.csv"
        is_valid, errors = validate_results_csv(csv_path)

        assert not is_valid
        assert "not found" in str(errors).lower()


class TestGIFTEvalRunner:
    """Tests for GIFTEvalRunner."""

    def test_runner_checkpoint(self, tmp_path):
        """Test checkpoint saving and loading."""
        from benchmarks.gift_eval.runner import GIFTEvalRunner

        # Create a mock predictor factory
        def mock_factory(config):
            return None

        runner = GIFTEvalRunner(
            predictor_factory=mock_factory,
            output_dir=tmp_path,
            resume=False,
        )

        # Manually add some completed configs
        runner._completed_configs = {"test1", "test2"}
        runner._save_checkpoint()

        # Create new runner and load checkpoint
        runner2 = GIFTEvalRunner(
            predictor_factory=mock_factory,
            output_dir=tmp_path,
            resume=True,
        )

        assert runner2._completed_configs == {"test1", "test2"}

    def test_runner_clear_checkpoint(self, tmp_path):
        """Test clearing checkpoint."""
        from benchmarks.gift_eval.runner import GIFTEvalRunner

        def mock_factory(config):
            return None

        runner = GIFTEvalRunner(
            predictor_factory=mock_factory,
            output_dir=tmp_path,
            resume=False,
        )

        runner._completed_configs = {"test1"}
        runner._save_checkpoint()
        runner.clear_checkpoint()

        assert len(runner._completed_configs) == 0
        assert not (tmp_path / "checkpoint.json").exists()


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_comparison_result_creation(self):
        """Test ComparisonResult creation."""
        from benchmarks.tabpfn_comparison.compare import ComparisonResult

        result = ComparisonResult(
            config_name="test_H_short",
            domain="Energy",
            ttm_metrics={"MSE": 0.1},
            tabpfn_metrics={"MSE": 0.2},
            ttm_runtime=1.0,
            tabpfn_runtime=2.0,
            winner="TTM",
        )

        assert result.winner == "TTM"
        assert result.ttm_metrics["MSE"] == 0.1

    def test_comparison_result_to_dict(self):
        """Test ComparisonResult to_dict."""
        from benchmarks.tabpfn_comparison.compare import ComparisonResult

        result = ComparisonResult(
            config_name="test_H_short",
            domain="Energy",
            ttm_metrics={"MSE": 0.1},
            tabpfn_metrics={"MSE": 0.2},
            winner="TTM",
        )

        d = result.to_dict()
        assert d["config_name"] == "test_H_short"
        assert d["winner"] == "TTM"


class TestModelComparator:
    """Tests for ModelComparator."""

    def test_comparator_creation(self):
        """Test ModelComparator creation."""
        from benchmarks.tabpfn_comparison.compare import ModelComparator

        def mock_factory(config):
            return None

        comparator = ModelComparator(
            ttm_predictor_factory=mock_factory,
            tabpfn_predictor_factory=None,
            output_dir="results/test",
        )

        assert comparator.primary_metric == "MASE"

    def test_determine_winner(self):
        """Test winner determination."""
        from benchmarks.tabpfn_comparison.compare import ModelComparator

        def mock_factory(config):
            return None

        comparator = ModelComparator(
            ttm_predictor_factory=mock_factory,
            primary_metric="MSE",
        )

        # TTM wins (lower MSE)
        winner = comparator._determine_winner(
            {"MSE": 0.1},
            {"MSE": 0.2},
            "MSE",
        )
        assert winner == "TTM"

        # TabPFN wins
        winner = comparator._determine_winner(
            {"MSE": 0.3},
            {"MSE": 0.1},
            "MSE",
        )
        assert winner == "TabPFN"

        # Tie
        winner = comparator._determine_winner(
            {"MSE": 0.1},
            {"MSE": 0.1},
            "MSE",
        )
        assert winner == "Tie"

    def test_compute_statistics(self):
        """Test computing statistics."""
        from benchmarks.tabpfn_comparison.compare import (
            ComparisonResult,
            ModelComparator,
        )

        def mock_factory(config):
            return None

        comparator = ModelComparator(
            ttm_predictor_factory=mock_factory,
        )

        # Add some results
        comparator.results = [
            ComparisonResult(
                config_name="test1",
                domain="Energy",
                ttm_metrics={"MSE": 0.1},
                tabpfn_metrics={"MSE": 0.2},
                winner="TTM",
            ),
            ComparisonResult(
                config_name="test2",
                domain="Energy",
                ttm_metrics={"MSE": 0.3},
                tabpfn_metrics={"MSE": 0.1},
                winner="TabPFN",
            ),
        ]

        stats = comparator.compute_statistics()
        assert stats["total_comparisons"] == 2
        assert stats["ttm_wins"] == 1
        assert stats["tabpfn_wins"] == 1


class TestTabPFNWrapper:
    """Tests for TabPFN wrapper availability check."""

    def test_tabpfn_availability_check(self):
        """Test TabPFN availability function."""
        from benchmarks.tabpfn_comparison.wrapper import is_tabpfn_available

        # Should return True or False without error
        result = is_tabpfn_available()
        assert isinstance(result, bool)


class TestDatasetLoader:
    """Tests for dataset loader."""

    def test_loader_creation(self, tmp_path):
        """Test GIFTEvalDatasetLoader creation."""
        from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader

        loader = GIFTEvalDatasetLoader(cache_dir=tmp_path)
        assert loader.cache_dir == tmp_path

    def test_create_synthetic_dataset(self, tmp_path):
        """Test synthetic dataset creation."""
        from benchmarks.gift_eval.config import DatasetConfig
        from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader

        loader = GIFTEvalDatasetLoader(cache_dir=tmp_path)

        config = DatasetConfig(
            name="test",
            domain="Test",
            freq="H",
            prediction_length=24,
            context_length=168,
            num_variates=1,
            seasonality=24,
        )

        train, test = loader._create_synthetic_dataset(config)

        # Check datasets are created
        train_list = list(train)
        test_list = list(test)

        assert len(train_list) > 0
        assert len(test_list) > 0

    def test_get_metadata(self, tmp_path):
        """Test getting metadata."""
        from benchmarks.gift_eval.config import DatasetConfig
        from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader

        loader = GIFTEvalDatasetLoader(cache_dir=tmp_path)

        config = DatasetConfig(
            name="test",
            domain="Test",
            freq="H",
            prediction_length=24,
        )

        metadata = loader.get_metadata(config)
        assert metadata["name"] == "test"
        assert metadata["domain"] == "Test"
