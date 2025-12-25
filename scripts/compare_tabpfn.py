#!/usr/bin/env python
"""Compare TTM with TabPFN-TS on GIFT-Eval.

Usage:
    python scripts/compare_tabpfn.py --ttm-path models/ttm.pt
    python scripts/compare_tabpfn.py --datasets 10 --output results/comparison
    python scripts/compare_tabpfn.py --domains Energy --metric MAE
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_comparison(args):
    """Run the TTM vs TabPFN comparison."""
    from benchmarks.gift_eval.adapter import load_predictor_from_path
    from benchmarks.gift_eval.config import GIFT_EVAL_DATASETS, DatasetConfig
    from benchmarks.gift_eval.dataset_loader import GIFTEvalDatasetLoader
    from benchmarks.tabpfn_comparison.compare import ModelComparator
    from benchmarks.tabpfn_comparison.wrapper import (
        TabPFNGluonTSPredictor,
        is_tabpfn_available,
    )

    logger = logging.getLogger(__name__)

    # Check TabPFN availability
    if not is_tabpfn_available():
        logger.error(
            "TabPFN-TS is not installed. "
            "Install with: pip install tabpfn-time-series"
        )
        sys.exit(1)

    # Select datasets
    configs = GIFT_EVAL_DATASETS

    if args.domains:
        configs = [c for c in configs if c.domain in args.domains]

    if args.datasets:
        configs = configs[: args.datasets]

    logger.info(f"Comparing on {len(configs)} datasets")

    # Create predictor factories
    def ttm_predictor_factory(config: DatasetConfig):
        return load_predictor_from_path(
            model_path=args.ttm_path,
            prediction_length=config.prediction_length,
            freq=config.freq,
            context_length=min(args.context_length, config.context_length),
            device=args.device,
            predictor_type="zero_shot",
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

    def tabpfn_predictor_factory(config: DatasetConfig):
        return TabPFNGluonTSPredictor(
            prediction_length=config.prediction_length,
            freq=config.freq,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

    # Initialize comparator
    comparator = ModelComparator(
        ttm_predictor_factory=ttm_predictor_factory,
        tabpfn_predictor_factory=tabpfn_predictor_factory,
        output_dir=args.output,
        primary_metric=args.metric,
    )

    # Initialize dataset loader
    dataset_loader = GIFTEvalDatasetLoader()

    # Run comparison
    logger.info("Starting TTM vs TabPFN comparison...")
    comparator.compare_all(configs, dataset_loader)

    # Save results
    comparator.save_results()

    # Print summary
    stats = comparator.compute_statistics()

    print("\n" + "=" * 60)
    print("Comparison Complete")
    print("=" * 60)
    print(f"Total comparisons: {stats['total_comparisons']}")
    print(f"TTM wins: {stats['ttm_wins']}")
    print(f"TabPFN wins: {stats['tabpfn_wins']}")
    print(f"Ties: {stats['ties']}")
    print(f"TTM win rate: {stats['ttm_win_rate']:.2%}")

    # Print average metrics
    print("\nAverage Metrics:")
    for metric in ["MSE", "MAE", "RMSE"]:
        ttm_avg = stats.get(f"ttm_avg_{metric}", "N/A")
        tabpfn_avg = stats.get(f"tabpfn_avg_{metric}", "N/A")
        if isinstance(ttm_avg, float):
            print(f"  {metric}: TTM={ttm_avg:.4f}, TabPFN={tabpfn_avg:.4f}")

    # Statistical significance
    sig = comparator.compute_statistical_significance(args.metric)
    if "error" not in sig:
        print(f"\nStatistical Significance ({args.metric}):")
        print(f"  Paired t-test p-value: {sig['paired_ttest_pvalue']:.4f}")
        print(f"  Significant at 0.05: {sig['significant_at_0.05']}")

    print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="TTM vs TabPFN-TS Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_tabpfn.py --ttm-path models/ttm.pt
  python scripts/compare_tabpfn.py --datasets 10 --output results/comparison
  python scripts/compare_tabpfn.py --domains Energy --metric MAE
        """,
    )

    parser.add_argument(
        "--ttm-path",
        type=str,
        required=True,
        help="Path to TTM model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory (default: results/comparison)",
    )
    parser.add_argument(
        "--datasets",
        type=int,
        default=None,
        help="Number of datasets to compare (None for all)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Filter by domains (e.g., Energy Web)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="MSE",
        choices=["MSE", "MAE", "RMSE", "MASE"],
        help="Primary comparison metric (default: MSE)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length for predictions (default: 512)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for probabilistic forecasts (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    run_comparison(args)


if __name__ == "__main__":
    main()
