#!/usr/bin/env python
"""Run GIFT-Eval benchmark for TinyTimeMixers.

Usage:
    python scripts/benchmark_gift.py --model-path models/ttm.pt
    python scripts/benchmark_gift.py --domains Energy Web --output results/ttm
    python scripts/benchmark_gift.py --resume
    python scripts/benchmark_gift.py --list-datasets
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


def list_datasets():
    """List all available GIFT-Eval datasets."""
    from benchmarks.gift_eval.config import DOMAINS, GIFT_EVAL_DATASETS

    print("\nGIFT-Eval Datasets")
    print("=" * 60)

    for domain in DOMAINS:
        configs = [c for c in GIFT_EVAL_DATASETS if c.domain == domain]
        print(f"\n{domain} ({len(configs)} configurations):")
        for config in configs:
            print(
                f"  - {config.config_name}: "
                f"pred_len={config.prediction_length}, "
                f"variates={config.num_variates}"
            )

    print(f"\nTotal: {len(GIFT_EVAL_DATASETS)} configurations")


def run_benchmark(args):
    """Run the GIFT-Eval benchmark."""
    from benchmarks.gift_eval.adapter import load_predictor_from_path
    from benchmarks.gift_eval.config import DatasetConfig
    from benchmarks.gift_eval.results import ModelMetadata
    from benchmarks.gift_eval.runner import GIFTEvalRunner
    from tinytimemixers.models.ttm import TTM

    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = TTM.load(args.model_path, map_location=args.device)

    # Create model metadata
    model_metadata = ModelMetadata(
        model_name="TTM",
        model_version="1.0.0",
        num_parameters=model.get_num_parameters(),
        context_length=args.context_length,
        github_url="https://github.com/your-repo/TinyTimeMixers",
    )

    # Create predictor factory
    def predictor_factory(config: DatasetConfig):
        return load_predictor_from_path(
            model_path=args.model_path,
            prediction_length=config.prediction_length,
            freq=config.freq,
            context_length=min(args.context_length, config.context_length),
            device=args.device,
            predictor_type="zero_shot",
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

    # Initialize runner
    runner = GIFTEvalRunner(
        predictor_factory=predictor_factory,
        output_dir=args.output,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        device=args.device,
    )

    # Filter datasets if needed
    domains = args.domains if args.domains else None

    # Run benchmark
    logger.info("Starting GIFT-Eval benchmark...")
    runner.run_all(domains=domains)

    # Save results
    runner.save_results(model_metadata)

    # Print summary
    aggregator = runner.get_aggregator(model_metadata)
    summary = aggregator.compute_summary_statistics()

    print("\n" + "=" * 60)
    print("GIFT-Eval Benchmark Complete")
    print("=" * 60)
    print(f"Total datasets: {summary['num_datasets']}")
    print(f"Successful: {summary['num_successful']}")
    print(f"Failed: {summary['num_failed']}")
    print(f"Total runtime: {summary['total_runtime_seconds']:.2f}s")
    print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="GIFT-Eval Benchmark for TinyTimeMixers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_gift.py --model-path models/ttm.pt
  python scripts/benchmark_gift.py --domains Energy Web --output results/ttm
  python scripts/benchmark_gift.py --resume
  python scripts/benchmark_gift.py --list-datasets
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to pre-trained TTM model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ttm",
        help="Output directory for results (default: results/ttm)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Specific domains to benchmark (e.g., Energy Web)",
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
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N datasets (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.list_datasets:
        list_datasets()
        return

    if not args.model_path:
        parser.error("--model-path is required unless using --list-datasets")

    run_benchmark(args)


if __name__ == "__main__":
    main()
