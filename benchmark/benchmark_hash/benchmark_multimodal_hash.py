"""
Simple benchmark to compare multimodal feature hashing algorithms.

This benchmark compares the performance of SHA-256 and xxHash algorithms
for multimodal feature hashing in SGLang (used by set_pad_value).

Example:
    python benchmark_multimodal_hash.py --num-features 8 --feature-size 32768 --trials 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import List

import torch

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.utils.hashing import HashAlgorithm


def _generate_feature_data(
    num_features: int, feature_size: int, seed: int
) -> List[torch.Tensor]:
    """Generate random feature data (simulating multimodal features)."""
    torch.manual_seed(seed)
    features = []
    for _ in range(num_features):
        # Generate random float16 tensor (more realistic for multimodal features)
        feature_tensor = torch.randn(feature_size, dtype=torch.float16)
        features.append(feature_tensor)
    return features


def _hash_all_features(
    algorithm: str,
    features: List[torch.Tensor],
) -> float:
    """Hash all features (simulating set_pad_value behavior)."""

    start = time.perf_counter()
    for feature_tensor in features:
        hash_feature(feature_tensor, algorithm=algorithm)
    end = time.perf_counter()
    return end - start


def _benchmark_multimodal_features(
    hash_algo: str,
    features: List[torch.Tensor],
    trials: int,
) -> tuple[float, float, float] | None:
    """Benchmark multimodal feature hashing (used by set_pad_value)."""
    try:
        # Check if algorithm is available
        if not HashAlgorithm.is_available(hash_algo):
            print(f"Skipping {hash_algo}: algorithm not available", file=sys.stderr)
            return None

        timings = [_hash_all_features(hash_algo, features) for _ in range(trials)]
    except Exception as exc:
        print(f"Skipping {hash_algo}: {exc}", file=sys.stderr)
        return None

    avg = statistics.mean(timings)
    best = min(timings)
    # throughput: features / second
    throughput = len(features) / best
    return avg, best, throughput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-features",
        type=int,
        default=10000,
        help="Number of multimodal features to hash.",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=1024,
        help="Size of each feature (number of float16 values).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of timed trials per algorithm."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Hash algorithms to benchmark (default: all available).",
    )
    args = parser.parse_args()

    # Get available algorithms
    available_algorithms = HashAlgorithm.choices()
    if args.algorithms:
        # Validate requested algorithms
        invalid = [a for a in args.algorithms if a not in available_algorithms]
        if invalid:
            print(
                f"Error: Invalid algorithms: {invalid}. "
                f"Available: {available_algorithms}",
                file=sys.stderr,
            )
            sys.exit(1)
        algorithms_to_test = args.algorithms
    else:
        algorithms_to_test = available_algorithms

    # Benchmark multimodal feature hashing
    features = _generate_feature_data(args.num_features, args.feature_size, args.seed)
    print(
        f"\n{'='*70}\n"
        f"Multimodal Feature Hashing Benchmark (set_pad_value)\n"
        f"{'='*70}"
    )
    print(
        f"Benchmarking {len(algorithms_to_test)} algorithm(s) on "
        f"{args.num_features} features (feature size={args.feature_size})."
    )
    feature_size_bytes = features[0].numel() * features[0].element_size()
    print(f"Total data: {args.num_features * feature_size_bytes / 1024 / 1024:.2f} MB")
    print()

    results = []
    for algo in algorithms_to_test:
        result = _benchmark_multimodal_features(algo, features, args.trials)
        if result is None:
            continue

        avg, best, throughput = result
        results.append((algo, avg, best, throughput))
        print(
            f"{algo:14s} avg: {avg:.6f}s  best: {best:.6f}s  "
            f"throughput: {throughput / 1e3:.2f}K features/s"
        )

    # Print speedup comparison if we have multiple results
    if len(results) > 1:
        print()
        print("Speedup comparison:")
        baseline = results[0]
        for algo, avg, best, throughput in results:
            if algo != baseline[0]:
                speedup = baseline[2] / best  # Compare best times
                print(f"  {algo} vs {baseline[0]}: {speedup:.2f}x faster")


if __name__ == "__main__":
    main()
