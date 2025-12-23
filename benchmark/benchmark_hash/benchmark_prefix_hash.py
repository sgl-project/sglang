"""
Simple benchmark to compare prefix-cache hashing algorithms.

This benchmark compares the performance of SHA-256 and xxHash algorithms
for prefix caching in SGLang. It includes two scenarios:
1. Prefix cache token hashing (get_hash_str)
2. Multimodal feature hashing (hash_bytes_to_int64, used by set_pad_value)

Example:
    python benchmark_prefix_hash.py --num-blocks 20000 --block-size 32
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from typing import List, Optional

import numpy as np

from sglang.srt.utils.hashing import HashAlgorithm, get_hash_str, hash_bytes_to_int64


def _generate_blocks(
    num_blocks: int, block_size: int, vocab_size: int, seed: int
) -> List[List[int]]:
    """Generate random token blocks for benchmarking."""
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(block_size)]
        for _ in range(num_blocks)
    ]


def _hash_all_blocks(
    algorithm: str,
    blocks: List[List[int]],
) -> float:
    """Hash all blocks with chaining (simulating prefix cache behavior)."""
    parent_hash: Optional[str] = None
    start = time.perf_counter()
    for block in blocks:
        parent_hash = get_hash_str(block, prior_hash=parent_hash, algorithm=algorithm)
    end = time.perf_counter()
    return end - start


def _generate_feature_data(
    num_features: int, feature_size: int, seed: int
) -> List[bytes]:
    """Generate random feature data (simulating multimodal features)."""
    rng = random.Random(seed)
    features = []
    for _ in range(num_features):
        # Generate random float16 array and convert to bytes
        feature_array = np.random.rand(feature_size).astype(np.float16)
        features.append(feature_array.tobytes())
    return features


def _hash_all_features(
    algorithm: str,
    features: List[bytes],
) -> float:
    """Hash all features (simulating set_pad_value behavior)."""
    start = time.perf_counter()
    for feature_bytes in features:
        hash_bytes_to_int64(feature_bytes, algorithm=algorithm)
    end = time.perf_counter()
    return end - start


def _benchmark_prefix_cache(
    hash_algo: str,
    blocks: List[List[int]],
    trials: int,
) -> tuple[float, float, float] | None:
    """Benchmark prefix cache token hashing."""
    try:
        # Check if algorithm is available
        if not HashAlgorithm.is_available(hash_algo):
            print(f"Skipping {hash_algo}: algorithm not available", file=sys.stderr)
            return None

        timings = [_hash_all_blocks(hash_algo, blocks) for _ in range(trials)]
    except Exception as exc:
        print(f"Skipping {hash_algo}: {exc}", file=sys.stderr)
        return None

    avg = statistics.mean(timings)
    best = min(timings)
    # throughput: tokens / second
    tokens_hashed = len(blocks) * len(blocks[0])
    throughput = tokens_hashed / best
    return avg, best, throughput


def _benchmark_multimodal_features(
    hash_algo: str,
    features: List[bytes],
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
        "--num-blocks", type=int, default=10000, help="Number of blocks to hash."
    )
    parser.add_argument("--block-size", type=int, default=32, help="Tokens per block.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Token id range [0, vocab_size).",
    )
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
    parser.add_argument(
        "--benchmark-type",
        choices=["prefix", "multimodal", "both"],
        default="both",
        help="Type of benchmark to run: 'prefix' (token hashing), "
        "'multimodal' (feature hashing), or 'both' (default).",
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

    # Benchmark prefix cache hashing
    if args.benchmark_type in ["prefix", "both"]:
        blocks = _generate_blocks(
            args.num_blocks, args.block_size, args.vocab_size, args.seed
        )
        print(f"\n{'='*70}\n" f"Prefix Cache Token Hashing Benchmark\n" f"{'='*70}")
        print(
            f"Benchmarking {len(algorithms_to_test)} algorithm(s) on "
            f"{args.num_blocks} blocks (block size={args.block_size})."
        )
        print(f"Total tokens: {args.num_blocks * args.block_size:,}")
        print()

        results = []
        for algo in algorithms_to_test:
            result = _benchmark_prefix_cache(algo, blocks, args.trials)
            if result is None:
                continue

            avg, best, throughput = result
            results.append((algo, avg, best, throughput))
            print(
                f"{algo:14s} avg: {avg:.6f}s  best: {best:.6f}s  "
                f"throughput: {throughput / 1e6:.2f}M tokens/s"
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

    # Benchmark multimodal feature hashing
    if args.benchmark_type in ["multimodal", "both"]:
        features = _generate_feature_data(
            args.num_features, args.feature_size, args.seed
        )
        print(
            f"\n{'='*70}\n"
            f"Multimodal Feature Hashing Benchmark (set_pad_value)\n"
            f"{'='*70}"
        )
        print(
            f"Benchmarking {len(algorithms_to_test)} algorithm(s) on "
            f"{args.num_features} features (feature size={args.feature_size})."
        )
        feature_bytes = len(features[0])
        print(f"Total data: {args.num_features * feature_bytes / 1024 / 1024:.2f} MB")
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
