import argparse
import statistics
import time
from collections.abc import Callable

import torch

from sglang.srt.utils.common import get_num_new_pages


def _legacy_count(seq_lens: torch.Tensor, page_size: int) -> int:
    return (seq_lens % page_size == 1).int().sum().item()


def _current_count(seq_lens: torch.Tensor, page_size: int) -> int:
    return get_num_new_pages(seq_lens, page_size=page_size, decode=True)


def _measure_us(
    fn: Callable[[torch.Tensor, int], int],
    seq_lens: torch.Tensor,
    page_size: int,
    iterations: int,
    repeats: int,
) -> float:
    samples = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        for _ in range(iterations):
            fn(seq_lens, page_size)
        samples.append((time.perf_counter_ns() - start_ns) / iterations / 1_000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    print(f"{'batch':>7} {'legacy_us':>12} {'current_us':>12} {'speedup':>10}")
    for batch_size in args.batch_sizes:
        seq_lens = torch.arange(1, batch_size + 1, dtype=torch.int64)
        expected = _legacy_count(seq_lens, args.page_size)
        actual = _current_count(seq_lens, args.page_size)
        if actual != expected:
            raise RuntimeError(
                f"page count mismatch for batch={batch_size}: {actual} != {expected}"
            )

        for _ in range(100):
            _legacy_count(seq_lens, args.page_size)
            _current_count(seq_lens, args.page_size)

        legacy_us = _measure_us(
            _legacy_count,
            seq_lens,
            args.page_size,
            args.iterations,
            args.repeats,
        )
        current_us = _measure_us(
            _current_count,
            seq_lens,
            args.page_size,
            args.iterations,
            args.repeats,
        )
        print(
            f"{batch_size:7d} {legacy_us:12.3f} {current_us:12.3f} "
            f"{legacy_us / current_us:9.2f}x"
        )


if __name__ == "__main__":
    main()
