"""Benchmark multimodal embedding scatter variants on Ascend NPU.

This compares the two implementations used by mm_utils._scatter:

1. masked_scatter_: dest.masked_scatter_(mask.expand_as(dest), src)
2. nonzero + index_copy_: indices = nonzero(mask); dest.index_copy_(0, indices, src)

Example:
    python3 benchmark/kernels/benchmark_npu_mm_scatter.py \
        --seq-lens 1024,2048,6102,8192 \
        --mm-tokens 324,1296,2592,5184 \
        --hidden-size 8192 \
        --repeat 50 --warmup 10
"""

import argparse
import csv
import statistics
import sys
import time
from typing import Callable, List

import torch


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def npu_sync() -> None:
    torch.npu.synchronize()


def make_row_indices(seq_len: int, mm_tokens: int, pattern: str, device: str):
    if pattern == "prefix":
        return torch.arange(mm_tokens, device=device, dtype=torch.long)
    if pattern == "suffix":
        return torch.arange(
            seq_len - mm_tokens, seq_len, device=device, dtype=torch.long
        )
    if pattern == "middle":
        start = (seq_len - mm_tokens) // 2
        return torch.arange(start, start + mm_tokens, device=device, dtype=torch.long)
    if pattern == "interleaved":
        step = max(seq_len // mm_tokens, 1)
        indices = torch.arange(
            0, step * mm_tokens, step, device=device, dtype=torch.long
        )
        return torch.clamp(indices, max=seq_len - 1)
    if pattern == "random":
        return torch.randperm(seq_len, device=device)[:mm_tokens].sort().values
    raise ValueError(f"Unsupported pattern: {pattern}")


def make_case(
    seq_len: int, hidden_size: int, mm_tokens: int, dtype, pattern: str, device: str
):
    dest = torch.randn((seq_len, hidden_size), device=device, dtype=dtype)
    src = torch.randn((mm_tokens, hidden_size), device=device, dtype=dtype)
    mask = torch.zeros((seq_len, 1), device=device, dtype=torch.bool)
    row_indices = make_row_indices(seq_len, mm_tokens, pattern, device)
    mask.index_fill_(0, row_indices, True)
    return dest, src, mask


def bench_op(fn: Callable[[], None], warmup: int, repeat: int) -> List[float]:
    for _ in range(warmup):
        fn()
    npu_sync()

    times = []
    for _ in range(repeat):
        npu_sync()
        start = time.perf_counter()
        fn()
        npu_sync()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarize(times: List[float]):
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def print_table_header() -> None:
    print(
        f"{'seq_len':>8} {'mm_tok':>8} {'hidden':>8} {'dtype':>5} {'pattern':>11} "
        f"{'masked mean':>13} {'masked med':>12} {'masked min':>12} {'masked max':>12} "
        f"{'index mean':>12} {'index med':>11} {'index min':>11} {'index max':>11} "
        f"{'speedup':>9}"
    )
    print("-" * 161)


def print_table_row(
    seq_len: int,
    mm_tokens: int,
    hidden_size: int,
    dtype: str,
    pattern: str,
    masked,
    indexed,
    speedup: float,
) -> None:
    print(
        f"{seq_len:8d} {mm_tokens:8d} {hidden_size:8d} {dtype:>5} {pattern:>11} "
        f"{masked['mean']:13.3f} {masked['median']:12.3f} "
        f"{masked['min']:12.3f} {masked['max']:12.3f} "
        f"{indexed['mean']:12.3f} {indexed['median']:11.3f} "
        f"{indexed['min']:11.3f} {indexed['max']:11.3f} "
        f"{speedup:9.3f}"
    )


def check_correctness(dest, src, mask) -> None:
    masked = dest.clone()
    indexed = dest.clone()
    masked.masked_scatter_(mask.expand_as(masked), src)
    row_indices = torch.nonzero(mask.squeeze(-1), as_tuple=False).flatten()
    indexed.index_copy_(0, row_indices, src)
    npu_sync()
    if not torch.equal(masked, indexed):
        max_diff = (masked.float() - indexed.float()).abs().max().item()
        raise AssertionError(f"Correctness check failed: max_diff={max_diff}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NPU multimodal scatter variants."
    )
    parser.add_argument("--seq-lens", default="1024,2048,4096,6102,8192,16384")
    parser.add_argument("--mm-tokens", default="324,1296,2592,5184")
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--pattern",
        choices=["prefix", "suffix", "middle", "interleaved", "random"],
        default="middle",
        help="Layout of multimodal token rows in the sequence.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--csv", action="store_true", help="Print CSV instead of an aligned table."
    )
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires torch_npu and an Ascend NPU runtime."
        ) from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available.")

    seq_lens = parse_int_list(args.seq_lens)
    mm_tokens_list = parse_int_list(args.mm_tokens)
    dtype = get_dtype(args.dtype)
    device = args.device

    csv_writer = None
    if args.csv:
        csv_writer = csv.writer(sys.stdout)
        csv_writer.writerow(
            [
                "seq_len",
                "mm_tokens",
                "hidden",
                "dtype",
                "pattern",
                "masked_mean_ms",
                "masked_median_ms",
                "masked_min_ms",
                "masked_max_ms",
                "index_mean_ms",
                "index_median_ms",
                "index_min_ms",
                "index_max_ms",
                "speedup_masked_over_index",
            ]
        )
    else:
        print_table_header()

    for seq_len in seq_lens:
        for mm_tokens in mm_tokens_list:
            if mm_tokens > seq_len:
                continue

            dest, src, mask = make_case(
                seq_len=seq_len,
                hidden_size=args.hidden_size,
                mm_tokens=mm_tokens,
                dtype=dtype,
                pattern=args.pattern,
                device=device,
            )

            if args.check:
                check_correctness(dest, src, mask)

            def run_masked_scatter():
                dest.masked_scatter_(mask.expand_as(dest), src)

            def run_nonzero_index_copy():
                row_indices = torch.nonzero(mask.squeeze(-1), as_tuple=False).flatten()
                dest.index_copy_(0, row_indices, src)

            masked = summarize(bench_op(run_masked_scatter, args.warmup, args.repeat))
            indexed = summarize(
                bench_op(run_nonzero_index_copy, args.warmup, args.repeat)
            )
            speedup = (
                masked["mean"] / indexed["mean"]
                if indexed["mean"] > 0
                else float("inf")
            )

            if args.csv:
                csv_writer.writerow(
                    [
                        seq_len,
                        mm_tokens,
                        args.hidden_size,
                        args.dtype,
                        args.pattern,
                        format_ms(masked["mean"]),
                        format_ms(masked["median"]),
                        format_ms(masked["min"]),
                        format_ms(masked["max"]),
                        format_ms(indexed["mean"]),
                        format_ms(indexed["median"]),
                        format_ms(indexed["min"]),
                        format_ms(indexed["max"]),
                        format_ms(speedup),
                    ]
                )
            else:
                print_table_row(
                    seq_len,
                    mm_tokens,
                    args.hidden_size,
                    args.dtype,
                    args.pattern,
                    masked,
                    indexed,
                    speedup,
                )


if __name__ == "__main__":
    main()
