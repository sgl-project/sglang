"""Local CUDA-graph benchmark for DSV4 live-prefix attention metadata.

This benchmark is intentionally not registered in CI. It compares the retained
full-tail Triton path (``live_prefix_only=False``) with the target-verify
live-prefix path from the same checkout across request batch sizes:

    python3 python/sglang/kernels/jit/benchmark/bench_dsv4_live_prefix_metadata.py

The comparison is conservative because the retained compressed-metadata
kernel already includes the new live-length mask on its page-table loads.
Timings include the production C128 alignment pad and CUDA-graph replay
overhead. Each request contributes ``VERIFY_WIDTH`` causal metadata rows.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildPageTablePositions,
)
from sglang.srt.layers.attention.deepseek_v4_backend import _pad_last_dim

PAGE_SIZE = 256
SWA_WINDOW = 128
VERIFY_WIDTH = 4
MAX_CONTEXT_LEN = 1 << 20
# Match the context-plus-sentinel request-token capacity in the server profile.
CAPTURE_SEQ_LEN = MAX_CONTEXT_LEN + 1
DEFAULT_MAX_SEQ_LENS = (1 << 10, 1 << 12, 1 << 15, MAX_CONTEXT_LEN)
DEFAULT_BATCH_SIZES = (1, 4, 8, 16, 32)


class Inputs(msgspec.Struct):
    req_to_token: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    raw_out_loc: torch.Tensor


class Captured(msgspec.Struct):
    graph: torch.cuda.CUDAGraph
    output: object


def make_inputs(max_seq_len: int, batch_size: int) -> Inputs:
    if not VERIFY_WIDTH <= max_seq_len <= MAX_CONTEXT_LEN:
        raise ValueError(
            f"max_seq_len must be in [{VERIFY_WIDTH}, {MAX_CONTEXT_LEN}], "
            f"got {max_seq_len}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Every target-verify request produces VERIFY_WIDTH consecutive causal rows.
    causal_seq_lens = torch.arange(
        max_seq_len - VERIFY_WIDTH + 1,
        max_seq_len + 1,
        dtype=torch.int32,
        device="cuda",
    )
    req_to_token = torch.arange(
        CAPTURE_SEQ_LEN, dtype=torch.int32, device="cuda"
    ).unsqueeze(0) + (
        torch.arange(batch_size, dtype=torch.int32, device="cuda").unsqueeze(1)
        * CAPTURE_SEQ_LEN
    )
    num_rows = batch_size * VERIFY_WIDTH
    return Inputs(
        # Give each request a disjoint identity-mapped physical-token range.
        req_to_token=req_to_token,
        req_pool_indices=torch.arange(
            batch_size, dtype=torch.int64, device="cuda"
        ).repeat_interleave(VERIFY_WIDTH),
        seq_lens=causal_seq_lens.repeat(batch_size),
        raw_out_loc=(
            torch.arange(1, num_rows + 1, dtype=torch.int64, device="cuda") * 128
        ),
    )


def run_metadata(inputs: Inputs, *, live_prefix_only: bool):
    page = BuildPageTablePositions.triton(
        req_to_token=inputs.req_to_token,
        req_pool_indices_repeated=inputs.req_pool_indices,
        seq_lens_casual=inputs.seq_lens,
        max_seq_len=CAPTURE_SEQ_LEN,
        page_size=PAGE_SIZE,
        swa_window=SWA_WINDOW,
        live_prefix_only=live_prefix_only,
    )
    compressed = init_compression_metadata(
        page.seq_lens_casual,
        page.positions_casual,
        inputs.raw_out_loc,
        page.page_table,
        PAGE_SIZE,
        compute_page_indices=True,
        live_prefix_only=live_prefix_only,
    )
    compressed = (*compressed[:-1], _pad_last_dim(compressed[-1]))
    return page, compressed


def check_live_prefixes(inputs: Inputs) -> None:
    full_page, full_compressed = run_metadata(inputs, live_prefix_only=False)
    live_page, live_compressed = run_metadata(inputs, live_prefix_only=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(live_page.seq_lens_casual, full_page.seq_lens_casual)
    torch.testing.assert_close(live_page.positions_casual, full_page.positions_casual)
    torch.testing.assert_close(live_page.swa_topk_lengths, full_page.swa_topk_lengths)
    for live, full in zip(live_compressed[:8], full_compressed[:8]):
        torch.testing.assert_close(live, full)

    for row, seq_len in enumerate(inputs.seq_lens.cpu().tolist()):
        live_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        torch.testing.assert_close(
            live_page.page_table[row, :live_pages],
            full_page.page_table[row, :live_pages],
        )
        live_c128 = max(seq_len // 128, 1)
        torch.testing.assert_close(
            live_compressed[8][row, :live_c128],
            full_compressed[8][row, :live_c128],
        )


def capture_one(fn: Callable[[], object], stream: torch.cuda.Stream) -> Captured:
    """Compile and capture exactly one allocation-returning invocation."""
    with torch.cuda.stream(stream):
        warm_output = fn()
    stream.synchronize()
    del warm_output

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return Captured(graph=graph, output=output)


def bench_cuda_graph_pair(
    full_tail_fn: Callable[[], object],
    live_prefix_fn: Callable[[], object],
    *,
    warmup_iters: int,
    replay_iters: int,
) -> tuple[float, float]:
    """Return paired p50 microseconds for one replay of each captured graph.

    Each graph owns one fixed set of output buffers, matching serving. Old and
    new replays are alternated so both measurements see the same GPU clock and
    thermal conditions.
    """
    stream = torch.cuda.Stream()
    full_tail = capture_one(full_tail_fn, stream)
    live_prefix = capture_one(live_prefix_fn, stream)

    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            full_tail.graph.replay()
            live_prefix.graph.replay()
    stream.synchronize()

    full_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(replay_iters)
    ]
    live_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(replay_iters)
    ]

    def record(events, captured):
        start, end = events
        start.record(stream)
        captured.graph.replay()
        end.record(stream)

    with torch.cuda.stream(stream):
        for i, (full_event, live_event) in enumerate(zip(full_events, live_events)):
            if i % 2 == 0:
                record(full_event, full_tail)
                record(live_event, live_prefix)
            else:
                record(live_event, live_prefix)
                record(full_event, full_tail)
    stream.synchronize()

    def median_us(events) -> float:
        return statistics.median(
            start.elapsed_time(end) * 1000 for start, end in events
        )

    # Captured.output keeps allocator-backed graph buffers alive through here.
    return (
        median_us(full_events),
        median_us(live_events),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-seq-lens",
        type=int,
        nargs="+",
        default=DEFAULT_MAX_SEQ_LENS,
        help="Maximum length among the four causal verify rows.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Logical target-verify request batch sizes.",
    )
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--replay-iters", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    capability = torch.cuda.get_device_capability()
    print(
        f"device={torch.cuda.get_device_name()} sm={capability[0]}{capability[1]} "
        f"capture_seq_len={CAPTURE_SEQ_LEN} verify_width={VERIFY_WIDTH}"
    )
    if capability[0] != 10:
        print("warning: production live-prefix dispatch is currently gated to SM100")

    header = (
        f"{'batch_size':>10}  {'kernel_rows':>11}  {'max_seq_len':>12}  "
        f"{'full_tail_p50_us':>16}  {'live_prefix_p50_us':>18}  "
        f"{'saved_us':>10}  {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))

    for batch_size in args.batch_sizes:
        for max_seq_len in args.max_seq_lens:
            inputs = make_inputs(max_seq_len, batch_size)
            check_live_prefixes(inputs)
            full_tail_us, live_prefix_us = bench_cuda_graph_pair(
                lambda: run_metadata(inputs, live_prefix_only=False),
                lambda: run_metadata(inputs, live_prefix_only=True),
                warmup_iters=args.warmup_iters,
                replay_iters=args.replay_iters,
            )
            saved_us = full_tail_us - live_prefix_us
            print(
                f"{batch_size:10d}  {batch_size * VERIFY_WIDTH:11d}  "
                f"{max_seq_len:12d}  {full_tail_us:16.3f}  "
                f"{live_prefix_us:18.3f}  {saved_us:10.3f}  "
                f"{full_tail_us / live_prefix_us:8.2f}x"
            )


if __name__ == "__main__":
    main()
