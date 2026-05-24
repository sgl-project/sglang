#!/usr/bin/env python3
"""Correctness and benchmark test for the greedy TP AG+argmax shortcut.

Run on AMD/ROCm with tensor parallel ranks, for example:

    SGLANG_AITER_AG_ARGMAX_SHORTCUT=1 \
    PYTHONPATH=/sgl-workspace/sglang/python \
    torchrun --nproc_per_node=4 --master_port=29510 \
      test/srt/test_greedy_argmax_shortcut.py --benchmark

The correctness test compares the shortcut against the baseline
`tensor_model_parallel_all_gather(local_logits, dim=-1).argmax(dim=-1)`,
including deterministic cross-rank tie cases.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


def _init_dist() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA/HIP-visible device.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size
        )

    import sglang.srt.distributed.parallel_state as ps

    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    ps.initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, world_size, local_rank, device


def _time_eager(fn: Callable[[], None], iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device=device)
    return start.elapsed_time(end) / iters


def _time_graph(fn: Callable[[], None], iters: int, warmup: int, device: torch.device) -> float:
    from sglang.srt.distributed import graph_capture

    with graph_capture() as gc:
        stream = gc.stream
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.synchronize(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device=device)
    return start.elapsed_time(end) / iters


def _check_correctness(
    m_values: list[int],
    v_local: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    from sglang.srt.distributed import tensor_model_parallel_all_gather
    from sglang.srt.layers.logits_processor import _fused_greedy_argmax_across_tp
    from sglang.srt.utils import is_hip

    if not is_hip():
        if rank == 0:
            print("SKIP: greedy shortcut is AMD/HIP-only; non-HIP routes are untouched.")
        return

    for m in m_values:
        torch.manual_seed(20260515 + rank * 17 + m)
        local = torch.randn(m, v_local, dtype=dtype, device=device)
        ref_full = tensor_model_parallel_all_gather(local, dim=-1)
        ref_ids = torch.argmax(ref_full, dim=-1).to(torch.int64)
        got_ids = _fused_greedy_argmax_across_tp(local).to(torch.int64)
        assert torch.equal(ref_ids, got_ids), (
            f"random correctness mismatch: M={m}, rank={rank}, "
            f"ref={ref_ids[:8]}, got={got_ids[:8]}"
        )

        # Tie case: every rank has the same winning value at local index 0.
        # torch.argmax over concatenated shards should pick rank 0, index 0.
        tied = torch.zeros(m, v_local, dtype=dtype, device=device)
        tied[:, 0] = 1.0
        ref_full = tensor_model_parallel_all_gather(tied, dim=-1)
        ref_ids = torch.argmax(ref_full, dim=-1).to(torch.int64)
        got_ids = _fused_greedy_argmax_across_tp(tied).to(torch.int64)
        expected = torch.zeros(m, dtype=torch.int64, device=device)
        assert torch.equal(ref_ids, expected)
        assert torch.equal(got_ids, expected), (
            f"tie-break mismatch: M={m}, rank={rank}, got={got_ids[:8]}"
        )

    if rank == 0:
        print(
            f"PASS correctness: M={m_values}, V_local={v_local}, "
            f"TP={world_size}, dtype={dtype}"
        )


def _run_benchmark(
    m_values: list[int],
    v_local: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    iters: int,
    warmup: int,
    csv_out: str | None,
    graph: bool,
) -> None:
    from sglang.srt.distributed import tensor_model_parallel_all_gather
    from sglang.srt.layers.logits_processor import _fused_greedy_argmax_across_tp
    from sglang.srt.utils import is_hip

    if not is_hip():
        if rank == 0:
            print("SKIP benchmark: greedy shortcut is AMD/HIP-only.")
        return

    rows = []
    for m in m_values:
        torch.manual_seed(4242 + rank * 31 + m)
        local = torch.randn(m, v_local, dtype=dtype, device=device)

        def baseline() -> None:
            full = tensor_model_parallel_all_gather(local, dim=-1)
            _ = torch.argmax(full, dim=-1)

        def shortcut() -> None:
            _ = _fused_greedy_argmax_across_tp(local)

        base_eager = _time_eager(baseline, iters, warmup, device)
        short_eager = _time_eager(shortcut, iters, warmup, device)
        base_graph = short_graph = float("nan")
        if graph:
            base_graph = _time_graph(baseline, iters, warmup, device)
            short_graph = _time_graph(shortcut, iters, warmup, device)

        stat = torch.tensor(
            [base_eager, short_eager, base_graph, short_graph],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(stat, op=dist.ReduceOp.AVG)
        be, se, bg, sg = stat.tolist()
        rows.append(
            {
                "M": m,
                "V_local": v_local,
                "TP": world_size,
                "dtype": str(dtype).replace("torch.", ""),
                "base_eager_ms": be,
                "shortcut_eager_ms": se,
                "eager_speedup": be / se if se > 0 else float("inf"),
                "base_graph_ms": bg,
                "shortcut_graph_ms": sg,
                "graph_speedup": bg / sg if graph and sg > 0 else float("nan"),
            }
        )

    if rank == 0:
        print(
            f"{'M':>6} {'base_eager':>12} {'shortcut':>12} {'eager_x':>9} "
            f"{'base_graph':>12} {'shortcut_g':>12} {'graph_x':>9}"
        )
        for row in rows:
            print(
                f"{row['M']:>6} "
                f"{row['base_eager_ms']:>12.4f} {row['shortcut_eager_ms']:>12.4f} "
                f"{row['eager_speedup']:>8.2f}x "
                f"{row['base_graph_ms']:>12.4f} {row['shortcut_graph_ms']:>12.4f} "
                f"{row['graph_speedup']:>8.2f}x"
            )
        if csv_out:
            out = Path(csv_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote benchmark CSV: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,2,8,16,32,64,128")
    parser.add_argument("--v-local", type=int, default=31040)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    m_values = [int(v) for v in args.m_values.split(",") if v.strip()]

    rank, world_size, _, device = _init_dist()
    try:
        _check_correctness(m_values, args.v_local, dtype, rank, world_size, device)
        if args.benchmark:
            _run_benchmark(
                m_values=m_values,
                v_local=args.v_local,
                dtype=dtype,
                rank=rank,
                world_size=world_size,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
                csv_out=args.csv_out or None,
                graph=not args.no_graph,
            )
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
