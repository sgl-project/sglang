from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.dsa.triton_paged_mqa_logits_sm80 import (
    triton_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    torch_paged_mqa_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="nightly", runner_config="1-gpu-large")

PAGE_SIZE = 64
HEAD_DIM = 128


@dataclass
class Result:
    batch_queries: int
    heads: int
    seq_len: int
    output_mib: float
    torch_first_ms: float
    triton_first_ms: float
    torch_steady_ms: float
    triton_steady_ms: float
    speedup: float
    torch_peak_alloc_mib: float
    triton_peak_alloc_mib: float
    torch_peak_temp_mib: float
    triton_peak_temp_mib: float


def _pack(keys: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    pages = keys.shape[0]
    packed = torch.empty(
        pages,
        PAGE_SIZE * (HEAD_DIM + 4),
        dtype=torch.uint8,
        device=keys.device,
    )
    split = PAGE_SIZE * HEAD_DIM
    packed[:, :split] = keys.view(torch.uint8).reshape(pages, -1)
    packed[:, split:] = scales.contiguous().view(torch.uint8).reshape(pages, -1)
    return packed.view(pages, PAGE_SIZE, 1, HEAD_DIM + 4)


def _make_case(batch_queries: int, heads: int, seq_len: int):
    torch.manual_seed(20260819 + batch_queries + heads + seq_len)
    pages = triton.cdiv(seq_len, PAGE_SIZE)
    q = torch.randn(batch_queries, heads, HEAD_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    keys = torch.randn(pages, PAGE_SIZE, HEAD_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pages, PAGE_SIZE, device="cuda") + 0.25
    cache = _pack(keys, scales)
    weights = torch.randn(batch_queries, heads, device="cuda")
    seq_lens = torch.full((batch_queries,), seq_len, dtype=torch.int32, device="cuda")
    page_table = (
        torch.arange(pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_queries, -1)
        .contiguous()
    )
    return q, cache, weights, seq_lens, page_table


def _first_call_ms(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    del output
    return elapsed


def _steady_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        output = fn()
        del output
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = fn()
        del output
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _peak_alloc_bytes(fn, output_bytes: int) -> tuple[int, int]:
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak_delta = max(0, torch.cuda.max_memory_allocated() - baseline)
    del output
    torch.cuda.synchronize()
    return peak_delta, max(0, peak_delta - output_bytes)


def _benchmark_shape(
    batch_queries: int,
    heads: int,
    seq_len: int,
    *,
    torch_repeats: int,
    triton_repeats: int,
) -> Result:
    q, cache, weights, seq_lens, page_table = _make_case(batch_queries, heads, seq_len)

    def run_torch():
        return torch_paged_mqa_logits(q, cache, weights, seq_lens, page_table, seq_len)

    def run_triton():
        return triton_paged_mqa_logits(q, cache, weights, seq_lens, page_table, seq_len)

    # This first Triton invocation includes JIT/cache lookup and compilation.
    # Later shapes with the same H may reuse its compiled specialization.
    triton_first_ms = _first_call_ms(run_triton)
    torch_first_ms = _first_call_ms(run_torch)

    triton_steady_ms = _steady_ms(run_triton, warmup=5, repeats=triton_repeats)
    torch_steady_ms = _steady_ms(run_torch, warmup=2, repeats=torch_repeats)

    output_bytes = batch_queries * seq_len * torch.float32.itemsize
    triton_peak, triton_temp = _peak_alloc_bytes(run_triton, output_bytes)
    torch_peak, torch_temp = _peak_alloc_bytes(run_torch, output_bytes)
    mib = 1024.0 * 1024.0
    return Result(
        batch_queries=batch_queries,
        heads=heads,
        seq_len=seq_len,
        output_mib=output_bytes / mib,
        torch_first_ms=torch_first_ms,
        triton_first_ms=triton_first_ms,
        torch_steady_ms=torch_steady_ms,
        triton_steady_ms=triton_steady_ms,
        speedup=torch_steady_ms / triton_steady_ms,
        torch_peak_alloc_mib=torch_peak / mib,
        triton_peak_alloc_mib=triton_peak / mib,
        torch_peak_temp_mib=torch_temp / mib,
        triton_peak_temp_mib=triton_temp / mib,
    )


def _default_shapes(full_matrix: bool) -> list[tuple[int, int, int]]:
    batches = [1, 8, 32]
    heads = [8, 16, 32, 64]
    lengths = [4096, 32768, 131072]
    if full_matrix:
        return [
            (batch, head, length)
            for length in lengths
            for head in heads
            for batch in batches
        ]

    # Acceptance matrix: every T/H at 4K, plus the model-representative H=64
    # for every T at 32K and 128K.
    shapes = {(batch, head, 4096) for batch in batches for head in heads}
    shapes.update(
        (batch, 64, length) for batch in batches for length in (32768, 131072)
    )
    return sorted(shapes, key=lambda shape: (shape[2], shape[1], shape[0]))


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[5]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Torch vs Triton SM80 DSA paged-MQA logits."
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Run the full 3x4x3 T/H/sequence-length Cartesian product.",
    )
    parser.add_argument("--torch-repeats", type=int, default=3)
    parser.add_argument("--triton-repeats", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("This benchmark requires NVIDIA SM80 (A100).")
    if args.torch_repeats < 2 or args.triton_repeats < 2:
        raise ValueError("Both repeat counts must be at least 2.")

    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu": properties.name,
        "gpu_total_memory_mib": properties.total_memory / (1024**2),
        "compute_capability": torch.cuda.get_device_capability(),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "sglang_commit": _git_commit(),
        "method": (
            "first call: perf_counter with synchronization before/after; "
            "steady state: CUDA events after 5 Triton or 2 Torch warmups; "
            "peak allocation: max_memory_allocated delta from live-input baseline"
        ),
        "torch_repeats": args.torch_repeats,
        "triton_repeats": args.triton_repeats,
    }
    print(json.dumps({"metadata": metadata}, sort_keys=True))

    results = []
    for batch_queries, heads, seq_len in _default_shapes(args.full_matrix):
        result = _benchmark_shape(
            batch_queries,
            heads,
            seq_len,
            torch_repeats=args.torch_repeats,
            triton_repeats=args.triton_repeats,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    representative = [
        result
        for result in results
        if result.heads == 64 and result.seq_len in (32768, 131072)
    ]
    minimum_speedup = min(result.speedup for result in representative)
    print(
        json.dumps(
            {
                "summary": {
                    "minimum_representative_speedup": minimum_speedup,
                    "target_speedup": 5.0,
                    "target_met": minimum_speedup >= 5.0,
                }
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
