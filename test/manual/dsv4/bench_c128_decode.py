"""CUDA Graph benchmark for DeepSeek-V4 online C128 decode.

The benchmark uses the production online planner, state-update JIT kernel,
norm/RoPE/cache-store JIT kernel, compressed-attention metadata kernel, and
FlashMLA sparse decode entry point. It also compares the two supported
BF16xBF16->FP32 wkv_gate GEMM backends at the production matrix sizes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.jit_kernel.dsv4 import (  # noqa: E402
    CompressorDecodePlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.layers import deep_gemm_wrapper  # noqa: E402
from sglang.srt.layers.attention.dsv4.metadata_kernel import (  # noqa: E402
    init_compression_metadata,
)

HEAD_DIM = 512
WKV_OUTPUT_DIM = HEAD_DIM * 2
HIDDEN_DIM = 4096
COMPRESS_RATIO = 128
FULL_PAGE_SIZE = 256
C128_PAGE_SIZE = FULL_PAGE_SIZE // COMPRESS_RATIO
FLASHMLA_HEADS = 64
FLASHMLA_CACHE_BYTES = 584
SWA_WINDOW = 128


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark production online C128 decode CUDA Graph paths."
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_csv_ints,
        default=parse_csv_ints("2,8,16,32"),
    )
    parser.add_argument(
        "--positions",
        type=parse_csv_ints,
        default=parse_csv_ints("1,2,64,127,128,129"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--max-timing-disagreement",
        type=float,
        default=0.05,
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--skip-flashmla",
        action="store_true",
        help="Skip FlashMLA stages when isolating compressor kernels.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class TimingResult:
    event_ms: float
    wall_ms: float

    @property
    def disagreement(self) -> float:
        return abs(self.event_ms - self.wall_ms) / max(self.wall_ms, 1e-12)


@dataclass
class C128Inputs:
    batch_size: int
    position: int
    hidden: torch.Tensor
    wkv_weight: torch.Tensor
    deepgemm_output: torch.Tensor
    state: torch.Tensor
    kv_score_input: torch.Tensor
    ape: torch.Tensor
    plan: CompressorDecodePlan
    compressed: torch.Tensor
    norm_weight: torch.Tensor
    freq_cis: torch.Tensor
    out_loc: torch.Tensor
    kvcache: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    req_to_token: torch.Tensor
    positions: torch.Tensor
    raw_out_loc: torch.Tensor
    page_table: torch.Tensor


@dataclass
class FlashMLAInputs:
    q: torch.Tensor
    swa_cache: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    extra_cache: torch.Tensor
    extra_indices: torch.Tensor
    extra_lengths: torch.Tensor
    attn_sink: torch.Tensor
    baseline_metadata: object
    extra_metadata: object


def make_freq_cis(max_position: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64
        )
    )
    angles = positions[:, None] * inv_freq[None, :]
    return torch.polar(torch.ones_like(angles), angles)


def make_inputs(
    batch_size: int,
    position: int,
    seed: int,
    device: torch.device,
) -> C128Inputs:
    generator = torch.Generator(device=device).manual_seed(
        seed + batch_size * 1000 + position
    )
    hidden = torch.randn(
        (batch_size, HIDDEN_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    wkv_weight = torch.randn(
        (WKV_OUTPUT_DIM, HIDDEN_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    deepgemm_output = torch.empty(
        (batch_size, WKV_OUTPUT_DIM), dtype=torch.float32, device=device
    )
    kv_score_input = torch.randn(
        (batch_size, WKV_OUTPUT_DIM),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    state = torch.empty(
        (batch_size, 1, HEAD_DIM * 3), dtype=torch.float32, device=device
    )
    state[:, :, :HEAD_DIM].normal_(generator=generator)
    state[:, :, HEAD_DIM : 2 * HEAD_DIM].uniform_(
        0.5, 2.0, generator=generator
    )
    state[:, :, 2 * HEAD_DIM :].normal_(generator=generator)
    ape = torch.randn(
        (COMPRESS_RATIO, HEAD_DIM),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    seq_lens = torch.full(
        (batch_size,), position, dtype=torch.int64, device=device
    )
    req_pool_indices = torch.arange(
        batch_size, dtype=torch.int64, device=device
    )
    req_to_token = torch.zeros(
        (batch_size, max(position, 1)), dtype=torch.int32, device=device
    )
    plan = CompressorDecodePlan.generate_online(
        seq_lens,
        req_pool_indices,
        req_to_token,
    )
    compressed = torch.empty(
        (batch_size, HEAD_DIM), dtype=torch.float32, device=device
    )
    norm_weight = torch.randn(
        (HEAD_DIM,), dtype=torch.float32, device=device, generator=generator
    )
    freq_cis = make_freq_cis(max(position + 1, COMPRESS_RATIO + 1), device)
    out_loc = torch.arange(batch_size, dtype=torch.int64, device=device)
    page_bytes = math.ceil(
        FLASHMLA_CACHE_BYTES * C128_PAGE_SIZE / 576
    ) * 576
    kvcache = torch.zeros(
        (math.ceil(batch_size / C128_PAGE_SIZE), page_bytes),
        dtype=torch.uint8,
        device=device,
    )
    positions = seq_lens.to(torch.int32) - 1
    raw_out_loc = torch.arange(
        batch_size, dtype=torch.int64, device=device
    ) * FULL_PAGE_SIZE + positions.to(torch.int64)
    page_table = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).view(batch_size, 1)
    return C128Inputs(
        batch_size=batch_size,
        position=position,
        hidden=hidden,
        wkv_weight=wkv_weight,
        deepgemm_output=deepgemm_output,
        state=state,
        kv_score_input=kv_score_input,
        ape=ape,
        plan=plan,
        compressed=compressed,
        norm_weight=norm_weight,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=kvcache,
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        positions=positions,
        raw_out_loc=raw_out_loc,
        page_table=page_table,
    )


def make_flashmla_inputs(inputs: C128Inputs) -> FlashMLAInputs:
    import sgl_kernel.flash_mla as flash_mla

    device = inputs.hidden.device
    batch_size = inputs.batch_size
    generator = torch.Generator(device=device).manual_seed(
        31 * inputs.batch_size + inputs.position
    )
    q = torch.randn(
        (batch_size, 1, FLASHMLA_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    swa_cache = torch.zeros(
        (batch_size, SWA_WINDOW, 1, FLASHMLA_CACHE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    swa_indices = torch.full(
        (batch_size, 1, SWA_WINDOW), -1, dtype=torch.int32, device=device
    )
    swa_len = min(inputs.position, SWA_WINDOW)
    for batch in range(batch_size):
        swa_indices[batch, 0, :swa_len] = (
            batch * SWA_WINDOW
            + torch.arange(swa_len, dtype=torch.int32, device=device)
        )
    swa_lengths = torch.full(
        (batch_size,), swa_len, dtype=torch.int32, device=device
    )

    extra_cache = torch.zeros(
        (batch_size, C128_PAGE_SIZE, 1, FLASHMLA_CACHE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    extra_indices = torch.full(
        (batch_size, 1, 64), -1, dtype=torch.int32, device=device
    )
    extra_len = inputs.position // COMPRESS_RATIO
    if extra_len:
        for batch in range(batch_size):
            extra_indices[batch, 0, 0] = batch * C128_PAGE_SIZE
    extra_lengths = torch.full(
        (batch_size,), max(extra_len, 1), dtype=torch.int32, device=device
    )
    attn_sink = torch.full(
        (FLASHMLA_HEADS,), -1e30, dtype=torch.float32, device=device
    )
    return FlashMLAInputs(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=attn_sink,
        baseline_metadata=flash_mla.get_mla_metadata()[0],
        extra_metadata=flash_mla.get_mla_metadata()[0],
    )


def cublas_gemm(inputs: C128Inputs) -> torch.Tensor:
    return torch.mm(
        inputs.hidden,
        inputs.wkv_weight.t(),
        out_dtype=torch.float32,
    )


def deepgemm_gemm(inputs: C128Inputs) -> torch.Tensor:
    deep_gemm_wrapper.gemm_nt_bf16bf16f32(
        inputs.hidden,
        inputs.wkv_weight,
        inputs.deepgemm_output,
    )
    return inputs.deepgemm_output


def online_plan_metadata(inputs: C128Inputs) -> CompressorDecodePlan:
    return CompressorDecodePlan.generate_online(
        inputs.seq_lens,
        inputs.req_pool_indices,
        inputs.req_to_token,
    )


def online_state_update(inputs: C128Inputs) -> torch.Tensor:
    return compress_forward(
        inputs.state,
        inputs.kv_score_input,
        inputs.ape,
        inputs.plan,
        head_dim=HEAD_DIM,
        compress_ratio=COMPRESS_RATIO,
        out=inputs.compressed,
        is_online=True,
    )


def norm_rope_store(inputs: C128Inputs) -> None:
    compress_norm_rope_store(
        inputs.compressed,
        inputs.plan,
        norm_weight=inputs.norm_weight,
        norm_eps=1e-6,
        freq_cis=inputs.freq_cis,
        out_loc=inputs.out_loc,
        kvcache=inputs.kvcache,
        page_size=C128_PAGE_SIZE,
    )


def existing_c128_core(inputs: C128Inputs) -> None:
    online_state_update(inputs)
    norm_rope_store(inputs)


def c128_attention_metadata(inputs: C128Inputs) -> object:
    return init_compression_metadata(
        inputs.seq_lens,
        inputs.positions,
        inputs.raw_out_loc,
        inputs.page_table,
        FULL_PAGE_SIZE,
        True,
    )


def flashmla_forward(
    flash_inputs: FlashMLAInputs,
    *,
    with_extra: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import sgl_kernel.flash_mla as flash_mla

    return flash_mla.flash_mla_with_kvcache(
        q=flash_inputs.q,
        k_cache=flash_inputs.swa_cache,
        head_dim_v=HEAD_DIM,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=(
            flash_inputs.extra_metadata
            if with_extra
            else flash_inputs.baseline_metadata
        ),
        softmax_scale=HEAD_DIM**-0.5,
        is_fp8_kvcache=True,
        indices=flash_inputs.swa_indices,
        topk_length=flash_inputs.swa_lengths,
        attn_sink=flash_inputs.attn_sink,
        extra_k_cache=flash_inputs.extra_cache if with_extra else None,
        extra_indices_in_kvcache=(
            flash_inputs.extra_indices if with_extra else None
        ),
        extra_topk_length=flash_inputs.extra_lengths if with_extra else None,
    )


def capture_graph(
    fn: Callable[[], object], warmup: int
) -> tuple[torch.cuda.CUDAGraph, object]:
    output = None
    for _ in range(max(warmup, 1)):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iters: int,
    trials: int,
) -> TimingResult:
    if iters <= 0 or trials <= 0:
        raise ValueError("--iters and --trials must be positive")
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(
            TimingResult(
                event_ms=start.elapsed_time(end) / iters,
                wall_ms=(time.perf_counter() - wall_start) * 1e3 / iters,
            )
        )
    samples.sort(key=lambda sample: sample.wall_ms)
    return samples[len(samples) // 2]


def benchmark_graph(
    fn: Callable[[], object], args: argparse.Namespace
) -> TimingResult:
    graph, static_output = capture_graph(fn, args.warmup)
    result = time_graph(
        graph,
        warmup=args.warmup,
        iters=args.iters,
        trials=args.trials,
    )
    graph.reset()
    del static_output
    torch.cuda.synchronize()
    gc.collect()
    return result


def format_timing(result: TimingResult) -> str:
    return (
        f"event={result.event_ms:.3f} ms, wall={result.wall_ms:.3f} ms, "
        f"delta={result.disagreement * 100:.1f}%"
    )


def write_result(
    args: argparse.Namespace,
    inputs: C128Inputs,
    timings: dict[str, TimingResult],
) -> None:
    if args.json_output is None:
        return
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "command": [sys.executable, *sys.argv],
        "batch_size": inputs.batch_size,
        "position": inputs.position,
        "is_boundary": inputs.position % COMPRESS_RATIO == 0,
        "warmup": args.warmup,
        "iters": args.iters,
        "trials": args.trials,
        "timing_valid": all(
            timing.disagreement <= args.max_timing_disagreement
            for timing in timings.values()
        ),
        "deepgemm_over_cublas": (
            timings["wkv_gate_cublas"].wall_ms
            / timings["wkv_gate_deepgemm"].wall_ms
        ),
        "timings": {
            name: {
                "event_ms": timing.event_ms,
                "wall_ms": timing.wall_ms,
                "disagreement": timing.disagreement,
            }
            for name, timing in timings.items()
        },
    }
    with args.json_output.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def benchmark_shape(
    args: argparse.Namespace,
    batch_size: int,
    position: int,
    device: torch.device,
) -> None:
    inputs = make_inputs(batch_size, position, args.seed, device)

    # Compile and initialize every backend before graph capture.
    cublas_gemm(inputs)
    deepgemm_gemm(inputs)
    online_plan_metadata(inputs)
    existing_c128_core(inputs)
    c128_attention_metadata(inputs)
    torch.cuda.synchronize()

    flash_inputs: Optional[FlashMLAInputs] = None
    if not args.skip_flashmla:
        flash_inputs = make_flashmla_inputs(inputs)
        flashmla_forward(flash_inputs, with_extra=False)
        flashmla_forward(flash_inputs, with_extra=True)
        torch.cuda.synchronize()

    timings = {
        "wkv_gate_cublas": benchmark_graph(
            lambda: cublas_gemm(inputs), args
        ),
        "wkv_gate_deepgemm": benchmark_graph(
            lambda: deepgemm_gemm(inputs), args
        ),
        "online_plan_metadata": benchmark_graph(
            lambda: online_plan_metadata(inputs), args
        ),
        "online_state_update": benchmark_graph(
            lambda: online_state_update(inputs), args
        ),
        "boundary_norm_rope_store": benchmark_graph(
            lambda: norm_rope_store(inputs), args
        ),
        "c128_core_existing": benchmark_graph(
            lambda: existing_c128_core(inputs), args
        ),
        "c128_attention_metadata": benchmark_graph(
            lambda: c128_attention_metadata(inputs), args
        ),
    }
    if flash_inputs is not None:
        timings["flashmla_swa"] = benchmark_graph(
            lambda: flashmla_forward(flash_inputs, with_extra=False), args
        )
        timings["flashmla_swa_c128"] = benchmark_graph(
            lambda: flashmla_forward(flash_inputs, with_extra=True), args
        )

    invalid = {
        name: timing.disagreement
        for name, timing in timings.items()
        if timing.disagreement > args.max_timing_disagreement
    }
    print(
        f"C128 decode graph: batch={batch_size}, position={position}, "
        f"boundary={position % COMPRESS_RATIO == 0}"
    )
    for name, timing in timings.items():
        print(f"  {name:26}: {format_timing(timing)}")
    gemm_speedup = (
        timings["wkv_gate_cublas"].wall_ms
        / timings["wkv_gate_deepgemm"].wall_ms
    )
    print(f"  DeepGEMM over cuBLAS     : {gemm_speedup:.3f}x")
    print(f"  timing validity          : {'FAIL' if invalid else 'PASS'}")
    write_result(args, inputs, timings)
    if invalid:
        details = ", ".join(
            f"{name}={delta * 100:.1f}%" for name, delta in invalid.items()
        )
        raise AssertionError(
            "CUDA-event and synchronized wall timings disagree beyond "
            f"{args.max_timing_disagreement * 100:.1f}%: {details}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    torch.cuda.set_device(device)
    for batch_size in args.batch_sizes:
        for position in args.positions:
            benchmark_shape(args, batch_size, position, device)


if __name__ == "__main__":
    main()
