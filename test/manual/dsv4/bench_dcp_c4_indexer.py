"""Benchmark the DCP=2 page-sharded DeepSeek-V4 C4 indexer.

The reference path runs the production full-history FP8 paged-MQA logits
kernel and topK transform on every rank. The candidate path interleaves logical
C4 pages across ranks, runs the same kernels on the local pages, all-gathers
local topK candidates, and merges the global topK.

Run on a two-GPU node:

    torchrun --nproc_per_node=2 test/manual/dsv4/bench_dcp_c4_indexer.py

An optional local performance gate can be used after choosing a stable shape:

    torchrun --nproc_per_node=2 test/manual/dsv4/bench_dcp_c4_indexer.py \
        --context-len 131072 --min-speedup 1.10

This is intentionally a manual benchmark. Latency thresholds are sensitive to
GPU type, clocks, driver, and NCCL topology, so they do not belong in regular
CI.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

import deep_gemm  # noqa: E402

from sglang.jit_kernel.dsv4 import (  # noqa: E402
    merge_dcp_topk_candidates_512,
    topk_candidates_512,
    topk_transform_512,
)
from sglang.srt.distributed.parallel_state import (  # noqa: E402
    get_dcp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.dsv4.metadata import (  # noqa: E402
    get_dcp_sharded_c4_seq_lens,
)

C4_PAGE_SIZE = 64
HEAD_DIM = 128
TOPK = 512
FP8_DTYPE = torch.float8_e4m3fn


@dataclass
class BenchInputs:
    q: torch.Tensor
    weights: torch.Tensor
    cache: torch.Tensor
    page_table: torch.Tensor
    c4_seq_lens: torch.Tensor
    global_metadata: torch.Tensor
    local_page_table: torch.Tensor
    local_c4_seq_lens: torch.Tensor
    local_metadata: torch.Tensor
    max_c4_seq_len: int


class DCPGroup:
    """Small benchmark adapter matching GroupCoordinator.all_gather semantics."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        input_ = input_.contiguous()
        input_shape = input_.shape
        output = torch.empty(
            (self.world_size * input_shape[0],) + input_shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.all_gather_into_tensor(output, input_)
        output = output.reshape((self.world_size,) + input_shape)
        output = output.movedim(0, dim)
        return output.reshape(
            input_shape[:dim]
            + (self.world_size * input_shape[dim],)
            + input_shape[dim + 1 :]
        )

    def all_gather_into_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        dist.all_gather_into_tensor(output, input_.contiguous())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DCP=2 C4 indexer page sharding."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--context-len",
        type=int,
        default=32768,
        help="Uncompressed token context length; must be divisible by 4.",
    )
    parser.add_argument("--num-heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture each measured path and benchmark CUDA Graph replay.",
    )
    parser.add_argument(
        "--use-sglang-group",
        action="store_true",
        help="Use the production GroupCoordinator (for example TP8/DCP2).",
    )
    parser.add_argument(
        "--max-timing-disagreement",
        type=float,
        default=0.05,
        help="Maximum relative disagreement between CUDA-event and wall timing.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=0.0,
        help="Fail if reference_ms / sharded_ms is below this value.",
    )
    return parser.parse_args()


def init_distributed(
    use_sglang_group: bool, dcp_size: int
) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if dcp_size <= 0 or world_size % dcp_size != 0:
        raise RuntimeError(
            f"DCP size must divide world size, got {world_size=} and {dcp_size=}."
        )
    if not use_sglang_group and world_size != dcp_size:
        raise RuntimeError(
            "The local adapter requires world size to equal DCP size; use "
            "--use-sglang-group for TP8/DCP2 topology."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if use_sglang_group:
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_parallel_size=dcp_size,
        )
    elif not dist.is_initialized():
        dist.init_process_group("nccl")
    return world_size, rank, device


def make_metadata(seq_lens: torch.Tensor) -> torch.Tensor:
    seq_lens = seq_lens.view(-1, 1).to(torch.int32)
    return deep_gemm.get_paged_mqa_logits_metadata(
        seq_lens,
        C4_PAGE_SIZE,
        deep_gemm.get_num_sms(),
    )


def make_inputs(
    args: argparse.Namespace,
    world_size: int,
    rank: int,
    device: torch.device,
) -> BenchInputs:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.context_len <= 0 or args.context_len % 4 != 0:
        raise ValueError("--context-len must be positive and divisible by 4")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be positive")

    c4_seq_len = args.context_len // 4
    num_pages = (c4_seq_len + C4_PAGE_SIZE - 1) // C4_PAGE_SIZE
    max_c4_seq_len = num_pages * C4_PAGE_SIZE
    total_slots = num_pages * C4_PAGE_SIZE
    generator = torch.Generator(device=device).manual_seed(args.seed)

    k = torch.randn(
        total_slots,
        HEAD_DIM,
        device=device,
        generator=generator,
        dtype=torch.float32,
    ).to(FP8_DTYPE)
    k_u8 = k.view(torch.uint8).reshape(total_slots, HEAD_DIM)
    k_scale = (
        torch.rand(total_slots, device=device, generator=generator) * 0.02 + 0.99
    )
    k_scale_u8 = k_scale.contiguous().view(torch.uint8).reshape(total_slots, 4)

    page_bytes = C4_PAGE_SIZE * (HEAD_DIM + 4)
    cache_2d = torch.empty(
        num_pages,
        page_bytes,
        device=device,
        dtype=torch.uint8,
    )
    key_bytes = C4_PAGE_SIZE * HEAD_DIM
    cache_2d[:, :key_bytes] = k_u8.view(
        num_pages, C4_PAGE_SIZE, HEAD_DIM
    ).reshape(num_pages, key_bytes)
    cache_2d[:, key_bytes:] = k_scale_u8.view(
        num_pages, C4_PAGE_SIZE, 4
    ).reshape(num_pages, C4_PAGE_SIZE * 4)
    cache = cache_2d.view(num_pages, C4_PAGE_SIZE, 1, HEAD_DIM + 4)

    q = torch.randn(
        args.batch_size,
        1,
        args.num_heads,
        HEAD_DIM,
        device=device,
        generator=generator,
        dtype=torch.float32,
    ).to(FP8_DTYPE)
    weights = torch.randn(
        args.batch_size,
        args.num_heads,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    page_table = torch.stack(
        [
            torch.randperm(num_pages, device=device, generator=generator)
            for _ in range(args.batch_size)
        ]
    ).to(torch.int32)
    c4_seq_lens = torch.full(
        (args.batch_size,),
        c4_seq_len,
        device=device,
        dtype=torch.int32,
    )

    local_page_table = page_table[:, rank::world_size].contiguous()
    local_c4_seq_lens = get_dcp_sharded_c4_seq_lens(
        c4_seq_lens,
        C4_PAGE_SIZE,
        world_size,
        rank,
    ).contiguous()
    return BenchInputs(
        q=q,
        weights=weights,
        cache=cache,
        page_table=page_table,
        c4_seq_lens=c4_seq_lens,
        global_metadata=make_metadata(c4_seq_lens),
        local_page_table=local_page_table,
        local_c4_seq_lens=local_c4_seq_lens,
        local_metadata=make_metadata(local_c4_seq_lens),
        max_c4_seq_len=max_c4_seq_len,
    )


def run_logits(
    inputs: BenchInputs,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    metadata: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    return deep_gemm.fp8_paged_mqa_logits(
        inputs.q,
        inputs.cache,
        inputs.weights,
        seq_lens.view(-1, 1),
        page_table,
        metadata,
        max_seq_len,
        False,
    )


def reference_topk(
    inputs: BenchInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = run_logits(
        inputs,
        inputs.page_table,
        inputs.c4_seq_lens,
        inputs.global_metadata,
        inputs.max_c4_seq_len,
    )
    raw = torch.empty(
        inputs.q.shape[0], TOPK, device=logits.device, dtype=torch.int32
    )
    physical = torch.empty_like(raw)
    topk_transform_512(
        logits,
        inputs.c4_seq_lens,
        inputs.page_table,
        physical,
        C4_PAGE_SIZE,
        raw,
    )
    return raw, physical, logits


def local_score(inputs: BenchInputs) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.q.shape[0]
    if inputs.local_page_table.shape[1] == 0:
        local_page_table = inputs.page_table[:, :1].contiguous()
        local_logits = torch.full(
            (batch_size, C4_PAGE_SIZE),
            float("-inf"),
            device=inputs.q.device,
            dtype=torch.float32,
        )
    else:
        local_page_table = inputs.local_page_table
        local_logits = run_logits(
            inputs,
            local_page_table,
            inputs.local_c4_seq_lens,
            inputs.local_metadata,
            local_page_table.shape[1] * C4_PAGE_SIZE,
        )
    return local_logits, local_page_table


def local_topk_candidates(
    inputs: BenchInputs,
    local_logits: torch.Tensor,
    local_page_table: torch.Tensor,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.q.shape[0]
    local_raw = torch.empty(
        batch_size, TOPK, device=inputs.q.device, dtype=torch.int32
    )
    local_physical = torch.empty_like(local_raw)
    topk_transform_512(
        local_logits,
        inputs.local_c4_seq_lens,
        local_page_table,
        local_physical,
        C4_PAGE_SIZE,
        local_raw,
    )

    local_raw_i64 = local_raw.to(torch.int64)
    local_scores = torch.gather(
        local_logits, 1, local_raw_i64.clamp(min=0)
    ).masked_fill(local_raw < 0, float("-inf"))

    page_bits = C4_PAGE_SIZE.bit_length() - 1
    page_mask = C4_PAGE_SIZE - 1
    global_raw = (
        (
            (local_raw_i64 >> page_bits) * group.world_size
            + group.rank_in_group
        )
        << page_bits
    ) | (local_raw_i64 & page_mask)
    global_raw = global_raw.to(torch.int32).masked_fill(local_raw < 0, -1)
    return local_scores, global_raw


def gather_candidates(
    local_scores: torch.Tensor,
    global_raw: torch.Tensor,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        group.all_gather(local_scores.contiguous(), dim=1),
        group.all_gather(global_raw.contiguous(), dim=1),
    )


def merge_candidates(
    inputs: BenchInputs,
    gathered_scores: torch.Tensor,
    gathered_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.q.shape[0]
    page_bits = C4_PAGE_SIZE.bit_length() - 1
    page_mask = C4_PAGE_SIZE - 1
    merged_scores, merged_pos = torch.topk(
        gathered_scores, k=TOPK, dim=1, largest=True, sorted=False
    )
    final_raw = torch.gather(gathered_raw, 1, merged_pos)
    valid = (merged_scores != float("-inf")) & (final_raw >= 0)
    final_raw = final_raw.masked_fill(~valid, -1)

    seq_lens = inputs.c4_seq_lens.to(torch.int64)
    positions = torch.arange(TOPK, device=inputs.q.device, dtype=torch.int32)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    sequential = positions.masked_fill(
        positions.to(torch.int64) >= seq_lens.unsqueeze(1), -1
    )
    final_raw = torch.where(
        (seq_lens <= TOPK).unsqueeze(1), sequential, final_raw
    )

    raw_i64 = final_raw.clamp(min=0).to(torch.int64)
    page_ids = raw_i64 >> page_bits
    offsets = raw_i64 & page_mask
    physical_pages = torch.gather(inputs.page_table, 1, page_ids)
    physical = (physical_pages << page_bits) | offsets.to(torch.int32)
    physical = physical.masked_fill(final_raw < 0, -1)
    return final_raw, physical


def sharded_topk(
    inputs: BenchInputs,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_logits, local_page_table = local_score(inputs)
    local_scores, global_raw = local_topk_candidates(
        inputs, local_logits, local_page_table, group
    )
    gathered_scores, gathered_raw = gather_candidates(
        local_scores, global_raw, group
    )
    return merge_candidates(inputs, gathered_scores, gathered_raw)


def packed_sharded_topk(
    inputs: BenchInputs,
    group: Any,
    local_candidates: Optional[torch.Tensor] = None,
    gathered_candidates: Optional[torch.Tensor] = None,
    out_raw: Optional[torch.Tensor] = None,
    out_physical: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.q.shape[0]
    local_logits, _ = local_score(inputs)
    if local_candidates is None:
        local_candidates = torch.empty(
            (batch_size, TOPK), device=inputs.q.device, dtype=torch.int64
        )
    if gathered_candidates is None:
        gathered_candidates = torch.empty(
            (group.world_size * batch_size, TOPK),
            device=inputs.q.device,
            dtype=torch.int64,
        )
    if out_raw is None:
        out_raw = torch.empty(
            (batch_size, TOPK), device=inputs.q.device, dtype=torch.int32
        )
    if out_physical is None:
        out_physical = torch.empty_like(out_raw)

    topk_candidates_512(
        local_logits,
        inputs.local_c4_seq_lens,
        local_candidates,
        C4_PAGE_SIZE,
        group.world_size,
        group.rank_in_group,
    )
    group.all_gather_into_tensor(gathered_candidates, local_candidates)
    merge_dcp_topk_candidates_512(
        gathered_candidates,
        inputs.c4_seq_lens,
        inputs.page_table,
        out_physical,
        C4_PAGE_SIZE,
        group.world_size,
        out_raw,
    )
    return out_raw, out_physical


def assert_equivalent(
    reference_raw: torch.Tensor,
    sharded_raw: torch.Tensor,
    reference_logits: torch.Tensor,
) -> bool:
    ref_sorted = torch.sort(reference_raw, dim=1).values
    sharded_sorted = torch.sort(sharded_raw, dim=1).values
    if torch.equal(ref_sorted, sharded_sorted):
        return True

    # Equal scores at the topK boundary may legally select different indices.
    ref_scores = torch.gather(
        reference_logits, 1, reference_raw.clamp(min=0).to(torch.int64)
    ).masked_fill(reference_raw < 0, float("-inf"))
    sharded_scores = torch.gather(
        reference_logits, 1, sharded_raw.clamp(min=0).to(torch.int64)
    ).masked_fill(sharded_raw < 0, float("-inf"))
    ref_scores = torch.sort(ref_scores, dim=1).values
    sharded_scores = torch.sort(sharded_scores, dim=1).values
    if torch.equal(ref_scores, sharded_scores):
        return False

    mismatch = (ref_sorted != sharded_sorted).nonzero()
    first = mismatch[0].tolist() if mismatch.numel() else ["unknown"]
    raise AssertionError(
        "sharded topK differs from full-history topK beyond tie freedom; "
        f"first raw-index mismatch at {first}"
    )


@dataclass(frozen=True)
class TimingResult:
    event_ms: float
    wall_ms: float

    @property
    def disagreement(self) -> float:
        return abs(self.event_ms - self.wall_ms) / max(self.wall_ms, 1e-12)


def time_cuda_ms(
    fn: Callable[[], object], warmup: int, iters: int
) -> TimingResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3 / iters
    dist.barrier()
    return TimingResult(event_ms=start.elapsed_time(end) / iters, wall_ms=wall_ms)


def reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def reduce_timing(result: TimingResult, device: torch.device) -> TimingResult:
    return TimingResult(
        event_ms=reduce_max(result.event_ms, device),
        wall_ms=reduce_max(result.wall_ms, device),
    )


def capture_cuda_graph(fn: Callable[[], object], warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return graph


def benchmark_path(
    fn: Callable[[], object],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> TimingResult:
    graph = None
    measured_fn = fn
    if args.cuda_graph:
        graph = capture_cuda_graph(fn, args.warmup)
        measured_fn = graph.replay

    result = reduce_timing(
        time_cuda_ms(measured_fn, args.warmup, args.iters), device
    )

    if graph is not None:
        torch.cuda.synchronize()
        dist.barrier()
        graph.reset()
        measured_fn = graph = None
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
    return result


def format_timing(result: TimingResult) -> str:
    return (
        f"event={result.event_ms:.3f} ms, wall={result.wall_ms:.3f} ms, "
        f"delta={result.disagreement * 100:.1f}%"
    )


def main() -> None:
    args = parse_args()
    world_size, rank, device = init_distributed(
        args.use_sglang_group, args.dcp_size
    )
    group = (
        get_dcp_group()
        if args.use_sglang_group
        else DCPGroup(world_size=world_size, rank=rank)
    )
    inputs = make_inputs(
        args, group.world_size, group.rank_in_group, device
    )

    gathered_lens = group.all_gather(
        inputs.local_c4_seq_lens.unsqueeze(1), dim=1
    )
    if not torch.equal(gathered_lens.sum(dim=1), inputs.c4_seq_lens):
        raise AssertionError("DCP local C4 lengths do not partition the global length")

    ref_raw, ref_physical, ref_logits = reference_topk(inputs)
    sharded_raw, sharded_physical = sharded_topk(inputs, group)
    exact_set_match = assert_equivalent(ref_raw, sharded_raw, ref_logits)
    if exact_set_match and not torch.equal(
        torch.sort(ref_physical, dim=1).values,
        torch.sort(sharded_physical, dim=1).values,
    ):
        raise AssertionError("raw topK matches, but physical C4 slots differ")
    packed_raw, packed_physical = packed_sharded_topk(inputs, group)
    packed_exact_set_match = assert_equivalent(ref_raw, packed_raw, ref_logits)
    if packed_exact_set_match and not torch.equal(
        torch.sort(ref_physical, dim=1).values,
        torch.sort(packed_physical, dim=1).values,
    ):
        raise AssertionError("packed raw topK matches, but physical slots differ")
    dist.barrier()

    # Build stable stage inputs once. Each benchmark path captures its own graph,
    # so these tensors remain live and keep replay addresses fixed.
    ref_stage_logits = run_logits(
        inputs,
        inputs.page_table,
        inputs.c4_seq_lens,
        inputs.global_metadata,
        inputs.max_c4_seq_len,
    )
    ref_stage_raw = torch.empty_like(ref_raw)
    ref_stage_physical = torch.empty_like(ref_physical)
    local_stage_logits, local_stage_page_table = local_score(inputs)
    local_stage_scores, local_stage_raw = local_topk_candidates(
        inputs, local_stage_logits, local_stage_page_table, group
    )
    gathered_stage_scores, gathered_stage_raw = gather_candidates(
        local_stage_scores, local_stage_raw, group
    )
    packed_local_candidates = torch.empty(
        (args.batch_size, TOPK), device=device, dtype=torch.int64
    )
    packed_gathered_candidates = torch.empty(
        (group.world_size * args.batch_size, TOPK),
        device=device,
        dtype=torch.int64,
    )
    packed_stage_raw = torch.empty_like(ref_raw)
    packed_stage_physical = torch.empty_like(ref_physical)

    def ref_topk_stage() -> None:
        topk_transform_512(
            ref_stage_logits,
            inputs.c4_seq_lens,
            inputs.page_table,
            ref_stage_physical,
            C4_PAGE_SIZE,
            ref_stage_raw,
        )

    timings = {
        "reference full": benchmark_path(
            lambda: reference_topk(inputs), args=args, device=device
        ),
        "reference score": benchmark_path(
            lambda: run_logits(
                inputs,
                inputs.page_table,
                inputs.c4_seq_lens,
                inputs.global_metadata,
                inputs.max_c4_seq_len,
            ),
            args=args,
            device=device,
        ),
        "reference topK": benchmark_path(
            ref_topk_stage, args=args, device=device
        ),
        "sharded full": benchmark_path(
            lambda: sharded_topk(inputs, group), args=args, device=device
        ),
        "local score": benchmark_path(
            lambda: local_score(inputs), args=args, device=device
        ),
        "local topK": benchmark_path(
            lambda: local_topk_candidates(
                inputs, local_stage_logits, local_stage_page_table, group
            ),
            args=args,
            device=device,
        ),
        "candidate collectives": benchmark_path(
            lambda: gather_candidates(
                local_stage_scores, local_stage_raw, group
            ),
            args=args,
            device=device,
        ),
        "global merge": benchmark_path(
            lambda: merge_candidates(
                inputs, gathered_stage_scores, gathered_stage_raw
            ),
            args=args,
            device=device,
        ),
        "packed full": benchmark_path(
            lambda: packed_sharded_topk(
                inputs,
                group,
                packed_local_candidates,
                packed_gathered_candidates,
                packed_stage_raw,
                packed_stage_physical,
            ),
            args=args,
            device=device,
        ),
        "packed local topK": benchmark_path(
            lambda: topk_candidates_512(
                local_stage_logits,
                inputs.local_c4_seq_lens,
                packed_local_candidates,
                C4_PAGE_SIZE,
                group.world_size,
                group.rank_in_group,
            ),
            args=args,
            device=device,
        ),
        "packed collective": benchmark_path(
            lambda: group.all_gather_into_tensor(
                packed_gathered_candidates, packed_local_candidates
            ),
            args=args,
            device=device,
        ),
        "packed merge": benchmark_path(
            lambda: merge_dcp_topk_candidates_512(
                packed_gathered_candidates,
                inputs.c4_seq_lens,
                inputs.page_table,
                packed_stage_physical,
                C4_PAGE_SIZE,
                group.world_size,
                packed_stage_raw,
            ),
            args=args,
            device=device,
        ),
    }
    ref_ms = timings["reference full"].wall_ms
    sharded_ms = timings["sharded full"].wall_ms
    packed_ms = timings["packed full"].wall_ms
    speedup = ref_ms / sharded_ms
    packed_speedup = ref_ms / packed_ms

    invalid_timings = {
        name: result.disagreement
        for name, result in timings.items()
        if result.disagreement > args.max_timing_disagreement
    }

    local_items = int(inputs.local_c4_seq_lens.sum().item())
    local_items_max = int(reduce_max(float(local_items), device))
    if rank == 0:
        c4_seq_len = args.context_len // 4
        print("DeepSeek-V4 DCP C4 indexer benchmark")
        print(
            f"world_size / DCP size      : {world_size}/{group.world_size}"
        )
        print(
            f"execution mode             : "
            f"{'cuda graph' if args.cuda_graph else 'eager'}"
        )
        print(f"batch_size                 : {args.batch_size}")
        print(f"raw context length         : {args.context_len}")
        print(f"C4 history length          : {c4_seq_len}")
        print(f"shard chunk                : {C4_PAGE_SIZE} C4 / 256 raw tokens")
        print(f"topK                       : {TOPK}")
        print(f"indexer heads/head_dim     : {args.num_heads}/{HEAD_DIM}")
        print(f"full score items/rank      : {args.batch_size * c4_seq_len}")
        print(f"max local score items/rank : {local_items_max}")
        print(
            f"candidate items gathered   : "
            f"{args.batch_size * TOPK * group.world_size}"
        )
        print(f"exact raw topK set match   : {exact_set_match}")
        print(f"packed raw topK set match  : {packed_exact_set_match}")
        for name, result in timings.items():
            print(f"{name:27}: {format_timing(result)}")
        print(f"wall-clock speedup         : {speedup:.3f}x")
        print(f"packed wall speedup        : {packed_speedup:.3f}x")
        print(f"timing validity            : {'FAIL' if invalid_timings else 'PASS'}")

    if invalid_timings:
        details = ", ".join(
            f"{name}={delta * 100:.1f}%"
            for name, delta in invalid_timings.items()
        )
        raise AssertionError(
            "CUDA-event and synchronized wall timings disagree beyond "
            f"{args.max_timing_disagreement * 100:.1f}%: {details}"
        )

    if args.min_speedup > 0 and packed_speedup < args.min_speedup:
        raise AssertionError(
            f"packed speedup {packed_speedup:.3f}x is below --min-speedup "
            f"{args.min_speedup:.3f}x"
        )

    torch.cuda.synchronize()
    dist.barrier()
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
