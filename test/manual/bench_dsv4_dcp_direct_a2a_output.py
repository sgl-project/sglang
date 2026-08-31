"""MI355X DCP8 graph benchmark for producer-integrated A2A output.

Times complete graph replay for:
  baseline: paged decode producer -> dcp_pack_a2a_send -> A2A -> LSE combine
  direct:   paged decode producer ----------------------> A2A -> LSE combine

The reported latency for each arm is the maximum rank latency. No individual
kernel timings are summed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.kernels.ops.attention.dcp_a2a import (
    DCPA2AOutputWorkspace,
    DCPA2APackedOutput,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)
from sglang.srt.layers.dcp import dcp_a2a_lse_reduce
from sglang.srt.utils import is_gfx95_supported, is_hip

HEADS = 128
HEAD_DIM = 512
WORLD_SIZE = 8
# DCP8 C128 decode: 16 owner-local SWA rows + ISL / (128 * 8).
CONTEXT_TO_LOCAL_KV = (
    (8 * 1024, 24),
    (64 * 1024, 80),
    (128 * 1024, 144),
    (256 * 1024, 272),
    (512 * 1024, 528),
    (768 * 1024, 784),
    (983040, 976),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-savings-us", type=float, default=2.5)
    parser.add_argument("--min-step-savings-us", type=float, default=150.0)
    parser.add_argument(
        "--require-go",
        action="store_true",
        help="exit nonzero when projected model-step savings miss the gate",
    )
    return parser.parse_args()


def _init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"benchmark requires exactly {WORLD_SIZE} ranks")
    if not torch.cuda.is_available() or not is_hip() or not is_gfx95_supported():
        raise RuntimeError("benchmark requires eight ROCm gfx950 GPUs")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    coordinator = parallel_state.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    return rank, device, coordinator


def _workspace(device: torch.device) -> DCPA2AOutputWorkspace:
    return DCPA2AOutputWorkspace.allocate(
        world_size=WORLD_SIZE,
        max_batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        device=device,
    )


def _decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    sink: torch.Tensor,
    *,
    output_workspace: DCPA2AOutputWorkspace | None = None,
):
    return _sparse_attn_v4_paged_decode_triton(
        q,
        kv,
        indices,
        indptr,
        sink,
        HEAD_DIM**-0.5,
        block_h=16,
        kv_splits=64,
        block_k=16,
        return_lse=True,
        attn_sink_logit_shift=-math.log(float(WORLD_SIZE)),
        output_workspace=output_workspace,
    )


def _capture_arms(
    *,
    coordinator,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    sink: torch.Tensor,
    device: torch.device,
):
    # Compile both producer specializations and initialize the collective before
    # capture. No JIT compilation or first-use collective setup may enter timing.
    warmup_baseline_workspace = _workspace(device)
    warmup_out, warmup_lse = _decode(q, kv, indices, indptr, sink)
    dcp_a2a_lse_reduce(
        warmup_out,
        warmup_lse,
        coordinator,
        cuda_graph_buffers={
            "send_combined": warmup_baseline_workspace.send_combined,
            "recv_combined": warmup_baseline_workspace.recv_combined,
        },
    )
    warmup_direct_workspace = _workspace(device)
    warmup_packed = _decode(
        q,
        kv,
        indices,
        indptr,
        sink,
        output_workspace=warmup_direct_workspace,
    )
    dcp_a2a_lse_reduce(warmup_packed, None, coordinator)
    torch.cuda.synchronize()

    baseline_workspace = _workspace(device)
    baseline_graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with coordinator.graph_capture() as graph_context:
        with torch.cuda.graph(baseline_graph, stream=graph_context.stream):
            partial_out, partial_lse = _decode(q, kv, indices, indptr, sink)
            baseline_result = dcp_a2a_lse_reduce(
                partial_out,
                partial_lse,
                coordinator,
                cuda_graph_buffers={
                    "send_combined": baseline_workspace.send_combined,
                    "recv_combined": baseline_workspace.recv_combined,
                },
            )

    direct_workspace = _workspace(device)
    direct_graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with coordinator.graph_capture() as graph_context:
        with torch.cuda.graph(direct_graph, stream=graph_context.stream):
            packed = _decode(
                q,
                kv,
                indices,
                indptr,
                sink,
                output_workspace=direct_workspace,
            )
            direct_result = dcp_a2a_lse_reduce(packed, None, coordinator)
    if not isinstance(packed, DCPA2APackedOutput):
        raise RuntimeError("direct producer unexpectedly fell back")

    baseline_graph.replay()
    direct_graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(
        baseline_workspace.send_combined.view(torch.int16),
        direct_workspace.send_combined.view(torch.int16),
    ):
        raise AssertionError("direct producer send bytes differ from old pack")
    if not torch.equal(
        baseline_result.view(torch.int16), direct_result.view(torch.int16)
    ):
        raise AssertionError("direct and old A2A results differ")
    keepalive = (
        baseline_workspace,
        direct_workspace,
        baseline_result,
        direct_result,
        partial_out,
        partial_lse,
        packed,
    )
    return baseline_graph, direct_graph, keepalive


def _rank_max_graph_us(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    warmup_replays: int,
    coordinator,
    device: torch.device,
) -> float:
    for _ in range(warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=coordinator.cpu_group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) * 1000.0 / iterations
    rank_max = torch.tensor(local_us, dtype=torch.float64, device=device)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX, group=coordinator.device_group)
    return float(rank_max.cpu())


def _benchmark_length(
    *,
    local_kv_len: int,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    coordinator,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(20260829 + rank)
    q = torch.randn(
        (1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv = torch.randn(
        (local_kv_len, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    indices = torch.arange(local_kv_len, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, local_kv_len], dtype=torch.int32, device=device)
    sink = torch.randn(
        (HEADS,), dtype=torch.float32, device=device, generator=generator
    )
    baseline_graph, direct_graph, keepalive = _capture_arms(
        coordinator=coordinator,
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        sink=sink,
        device=device,
    )

    baseline_samples = []
    direct_samples = []
    for repeat in range(args.repeats):
        ordered = (
            (("baseline", baseline_graph), ("direct", direct_graph))
            if repeat % 2 == 0
            else (("direct", direct_graph), ("baseline", baseline_graph))
        )
        for name, graph in ordered:
            latency = _rank_max_graph_us(
                graph,
                iterations=args.iterations,
                warmup_replays=args.warmup_replays,
                coordinator=coordinator,
                device=device,
            )
            (baseline_samples if name == "baseline" else direct_samples).append(latency)

    baseline_us = statistics.median(baseline_samples)
    direct_us = statistics.median(direct_samples)
    # Keep graph inputs/outputs and their private-pool views alive through the
    # final replay before this shape is released.
    del keepalive
    return {
        "local_kv_len": local_kv_len,
        "baseline_rank_max_us": baseline_us,
        "direct_rank_max_us": direct_us,
        "savings_us": baseline_us - direct_us,
        "baseline_samples_us": baseline_samples,
        "direct_samples_us": direct_samples,
    }


def main() -> int:
    args = _parse_args()
    rank, device, coordinator = _init_distributed()
    try:
        rows = []
        for context_len, local_kv_len in CONTEXT_TO_LOCAL_KV:
            row = _benchmark_length(
                local_kv_len=local_kv_len,
                args=args,
                rank=rank,
                device=device,
                coordinator=coordinator,
            )
            row["context_len"] = context_len
            rows.append(row)
            if rank == 0:
                print(json.dumps(row, sort_keys=True), flush=True)

        savings = [row["savings_us"] for row in rows]
        median_savings = statistics.median(savings)
        minimum_savings = min(savings)
        # C4 attention is capped at global K=1024, so on DCP8 its production
        # stream is 128 compressed entries plus 16 owner-local SWA entries.
        # The local_kv_len=144 row therefore applies to all 30 C4 layers,
        # while each context row describes the 31 C128 layers.
        c4_savings = next(
            row["savings_us"] for row in rows if row["local_kv_len"] == 144
        )
        for row in rows:
            row["projected_model_savings_us"] = 30 * c4_savings + 31 * row["savings_us"]
        minimum_step_savings = min(row["projected_model_savings_us"] for row in rows)
        decision = "GO" if minimum_step_savings >= args.min_step_savings_us else "NO-GO"
        summary = {
            "decision": decision,
            "gate": "minimum projected 30-C4 + 31-C128 model-step savings",
            "min_savings_us": args.min_savings_us,
            "min_step_savings_us": args.min_step_savings_us,
            "median_savings_us": median_savings,
            "minimum_savings_us": minimum_savings,
            "minimum_step_savings_us": minimum_step_savings,
            "rows": rows,
        }
        if rank == 0:
            print(json.dumps(summary, sort_keys=True), flush=True)
        return int(args.require_go and decision != "GO")
    finally:
        coordinator.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
