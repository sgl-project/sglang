#!/usr/bin/env python3
"""M=2 CUDA-graph microbench: AR + agg HAS_ADD vs fused AR+residual + agg.

Launch: torchrun --nproc_per_node=8 \\
  python/sglang/test/manual/kernels/bench_hip_ar_residual.py

SGLANG_ROCM_AR_RESIDUAL defaults on after this pair won (17.90 vs 17.01 us).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

M = int(os.environ.get("K3_HIP_AR_RESIDUAL_M", "2"))
H = 7168
NVB = 4
WARMUP = 20
ITERS = 50


def _init():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        local_rank=local,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(int(os.environ["WORLD_SIZE"]), 1)


def _ca():
    from aiter.dist.parallel_state import get_tp_group

    comm = get_tp_group().device_communicator
    return None if comm is None else comm.ca_comm


def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
    stream = torch.cuda.current_stream()
    for _ in range(WARMUP):
        graph.replay()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    stream.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


def main() -> int:
    _init()
    rank = dist.get_rank()
    ca = _ca()
    if ca is None or getattr(ca, "disabled", True):
        if rank == 0:
            print("custom AR unavailable", file=sys.stderr)
        return 1

    device = torch.device(f"cuda:{rank}")
    g = torch.Generator(device=device)
    g.manual_seed(rank + 7)
    partial = torch.randn(M, H, dtype=torch.bfloat16, device=device, generator=g)
    prefix = torch.randn(M, H, dtype=torch.bfloat16, device=device, generator=g)
    dist.broadcast(prefix, src=0)
    bank = torch.randn(M, NVB + 1, H, dtype=torch.bfloat16, device=device, generator=g)
    cw = torch.randn(H, dtype=torch.float32, device=device, generator=g)
    ow = torch.randn(H, dtype=torch.bfloat16, device=device, generator=g)

    from aiter.dist.parallel_state import graph_capture

    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    def agg(prefix_in, addend, out, prefix_out):
        attn_res_hip(
            prefix_in,
            bank,
            cw,
            ow,
            out,
            NVB,
            1e-6,
            1e-6,
            addend=addend,
            prefix_out=prefix_out,
            write_prefix=False,
        )

    # Eager reference: AR in bf16, then fp32 add (matches the fused writeback).
    reduced = ca.custom_all_reduce(partial)
    ref_prefix = (reduced.float() + prefix.float()).to(torch.bfloat16)
    ref_out = torch.empty_like(prefix)
    dummy_prefix_out = torch.empty_like(prefix)
    agg(ref_prefix, None, ref_out, dummy_prefix_out)

    fused = ca.custom_all_reduce_residual(partial, prefix)
    if fused is None:
        if rank == 0:
            print("1-stage residual AR unavailable for this size", file=sys.stderr)
        return 1
    if rank == 0:
        max_diff = (fused.float() - ref_prefix.float()).abs().max().item()
        split_bf16 = (reduced + prefix).float()
        split_diff = (fused.float() - split_bf16).abs().max().item()
        print(
            f"AR+res vs AR then fp32-add: max_abs={max_diff:.6g}; "
            f"vs bf16-add: max_abs={split_diff:.6g}"
        )

    fused_out = torch.empty_like(prefix)
    agg(fused, None, fused_out, dummy_prefix_out)
    if rank == 0:
        agg_diff = (fused_out.float() - ref_out.float()).abs().max().item()
        print(f"agg(fused) vs agg(AR+add): max_abs={agg_diff:.6g}")

    # Graph A: AR + HAS_ADD agg
    a_out = torch.empty_like(prefix)
    a_prefix_out = torch.empty_like(prefix)
    with graph_capture() as gc:
        s = gc.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                a_reduced = ca.custom_all_reduce(partial)
                agg(prefix, a_reduced, a_out, a_prefix_out)
        torch.cuda.current_stream().wait_stream(s)
        g_a = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_a, stream=s):
            a_reduced = ca.custom_all_reduce(partial)
            agg(prefix, a_reduced, a_out, a_prefix_out)

    # Graph B: fused AR+res + agg without HAS_ADD
    b_out = torch.empty_like(prefix)
    b_prefix_out = torch.empty_like(prefix)
    with graph_capture() as gc:
        s = gc.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                b_sum = ca.custom_all_reduce_residual(partial, prefix)
                agg(b_sum, None, b_out, b_prefix_out)
        torch.cuda.current_stream().wait_stream(s)
        g_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_b, stream=s):
            b_sum = ca.custom_all_reduce_residual(partial, prefix)
            agg(b_sum, None, b_out, b_prefix_out)

    us_a = _time_graph(g_a)
    us_b = _time_graph(g_b)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (us_a, us_b))
    if rank == 0:
        mean_a = sum(x[0] for x in gathered) / len(gathered)
        mean_b = sum(x[1] for x in gathered) / len(gathered)
        print(
            f"M={M} H={H} nvb={NVB} graph AR+HAS_ADD={mean_a:.2f} us  "
            f"AR_res+agg={mean_b:.2f} us  delta={mean_a - mean_b:.2f} us"
        )
        win = mean_b < mean_a * 0.98
        close = (fused.float() - ref_prefix.float()).abs().max().item() < 0.1
        print("WIN" if win and close else "NO_WIN")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
