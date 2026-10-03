"""Compare the complete Beam MoE boundary under CUDA graph replay.

Run with ``torchrun --standalone --nproc-per-node=8 <this file>``.
Times are maximum-rank GPU event latency, not full-model latency.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.beam_all_reduce import beam_all_reduce_norm
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.layernorm.beam_fused import fused_sandwich_residual_rmsnorm
from sglang.kernels.ops.moe.fused_moe_triton_kernels import moe_sum_reduce_triton
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 64])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    ps._WORLD = coord = ps.init_world_group(list(range(world)), rank, "nccl")
    get_parallel().override_permanently(world_group=coord)
    comm = CustomAllReduceV2(coord.cpu_group, torch.device("cuda", rank))
    if comm.disabled:
        raise RuntimeError("An NVLink V2 communicator is required")
    register_comm_cleanup(comm)

    def latency(fn):
        for _ in range(5):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                fn()
        for _ in range(1000):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=coord.device_group)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        us = torch.tensor(start.elapsed_time(end) * 1000 / 600, device="cuda")
        dist.all_reduce(us, op=dist.ReduceOp.MAX, group=coord.device_group)
        return us.item()

    for tokens in args.tokens:
        torch.manual_seed(42 + tokens + rank)
        x = torch.randn(tokens, 8, 5120, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
        post = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
        next_weight = torch.randn_like(post)

        def base():
            local = torch.empty_like(residual)
            moe_sum_reduce_triton(x, local, 2.5)
            reduced = comm.custom_all_reduce(local)
            return fused_sandwich_residual_rmsnorm(
                reduced, residual, post, next_weight, 1e-6, 0.13, 0.97
            )

        def fused():
            return beam_all_reduce_norm(
                comm.obj,
                x,
                residual,
                post,
                next_weight,
                1e-6,
                (0.13, 0.97),
                routed_scale=2.5,
            )

        samples = {"base": [], "fused": []}
        for order in (("base", "fused"), ("fused", "base")) * 2:
            for name in order:
                samples[name].append(latency(base if name == "base" else fused))
        if rank == 0:
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "tp": world,
                        "gpu": torch.cuda.get_device_name(),
                        "microseconds": samples,
                        "latency_reduction_pct": 100
                        * (
                            1
                            - statistics.median(samples["fused"])
                            / statistics.median(samples["base"])
                        ),
                    }
                ),
                flush=True,
            )
    dist.barrier(group=coord.device_group)
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
