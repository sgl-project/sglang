"""Benchmark the K3 column-parallel up_proj + multicast all-gather + add3
(gemm_ag, staged in the CustomAllReduceV2 push workspace) against the
replicated cublas GEMM + add3 baseline, at decode sizes over TP8.

Correctness lives in test/registered/jit/kimi_k3/test_gemm_ag.py.
"""

from __future__ import annotations

import atexit
import logging
import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.elementwise.add3 import add3
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.ops.kimi_k3 import all_reduce, gemm_ag
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.jit.utils import cache_once
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires 8 GPUs with NVLS multicast, self-skips in CI",
)

K, N, WORLD_N = gemm_ag.K, gemm_ag.N, gemm_ag.N // 8
BATCH_SIZES = list(range(1, gemm_ag.MAX_TOKENS + 1))
PROVIDERS = ["gemm_ag", "cublas+add3", "cublas"]


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return coord.cpu_group


@cache_once
def _init_nccl_group():
    _init_world()
    local_rank = int(os.environ["LOCAL_RANK"])
    return dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))


@cache_once
def _init_comm() -> CustomAllReduceV2:
    """Production-shaped comm: default workspace sizes, push counters."""
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled or comm.mc_base_ptr == 0:
        marker.skip("gemm_ag requires CustomAllReduceV2 with multicast")
    all_reduce.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    register_comm_cleanup(comm)
    return comm


@cache_once
def _weight() -> torch.Tensor:
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    g = torch.Generator(device="cpu").manual_seed(1234)
    return (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16).to(device)


def _make_inputs(m: int):
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    g = torch.Generator(device="cpu").manual_seed(500 + m)
    x = (torch.randn(m, K, generator=g) * 0.05).to(torch.bfloat16).to(device)
    b = torch.randn(m, N, generator=g).to(torch.bfloat16).to(device)
    c = torch.randn(m, N, generator=g).to(torch.bfloat16).to(device)
    return x, b, c


@marker.parametrize("bs", BATCH_SIZES, [1, 8])
@marker.benchmark("provider", PROVIDERS)
def benchmark(bs: int, provider: str):
    comm = _init_comm()
    world = comm.world_size
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    gpu_group = _init_nccl_group()
    weight = _weight()
    x, b, c = _make_inputs(bs)
    out = torch.empty(bs, N, dtype=torch.bfloat16, device=device)

    if provider == "cublas":

        def fn(x_, w_):
            torch.nn.functional.linear(x_, w_, out=out)

        input_args = (x, weight)
    elif provider == "cublas+add3":

        def fn(x_, w_, b_, c_):
            # mirrors the production tail: GEMM then the 3-way JIT add with
            # b/c prefetched before the PDL wait (their producers are old)
            y = torch.nn.functional.linear(x_, w_)
            add3(y, b_, c_, prefetch_bc=True)

        input_args = (x, weight, b, c)
    else:  # gemm_ag

        def fn(x_, w_, b_, c_):
            gemm_ag.gemm_ag_up_proj(
                world, x_, w_, b_, c_, out, ws_mc_base=comm.mc_base_ptr
            )

        input_args = (x, weight, b, c)

    # graph_clone_args="all" rotates x/weight/b/c copies inside the capture,
    # so the weight is HBM-cold on every call; out and the workspace stay
    # fixed via the closure. gemm_ag only READS its rank's row block of the
    # cloned weight — the traffic-relevant bytes.
    return marker.do_bench(
        fn,
        input_args=input_args,
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=sum(t.nbytes for t in (*input_args, out)),
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=(8,),
        main_fn=benchmark.run,
    )
