"""Benchmark the K3 low-SM NVLS pull all-reduce (ar_fusion pull) against
NCCL with and without symmetric-window registration, at Kimi-K3 sizes
(bs x 7168 bf16), plus the fused-RMSNorm pull on the latent|shared MoE
buffer.

Bare pull providers use the tuned per-size (num_blocks, unroll) defaults; a
``_bN`` suffix forces a block count (of 512 threads each).
"""

from __future__ import annotations

import atexit
import logging
import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import multigpu_bench_main
from sglang.jit_kernel.kimi_k3 import all_reduce
from sglang.jit_kernel.mp import register_comm_cleanup
from sglang.jit_kernel.utils import cache_once
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires 8 GPUs with NVLS multicast, self-skips in CI",
)

H = 7168
NORM_DIM = 3584
MB = 1024 * 1024
BATCH_SIZES = [1, 8, 32, 64, 128, 1024, 4096]
NORM_TOKENS = [8, 64, 512, 1024, 4096]
PROVIDERS = ["pull2s", "pull2s_b4", "pull2s_b8", "nccl_symm", "nccl"]


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
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group():
    _init_world()
    local_rank = int(os.environ["LOCAL_RANK"])
    return dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))


def _symm_alloc_mc(shape, dtype):
    import torch.distributed._symmetric_memory as torch_symm_mem

    cpu_group = _init_world()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    pool = torch_symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        buf = torch.empty(shape, dtype=dtype, device=device)
    hdl = torch_symm_mem.rendezvous(buf, cpu_group.group_name)
    if hdl.multicast_ptr == 0:
        marker.skip("the pull kernels require NVLS multicast symmetric memory")
    mc = hdl.multicast_ptr + (buf.data_ptr() - hdl.buffer_ptrs[rank])
    return buf, mc


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(
        cpu_group, device, max_pull_size=1 * MB, max_push_size=2 * MB
    )
    if comm.disabled or comm.mc_base_ptr == 0:
        marker.skip("the pull kernels require CustomAllReduceV2 with multicast")
    all_reduce.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_pool_buf():
    # 1.5x headroom: the norm table views the buffer as [N, 3584 + 7168]
    return _symm_alloc_mc((max(BATCH_SIZES) * H * 3 // 2,), torch.bfloat16)


@cache_once
def _init_pynccl_symm():
    """A PyNcclCommunicator plus a max-size buffer allocated from the NCCL
    symmetric mem pool (ncclMemAlloc) and window-registered with the comm, so
    ncclAllReduce on it takes NCCL's symmetric (NVLS) fast path."""
    from sglang.srt.distributed.device_communicators import pynccl_allocator
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = PyNcclCommunicator(cpu_group, device)
    pool = pynccl_allocator.get_nccl_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        buf = torch.empty(max(BATCH_SIZES) * H, dtype=torch.bfloat16, device=device)
    result = pynccl_allocator._register_func(comm.comm.value)
    if result != 0:
        marker.skip(f"nccl window registration failed: {result}")
    comm.disabled = False
    return comm, buf


@marker.parametrize("bs", BATCH_SIZES, [1, 64, 1024])
@marker.benchmark("provider", PROVIDERS)
def benchmark(bs: int, provider: str):
    gpu_group = _init_nccl_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    world = dist.get_world_size()
    n = bs * H
    nbytes = n * 2

    if provider.startswith("pull"):
        _init_comm()
        num_blocks = int(provider.rsplit("_b", 1)[1]) if "_b" in provider else None
        buf, mc = _init_pool_buf()
        x = buf[:n]
        x.normal_()

        def fn(t):
            all_reduce.all_reduce_pull_res(
                world, t, None, input_mc_ptr=mc, num_blocks=num_blocks
            )

        clone = None  # the symm buffer must not be cloned
    elif provider == "nccl_symm":
        comm, sbuf = _init_pynccl_symm()
        x = sbuf[:n]
        x.normal_()

        def fn(t):
            comm.all_reduce(t)

        clone = None  # the registered symm buffer must not be cloned
    else:  # nccl
        x = torch.randn(n, dtype=torch.bfloat16, device=device)

        def fn(t):
            dist.all_reduce(t, group=gpu_group)

        clone = (0,)

    effective_bytes = int(nbytes * 2 * (world - 1) / world)
    return marker.do_bench(
        fn,
        input_args=(x,),
        graph_clone_args=clone,
        sync_multigpu_fn=lambda: dist.barrier(_init_nccl_group()),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=effective_bytes,
    )


def _nccl_then_norm(x: torch.Tensor, num_tokens: int, weight, gpu_group):
    dist.all_reduce(x, group=gpu_group)
    latent = x[: num_tokens * NORM_DIM].view(num_tokens, NORM_DIM)
    f32 = latent.float()
    factor = torch.rsqrt(f32.pow(2).mean(-1, keepdim=True) + 1e-6)
    latent.copy_((f32 * factor * weight.float()).to(torch.bfloat16))


@marker.parametrize("tokens", NORM_TOKENS, [64, 4096])
@marker.benchmark("provider", ["pulln", "nccl_norm"])
def benchmark_norm(tokens: int, provider: str):
    """AR + fused RMSNorm on the K3 latent|shared MoE buffer ([tokens, 3584]
    latent + [tokens, 7168] shared); the baseline is NCCL followed by an
    unfused torch-eager RMSNorm on the latent."""
    gpu_group = _init_nccl_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    world = dist.get_world_size()
    n = tokens * 3 * NORM_DIM
    weight = torch.randn(NORM_DIM, dtype=torch.bfloat16, device=device)

    if provider == "pulln":
        _init_comm()
        buf, mc = _init_pool_buf()
        x = buf[:n]
        x.normal_()

        def fn(t):
            all_reduce.all_reduce_pull_norm(
                world, t, weight, 1e-6, num_norm_rows=tokens, input_mc_ptr=mc
            )

        clone = None
    else:  # nccl + unfused norm
        x = torch.randn(n, dtype=torch.bfloat16, device=device)

        def fn(t):
            _nccl_then_norm(t, tokens, weight, gpu_group)

        clone = (0,)

    effective_bytes = int(n * 2 * 2 * (world - 1) / world)
    return marker.do_bench(
        fn,
        input_args=(x,),
        graph_clone_args=clone,
        sync_multigpu_fn=lambda: dist.barrier(_init_nccl_group()),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=effective_bytes,
    )


def _main():
    benchmark.run()
    benchmark_norm.run()


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=(8,),
        main_fn=_main,
    )
