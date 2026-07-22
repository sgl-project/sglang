"""Benchmark the K3 MNNVL fused all-reduce (ar_fusion) against the generic
CustomAllReduceV2 push and NCCL, at Kimi-K3 sizes (bs x 7168 bf16)."""

from __future__ import annotations

import atexit
import logging
import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.all_reduce import AllReduceAlgo, custom_all_reduce
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
MB = 1024 * 1024
BATCH_SIZES = [1, 8, 32, 64, 128, 1024, 4096]
PROVIDERS = ["arf", "arf+res", "car_v2", "nccl"]


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
        marker.skip("ar_fusion requires CustomAllReduceV2 with multicast")
    all_reduce.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_pool_buf():
    return _symm_alloc_mc((max(BATCH_SIZES) * H,), torch.bfloat16)


# push covers <= 0.5 MB (bs<=32 at H=7168); larger sizes go through the
# in-place NVLS 2shot on the symm-pool buffer — mirrors the serving dispatch
_PUSH_MAX_BYTES = 512 * 1024


@marker.parametrize("bs", BATCH_SIZES, [1, 64, 1024])
@marker.benchmark("provider", PROVIDERS)
def benchmark(bs: int, provider: str):
    comm = _init_comm()
    world = comm.world_size
    gpu_group = _init_nccl_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    n = bs * H
    nbytes = n * 2
    is_push = nbytes <= _PUSH_MAX_BYTES
    residual = torch.randn(n, dtype=torch.bfloat16, device=device)

    if provider in ("arf", "arf+res"):
        res = residual if provider == "arf+res" else None
        if is_push:
            x = torch.randn(n, dtype=torch.bfloat16, device=device)

            def fn(t):
                all_reduce.all_reduce_push_res(
                    world, t, res, ws_mc_base=comm.mc_base_ptr
                )

            clone = (0,)
        else:
            buf, mc = _init_pool_buf()
            x = buf[:n]
            x.normal_()

            def fn(t):
                all_reduce.all_reduce_pull_res(world, t, res, input_mc_ptr=mc)

            clone = None  # the symm buffer must not be cloned
    elif provider == "car_v2":
        if not is_push:
            marker.skip("car_v2 1shot_push covers the small-message regime only")
        x = torch.randn(n, dtype=torch.bfloat16, device=device)

        def fn(t):
            custom_all_reduce(comm.obj, t, AllReduceAlgo.ONE_SHOT_PUSH, False)

        clone = (0,)
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


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=(8,),
        main_fn=benchmark.run,
    )
