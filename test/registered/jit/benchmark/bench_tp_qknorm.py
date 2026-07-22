"""Benchmark fused TP QKNorm (push-mode custom-AR + RMSNorm) vs the serial
baseline (RMS sum-sq -> pull-mode all-reduce -> RMS apply).

Usage::

    # Benchmark on every supported world size (2..8 GPUs):
    python benchmark/bench_tp_qknorm.py
    # Specific world sizes:
    python benchmark/bench_tp_qknorm.py --num-gpu 4
    python benchmark/bench_tp_qknorm.py --num-gpu 2,4,8
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
from multiprocessing.context import SpawnProcess
from typing import List

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import multigpu_bench_main
from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.kernels.ops.communication.all_reduce import (
    fused_parallel_qknorm,
    get_all_reduce_module,
    get_fused_parallel_qknorm_max_occupancy,
    get_fused_parallel_qknorm_module,
)
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)
register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
EPS = 1e-6
Q_K_DIMS = [(6144, 1024)]
BATCH_SIZES = get_ci_test_range([2**i for i in range(15)], [1, 64, 1024])
MAX_PUSH_SIZE = 8 * max(BATCH_SIZES)
PROVIDERS = ["fused", "baseline"]


# ---------------------------------------------------------------------------
# Parallel JIT precompile (outer process, before any torchrun child starts)
# ---------------------------------------------------------------------------


def _compile_one(world_size: int) -> None:
    """Compile every kernel this bench touches for a single world_size.

    Top-level so it survives ``spawn`` pickling. Compiled artifacts are
    cached on disk by ``tvm_ffi``; torchrun children will reuse them.
    """
    # baseline path: sum-sq -> all-reduce -> apply (also covers push mode)
    get_all_reduce_module(DTYPE, world_size)
    # fused path: fused QKNorm kernel (one per (dtype, world_size, q_dim, k_dim))
    for q_dim, k_dim in Q_K_DIMS:
        get_fused_parallel_qknorm_module(DTYPE, world_size, q_dim, k_dim)


def _precompile_kernels(num_gpus: List[int]) -> None:
    ctx = multiprocessing.get_context("spawn")
    procs: list[tuple[int, SpawnProcess]] = []
    for world_size in num_gpus:
        p = ctx.Process(target=_compile_one, args=(world_size,))
        p.start()
        procs.append((world_size, p))
    for world_size, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"TP QKNorm precompile failed for {world_size=} " f"(exit {p.exitcode})"
            )


# ---------------------------------------------------------------------------
# Per-rank distributed init (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
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
def _init_gpu_group() -> dist.ProcessGroup:
    _init_cpu_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    gpu_group = dist.new_group(backend="nccl", device_id=device)
    assert isinstance(gpu_group, dist.ProcessGroup)
    atexit.register(lambda: dist.destroy_process_group(gpu_group))
    return gpu_group


@cache_once
def _init_fused_comm() -> CustomAllReduceV2:
    """Push-mode workspace sized for the fused-QKNorm bench."""
    cpu_group = _init_cpu_group()
    world_size = dist.get_world_size(cpu_group)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    q_dim, k_dim = Q_K_DIMS[0]
    max_occupancy = get_fused_parallel_qknorm_max_occupancy(
        DTYPE, world_size, q_dim, k_dim
    )
    if dist.get_rank(cpu_group) == 0:
        print(f"Max occupancy for fused_parallel_qknorm: {max_occupancy} blocks/SM")
    props = torch.cuda.get_device_properties(device)
    comm = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=0,
        max_push_size=MAX_PUSH_SIZE,
        max_push_blocks=props.multi_processor_count * max_occupancy,
    )
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_baseline_comm() -> CustomAllReduceV2:
    """Default (pull-mode) workspace for the serial baseline."""
    cpu_group = _init_cpu_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _rmsnorm_baseline(
    comm: CustomAllReduceV2,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    world_size: int,
) -> None:
    from sglang.srt.models.minimax_m2 import rms_apply_serial, rms_sumsq_serial

    sum_sq = rms_sumsq_serial(q, k)
    sum_sq = comm.custom_all_reduce(sum_sq)
    rms_apply_serial(q, k, q_weight, k_weight, sum_sq, world_size, EPS)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@marker.parametrize("q_dim,k_dim", Q_K_DIMS)
@marker.parametrize("batch_size", BATCH_SIZES)
@marker.benchmark("provider", PROVIDERS)
def benchmark(q_dim: int, k_dim: int, batch_size: int, provider: str):
    cpu_group = _init_cpu_group()
    gpu_group = _init_gpu_group()
    world_size = dist.get_world_size(cpu_group)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    local_q_dim = q_dim // world_size
    local_k_dim = k_dim // world_size

    q = torch.randn(batch_size, local_q_dim, device=device, dtype=DTYPE)
    k = torch.randn(batch_size, local_k_dim, device=device, dtype=DTYPE)
    q_weight = torch.randn(local_q_dim, device=device, dtype=DTYPE)
    k_weight = torch.randn(local_k_dim, device=device, dtype=DTYPE)

    if provider == "fused":
        comm = _init_fused_comm()

        def fn(q, k, q_weight, k_weight):
            fused_parallel_qknorm(comm.obj, q, k, q_weight, k_weight, EPS)

    else:
        comm = _init_baseline_comm()

        def fn(q, k, q_weight, k_weight):
            _rmsnorm_baseline(comm, q, k, q_weight, k_weight, world_size)

    return marker.do_bench(
        fn,
        input_args=(q, k, q_weight, k_weight),
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        memory_output=(q, k),  # NOTE: In-place updates on q, k;
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=[2, 4, 8],  # NOTE: don't support other world size now
        main_fn=benchmark.run,
        pre_launch_fn=_precompile_kernels,
    )
