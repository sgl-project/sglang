from __future__ import annotations

import atexit
import itertools
import logging
import multiprocessing
import os
from multiprocessing.context import SpawnProcess
from typing import List

import pytest
import torch
import torch.distributed as dist
import triton

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.all_reduce import (
    _jit_custom_all_reduce_push_module,
    _jit_fused_parallel_qknorm_module,
    fused_parallel_qknorm,
)
from sglang.jit_kernel.mp import register_comm_cleanup
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    stage="extra-b",
    runner_config="8-gpu-h200",
)
register_cuda_ci(
    est_time=300,
    suite="nightly-kernel-8-gpu-h200",
    nightly=True,
)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

Q_K_DIMS = [(6144, 1024)]
EPS = 1e-6
BATCH_SIZES = [2**n for n in range(0, 14)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


# ---------------------------------------------------------------------------
# Parallel JIT precompile (outer process, before any torchrun child starts)
# ---------------------------------------------------------------------------


def _compile_one(dtype: torch.dtype, world_size: int) -> None:
    """Compile every kernel this test touches for one (dtype, world_size).

    Top-level so it survives ``spawn`` pickling. Compiled artifacts are
    cached on disk by ``tvm_ffi``; torchrun children will reuse them.
    """
    _jit_custom_all_reduce_push_module(dtype, world_size)
    for q_dim, k_dim in Q_K_DIMS:
        _jit_fused_parallel_qknorm_module(dtype, world_size, q_dim, k_dim)


def _precompile_kernels(num_gpus: List[int]) -> None:
    ctx = multiprocessing.get_context("spawn")
    procs: list[tuple[torch.dtype, int, SpawnProcess]] = []
    for dtype, world_size in itertools.product(DTYPES, num_gpus):
        p = ctx.Process(target=_compile_one, args=(dtype, world_size))
        p.start()
        procs.append((dtype, world_size, p))
    for dtype, world_size, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"TP QKNorm precompile failed for {dtype=} {world_size=} "
                f"(exit {p.exitcode})"
            )


# ---------------------------------------------------------------------------
# Per-rank distributed setup (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group_once() -> dist.ProcessGroup:
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
    cpu_group = coord.cpu_group
    assert isinstance(cpu_group, dist.ProcessGroup)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return cpu_group


@cache_once
def _init_nccl_group_once() -> dist.ProcessGroup:
    _init_cpu_group_once()
    coord = ps._WORLD
    assert coord is not None and coord.device_group is not None
    return coord.device_group


@cache_once
def _init_comm_once() -> CustomAllReduceV2:
    cpu_group = _init_cpu_group_once()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    max_pull_size = 0
    max_push_size = 8 * max(BATCH_SIZES)
    comm = CustomAllReduceV2(cpu_group, device, max_pull_size, max_push_size)
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_gather_cat(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=-1)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    scale = (x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (x_fp32 * scale * weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("q_k_dim", Q_K_DIMS)
@torch.inference_mode()
def test_tp_qknorm(
    q_k_dim: tuple[int, int],
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    nccl_group = _init_nccl_group_once()
    comm = _init_comm_once()
    rank = dist.get_rank(group=nccl_group)
    world_size = dist.get_world_size(group=nccl_group)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    q_dim, k_dim = q_k_dim
    local_q_dim = q_dim // world_size
    local_k_dim = k_dim // world_size

    q = torch.randn(batch_size, local_q_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, local_k_dim, device=device, dtype=dtype)
    q_weight = torch.randn(local_q_dim, device=device, dtype=dtype)
    k_weight = torch.randn(local_k_dim, device=device, dtype=dtype)

    q_ref = _all_gather_cat(q, nccl_group)
    k_ref = _all_gather_cat(k, nccl_group)
    q_weight_ref = _all_gather_cat(q_weight.unsqueeze(0), nccl_group).squeeze(0)
    k_weight_ref = _all_gather_cat(k_weight.unsqueeze(0), nccl_group).squeeze(0)

    q_expected = _rmsnorm_ref(q_ref, q_weight_ref, EPS)
    k_expected = _rmsnorm_ref(k_ref, k_weight_ref, EPS)
    q_expected = q_expected[:, rank * local_q_dim : (rank + 1) * local_q_dim]
    k_expected = k_expected[:, rank * local_k_dim : (rank + 1) * local_k_dim]

    fused_parallel_qknorm(comm.obj, q, k, q_weight, k_weight, EPS)

    triton.testing.assert_close(q, q_expected, atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(k, k_expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(2, 4, 8),
        pre_launch_fn=_precompile_kernels,
    )
