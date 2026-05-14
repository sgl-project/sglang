from __future__ import annotations

import itertools
import os
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import triton

from sglang.jit_kernel.all_reduce import fused_parallel_qknorm
from sglang.jit_kernel.tests.test_custom_all_reduce import multiprocess_test
from sglang.jit_kernel.tests.utils import multiprocess_main
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    suite="stage-b-kernel-unit-8-gpu-h200",
)
register_cuda_ci(
    est_time=300,
    suite="nightly-kernel-8-gpu-h200",
    nightly=True,
)


Q_K_DIMS = [(6144, 1024)]
EPS = 1e-6
BATCH_SIZES = [2**n for n in range(0, 14)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_CONFIG = list(itertools.product(Q_K_DIMS, BATCH_SIZES, DTYPES))


@pytest.mark.parametrize("nproc", [2, 4, 8])
def test_tp_qknorm(nproc: int) -> None:
    device_count = torch.cuda.device_count()
    if device_count < nproc:
        pytest.skip(
            f"Requires at least {nproc} GPUs, but only {device_count} available"
        )
    multiprocess_test(__file__, nproc)


def init_distributed():
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )

    cpu_group = coord.cpu_group
    nccl_group = coord.device_group
    assert nccl_group is not None

    max_pull_size = 0
    max_push_size = 8 * max(BATCH_SIZES)
    comm = CustomAllReduceV2(cpu_group, device, max_pull_size, max_push_size)
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")

    return rank, world_size, device, cpu_group, nccl_group, comm


def _all_gather_cat(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=-1)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    scale = (x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (x_fp32 * scale * weight.float()).to(x.dtype)


@torch.inference_mode()
def worker_test(
    rank: int,
    world_size: int,
    device: torch.device,
    nccl_group: dist.ProcessGroup,
    comm,
    q_k_dim: tuple[int, int],
    batch_size: int,
    dtype: torch.dtype,
) -> Optional[RuntimeError]:
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

    fused_parallel_qknorm(
        comm.obj,
        q,
        k,
        q_weight,
        k_weight,
        EPS,
    )

    try:
        triton.testing.assert_close(q, q_expected, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(k, k_expected, atol=1e-2, rtol=1e-2)
    except AssertionError as err:
        return RuntimeError(
            f"TP QKNorm mismatch for {batch_size=}, {dtype=}, {world_size=}, {rank=}: {err}"
        )
    return None


def worker_main() -> None:
    rank, world_size, device, cpu_group, nccl_group, comm = init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())

    for q_k_dim, batch_size, dtype in TEST_CONFIG:
        error = worker_test(
            rank,
            world_size,
            device,
            nccl_group,
            comm,
            q_k_dim,
            batch_size,
            dtype,
        )
        result = torch.tensor([int(error is not None)])
        dist.all_reduce(result, group=cpu_group)
        if error is not None:
            print(str(error))
        if bool(result.item()):
            raise RuntimeError(
                f"TP QKNorm test failed for {q_k_dim=}, {batch_size=}, {dtype=}, {world_size=}"
            )

    print(f"Rank {rank} passed all tests.")
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    multiprocess_main(__file__, worker_main)
