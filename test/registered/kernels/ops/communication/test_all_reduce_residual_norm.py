"""Push-plane all-reduce + residual add + RMSNorm, plain and MoE-finalize, against
an fp32 reference at each 1shot / 2shot geometry; every rank must match bit for bit.

Run with ``python <this file>``; it relaunches under torchrun on 4 GPUs.
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import all_reduce_fusion
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-b200")

H = 6144
TOP_K = 8
EPS = 1e-6
# The last row count of each geometry bucket in `_TWO_SHOT_GEOMETRY` (ws=4), where
# its residency is tightest; 341 rows fill the 4 MB plane.
PLAIN_ROWS = [47, 272, 341]
FINALIZE_ROWS = [47, 208, 232, 341]


@cache_once
def _init():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    assert not comm.disabled
    all_reduce_fusion.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return coord


def _gather(t: torch.Tensor) -> list[torch.Tensor]:
    coord = _init()
    out = [torch.empty_like(t) for _ in range(coord.world_size)]
    dist.all_gather(out, t, group=coord.device_group)
    return out


def _check(normed, residual_out, partials, residual, gamma):
    total = torch.stack([p.float() for p in _gather(partials)]).sum(0)
    ref_res = total + residual.float()
    ref_norm = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + EPS)
    ref_norm = ref_norm * gamma.float()
    torch.testing.assert_close(residual_out.float(), ref_res, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(normed.float(), ref_norm, rtol=2e-2, atol=2e-2)
    for name, t in (("normed", normed), ("residual_out", residual_out)):
        for i, other in enumerate(_gather(t)):
            assert torch.equal(other, t), f"{name} differs on rank {i}"


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    gamma = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
    return residual, gamma


@pytest.mark.parametrize("rows", PLAIN_ROWS)
def test_all_reduce_residual_norm(rows: int):
    coord = _init()
    residual, gamma = _inputs(rows, 0)
    partial = torch.randn(rows, H, device="cuda").bfloat16() * (coord.rank + 1) / 4
    normed, residual_out = all_reduce_fusion.all_reduce_residual_norm(
        partial,
        residual,
        gamma,
        EPS,
        world_size=coord.world_size,
        residual_out=torch.empty_like(residual),
    )
    torch.cuda.synchronize()
    _check(normed, residual_out, partial, residual, gamma)


@pytest.mark.parametrize("rows", FINALIZE_ROWS)
def test_moe_finalize_all_reduce_residual_norm(rows: int):
    coord = _init()
    residual, gamma = _inputs(rows, 1)
    permuted = rows * TOP_K + 64
    gemm2_out = torch.randn(permuted, H, device="cuda").bfloat16() / 4
    idx = torch.randperm(permuted, device="cuda")[: rows * TOP_K].int()
    idx[:: 2 * TOP_K + 1] = -1  # dropped slots contribute nothing
    weights = torch.rand(rows, TOP_K, device="cuda")
    shared = torch.randn(rows, H, device="cuda").bfloat16() / 4
    normed, residual_out = all_reduce_fusion.moe_finalize_all_reduce_residual_norm(
        gemm2_out,
        idx,
        weights,
        TOP_K,
        shared,
        residual,
        gamma,
        EPS,
        world_size=coord.world_size,
        hidden_dim=H,
        residual_out=torch.empty_like(residual),
    )
    torch.cuda.synchronize()
    rows_k = gemm2_out.float()[idx.clamp(min=0).long()].view(rows, TOP_K, H)
    w = weights * (idx.view(rows, TOP_K) >= 0)
    local = (rows_k * w.unsqueeze(-1)).sum(1) + shared.float()
    _check(normed, residual_out, local, residual, gamma)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
