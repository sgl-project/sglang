"""CustomAllReduceV2 fused all-reduce + residual add + RMSNorm (gfx1250, 2shot_push)
against the plain v2 all-reduce followed by an fp32 add + norm; every rank must
match bit for bit, eagerly and under CUDA-graph replay.

Run with ``python <this file>``; it relaunches under torchrun on 8 GPUs.
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
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx1250_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main

# No MI450 runner pool yet; skips elsewhere.
register_amd_ci(est_time=60, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

pytestmark = pytest.mark.skipif(not is_gfx1250_supported(), reason="gfx1250 only")

H = 7168
EPS = 1e-6
ROWS = [5, 8, 9, 16, 31, 64, 100, 256, 512]


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
    comm = CustomAllReduceV2(coord.cpu_group, torch.device(f"cuda:{local_rank}"))
    assert not comm.disabled
    register_comm_cleanup(comm)
    return coord, comm


def _gather(t: torch.Tensor) -> list[torch.Tensor]:
    coord, _ = _init()
    out = [torch.empty_like(t) for _ in range(coord.world_size)]
    dist.all_gather(out, t, group=coord.device_group)
    return out


def _inputs(rows: int, seed: int):
    coord, _ = _init()
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = (4 * torch.randn(rows, H, device="cuda", generator=g)).bfloat16()
    gamma = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
    g.manual_seed(seed * 100 + coord.rank + 1)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    return x, residual, gamma


def _check(out, residual_out, x, residual, gamma):
    _, comm = _init()
    ref_res = comm.custom_all_reduce(x).float() + residual.float()
    ref_out = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + EPS)
    ref_out = ref_out * gamma.float()
    assert torch.equal(residual_out, ref_res.bfloat16())
    torch.testing.assert_close(out.float(), ref_out, rtol=1e-2, atol=1e-2)
    for name, t in (("out", out), ("residual_out", residual_out)):
        for i, other in enumerate(_gather(t)):
            assert torch.equal(other, t), f"{name} differs on rank {i}"


@pytest.mark.parametrize("rows", ROWS)
def test_fused_all_reduce_rmsnorm(rows: int):
    _, comm = _init()
    x, residual, gamma = _inputs(rows, rows)
    assert comm.uses_two_shot_push(x.numel() * x.element_size())
    # repeated calls cycle the push epoch and the slot reset
    for _ in range(3):
        out, residual_out = comm.fused_allreduce_rmsnorm(x, residual, gamma, EPS)
        _check(out, residual_out, x, residual, gamma)


@pytest.mark.parametrize("rows", [8, 64])
def test_fused_all_reduce_rmsnorm_cuda_graph(rows: int):
    _, comm = _init()
    x, residual, gamma = _inputs(rows, rows)
    static_x, static_residual = torch.zeros_like(x), torch.zeros_like(residual)
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        comm.fused_allreduce_rmsnorm(static_x, static_residual, gamma, EPS)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            out, residual_out = comm.fused_allreduce_rmsnorm(
                static_x, static_residual, gamma, EPS
            )
    static_x.copy_(x)
    static_residual.copy_(residual)
    graph.replay()
    torch.cuda.synchronize()
    _check(out, residual_out, x, residual, gamma)


def test_one_shot_sizes_are_not_fused():
    _, comm = _init()
    x, residual, gamma = _inputs(1, 0)
    assert comm.uses_one_shot_push(x.numel() * x.element_size())
    assert comm.fused_allreduce_rmsnorm(x, residual, gamma, EPS) is None


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
