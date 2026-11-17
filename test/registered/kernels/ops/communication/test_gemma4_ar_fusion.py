"""Gemma-4 fused all-reduce + norm -> residual add -> norm -> per-128 FP8 quant,
against an fp32 reference of the unfused path; every rank must receive the same
rows.

Run with ``python <this file>``; it relaunches under torchrun on 2, 4 and 8 GPUs.
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import gemma4_ar_fusion
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-b200")

HIDDENS = [5376, 4096]  # Gemma-4 31B, and one more block width
EPS = 1e-6
GROUP = gemma4_ar_fusion.GROUP
MAX_ROWS = 1024
# fewer rows than ranks, trailing ranks owning none (5 at TP4, 9 at TP8), ragged
# shards, and a full barrier-plane grid
ROWS = [1, 5, 9, 512, 1024]


class _Symm:
    def __init__(self, shape, dtype, group_name):
        self.tensor = _SymmetricMemory.empty_strided_p2p(
            shape, [shape[1], 1], dtype, torch.device("cuda"), group_name
        )
        handle = _SymmetricMemory.rendezvous(self.tensor)
        assert handle is not None and handle.multicast_ptr != 0, (
            "needs NVLink multicast"
        )
        self.mc = int(handle.multicast_ptr)
        self.tensor.zero_()


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
        max_pull_size=1024 * 1024,
        max_push_size=0,
        max_push_blocks=0,
    )
    assert not comm.disabled and comm.has_multicast, "needs NVLink multicast"
    gemma4_ar_fusion.register_comm(lambda: comm.obj, world_size=comm.world_size)
    register_comm_cleanup(comm)
    return coord


@cache_once
def _buffers(hidden: int):
    coord = _init()
    name = coord.cpu_group.group_name
    return (
        _Symm((MAX_ROWS, hidden), torch.bfloat16, name),
        _Symm((MAX_ROWS, hidden), torch.float8_e4m3fn, name),
        _Symm((MAX_ROWS, hidden // GROUP), torch.float32, name),
    )


def _gather(t: torch.Tensor) -> list[torch.Tensor]:
    coord = _init()
    out = [torch.empty_like(t) for _ in range(coord.world_size)]
    dist.all_gather(out, t, group=coord.device_group)
    return out


def _rows(num_tokens: int, rank: int, world_size: int):
    per_rank = (num_tokens + world_size - 1) // world_size
    row0 = min(rank * per_rank, num_tokens)
    return row0, max(0, min(num_tokens, row0 + per_rank) - row0)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _reference(partials, residual, w_a, w_b, scalar, round_a):
    """The unfused path in fp32 with its bf16 rounding points (see Params.round_a)."""
    x = torch.stack([p.float() for p in partials]).sum(0)
    a = _rmsnorm(x, w_a)
    if round_a:
        a = a.bfloat16().float()
    r = (a + residual.float()) * scalar.float()
    stream = r if round_a else r.bfloat16().float()
    y = _rmsnorm(stream, w_b).bfloat16().float()
    return r.bfloat16(), y


def _inputs(rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(rows, hidden, device="cuda", generator=g).bfloat16()
    w_a = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
    w_b = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
    scalar = torch.full((1,), 0.75, device="cuda")
    return residual, w_a, w_b, scalar


@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("round_a", [True, False])
def test_site(hidden: int, rows: int, round_a: bool):
    coord = _init()
    partial_buf, out_buf, scales_buf = _buffers(hidden)
    residual_all, w_a, w_b, scalar = _inputs(rows, hidden, 0)
    partial = torch.randn(rows, hidden, device="cuda").bfloat16() * (coord.rank + 1) / 4
    row0, mine = _rows(rows, coord.rank, coord.world_size)
    residual = residual_all[row0 : row0 + mine].clone()
    partial_buf.tensor[:rows].copy_(partial)
    gemma4_ar_fusion.site(
        partial_buf.tensor[:rows],
        partial_buf.mc,
        out_buf.tensor[:rows],
        out_buf.mc,
        scales_buf.tensor[:rows],
        scales_buf.mc,
        residual,
        w_a,
        w_b,
        scalar,
        EPS,
        row0,
        mine,
        round_a,
        world_size=coord.world_size,
    )
    torch.cuda.synchronize()
    out, scales = out_buf.tensor[:rows].clone(), scales_buf.tensor[:rows].clone()

    ref_res, ref_y = _reference(
        _gather(partial), residual_all, w_a, w_b, scalar, round_a
    )
    # gather before any assert: a rank that fails alone would hang its peers
    all_out, all_scales = _gather(out), _gather(scales)
    # this rank's residual rows, then every rank's view of the quantized rows
    # multimem.ld_reduce does not round to nearest, so x can sit one bf16 ulp off the
    # fp32 sum; that moves a = bf16(norm(x)) by up to two ulps (2/32 for |a| < 8), times 0.75
    torch.testing.assert_close(
        residual.float(), ref_res[row0 : row0 + mine].float(), rtol=2e-2, atol=5e-2
    )
    deq = out.float() * scales.repeat_interleave(GROUP, dim=1)
    torch.testing.assert_close(
        deq, ref_y, rtol=7e-2, atol=4e-2
    )  # e4m3 has 3 mantissa bits
    assert torch.all(scales > 0)
    for name, t, gathered in (("out", out, all_out), ("scales", scales, all_scales)):
        for i, other in enumerate(gathered):
            assert torch.equal(other, t), f"{name} differs on rank {i}"


@pytest.mark.parametrize("hidden", HIDDENS[:1])
@pytest.mark.parametrize("rows", ROWS)
def test_final(hidden: int, rows: int):
    coord = _init()
    partial_buf, _, _ = _buffers(hidden)
    residual_all, w_a, _, scalar = _inputs(rows, hidden, 1)
    partial = torch.randn(rows, hidden, device="cuda").bfloat16() * (coord.rank + 1) / 4
    row0, mine = _rows(rows, coord.rank, coord.world_size)
    residual = residual_all[row0 : row0 + mine].clone()
    staged = partial_buf.tensor[:rows]
    staged.copy_(partial)
    gemma4_ar_fusion.final(
        staged,
        partial_buf.mc,
        staged,
        partial_buf.mc,
        residual,
        w_a,
        scalar,
        EPS,
        row0,
        mine,
        world_size=coord.world_size,
    )
    torch.cuda.synchronize()
    out = staged.clone()
    ref_res, _ = _reference(_gather(partial), residual_all, w_a, w_a, scalar, False)
    all_out = _gather(out)
    torch.testing.assert_close(out.float(), ref_res.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual, out[row0 : row0 + mine])
    for i, other in enumerate(all_out):
        assert torch.equal(other, out), f"out differs on rank {i}"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
