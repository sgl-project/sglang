"""Correctness of the NVLink collectives (``nvlink_comm``) against NCCL.

All-reduce, all-gather and reduce-scatter on the push and pull planes of a
``CustomAllReduceV2`` communicator, with and without the folded residual, plus
the two copy-engine all-gathers, over token counts that cover the ragged split
(7), the remainder loop alone (1) and the bandwidth band (1024). The residual
is the same on every rank, as it is in a TP layer: the pull all-reduce folds
it in over each rank's token slice.

Usage::

    python test/registered/kernels/ops/communication/test_nvlink_comm.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import os
from typing import Dict, Tuple

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import nvlink_comm as nvl
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="base-c", runner_config="4-gpu-b200")

HIDDEN = 7168
DTYPE = torch.bfloat16
PUSH_SLOT_MB = 32
PULL_MB = 4


def _device() -> torch.device:
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    atexit.register(dist.destroy_process_group)
    nccl_group = dist.new_group(backend="nccl", device_id=_device())
    return coord.cpu_group, nccl_group


@cache_once
def _init_comm():
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    cpu_group, _ = _init_world()
    comm = CustomAllReduceV2(
        cpu_group,
        _device(),
        max_push_size=PUSH_SLOT_MB << 20,
        max_pull_size=PULL_MB << 20,
    )
    if comm.disabled:
        pytest.skip("CustomAllReduceV2 is disabled on this system")
    if not comm.has_multicast:
        pytest.skip("the nvlink collectives need a multicast plane")
    register_comm_cleanup(comm)
    return comm


_SYMM: Dict[Tuple[int, int], torch.Tensor] = {}


def _symm(shape: Tuple[int, int]) -> torch.Tensor:
    """Symmetric memory with a multicast alias; one allocation per shape, since
    the allocation is collective and never returned."""
    from torch._C._distributed_c10d import _SymmetricMemory

    if shape not in _SYMM:
        cpu_group, _ = _init_world()
        t = _SymmetricMemory.empty_strided_p2p(
            (shape[0] * shape[1],), [1], DTYPE, _device(), cpu_group.group_name
        )
        _SymmetricMemory.rendezvous(t)
        _SYMM[shape] = t.view(shape)
    return _SYMM[shape]


def _shapes(op: str, tokens: int, world_size: int):
    if op == "all_gather":
        return (tokens, HIDDEN), (tokens * world_size, HIDDEN)
    if op == "reduce_scatter":
        return (tokens * world_size, HIDDEN), (tokens, HIDDEN)
    return (tokens, HIDDEN), (tokens, HIDDEN)


def _reference(op, x, residual, nccl_group, world_size):
    """fp32 NCCL reference with the kernels' residual placement: the gather adds
    it to this rank's shard before gathering, the reductions to the output."""
    x = x.float()
    res = residual.float() if residual is not None else 0
    if op == "all_reduce":
        y = x.clone()
        dist.all_reduce(y, group=nccl_group)
        return y + res
    if op == "all_gather":
        x = (x + res).contiguous()
        out = torch.empty(
            (x.shape[0] * world_size, HIDDEN), dtype=torch.float32, device=x.device
        )
        dist.all_gather_into_tensor(out, x, group=nccl_group)
        return out
    out = torch.empty(
        (x.shape[0] // world_size, HIDDEN), dtype=torch.float32, device=x.device
    )
    dist.reduce_scatter_tensor(out, x.contiguous(), group=nccl_group)
    return out + res


_FNS = {
    ("all_reduce", "push"): nvl.all_reduce_push,
    ("all_gather", "push"): nvl.all_gather_push,
    ("reduce_scatter", "push"): nvl.reduce_scatter_push,
    ("all_reduce", "pull"): nvl.all_reduce_pull,
    ("all_gather", "pull"): nvl.all_gather_pull,
    ("reduce_scatter", "pull"): nvl.reduce_scatter_pull,
}


@pytest.mark.parametrize("tokens", [1, 7, 128, 1024])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("plane", ["push", "pull"])
@pytest.mark.parametrize("op", nvl.SUPPORTED_OPS)
def test_collective(op: str, plane: str, residual: bool, tokens: int) -> None:
    cpu_group, nccl_group = _init_world()
    comm = _init_comm()
    world_size = dist.get_world_size(cpu_group)
    rank = dist.get_rank(cpu_group)
    device = _device()
    in_shape, out_shape = _shapes(op, tokens, world_size)
    sym_in, sym_out = _symm(in_shape), _symm(out_shape)
    gen = torch.Generator(device=device).manual_seed(1000 * tokens + rank)
    sym_in.copy_(torch.randn(in_shape, dtype=DTYPE, device=device, generator=gen))
    sym_out.zero_()
    res = None
    if residual:
        shared = torch.Generator(device=device).manual_seed(7 * tokens)
        res_shape = in_shape if op == "all_gather" else out_shape
        res = torch.randn(res_shape, dtype=DTYPE, device=device, generator=shared)
    ref = _reference(op, sym_in, res, nccl_group, world_size)
    dist.barrier(nccl_group)
    torch.cuda.synchronize()
    _FNS[(op, plane)](comm.obj, sym_in, sym_out, res)
    torch.cuda.synchronize()
    dist.barrier(nccl_group)
    if op == "all_gather":
        # a pure copy (plus one bf16 add with the residual): bit-exact
        torch.testing.assert_close(
            sym_out.float(), ref.to(DTYPE).float(), atol=0, rtol=0
        )
    else:
        # bf16 sums in a different order than NCCL's fp32 tree
        torch.testing.assert_close(sym_out.float(), ref, atol=0.1, rtol=0.02)


@pytest.mark.parametrize("tokens", [1, 7, 128])
@pytest.mark.parametrize("variant", ["multicast", "unicast"])
def test_copy_engine_all_gather(variant: str, tokens: int) -> None:
    cpu_group, nccl_group = _init_world()
    comm = _init_comm()
    world_size = dist.get_world_size(cpu_group)
    device = _device()
    in_shape, out_shape = _shapes("all_gather", tokens, world_size)
    sym_in, sym_out = _symm(in_shape), _symm(out_shape)
    gen = torch.Generator(device=device).manual_seed(
        50 * tokens + dist.get_rank(cpu_group)
    )
    sym_in.copy_(torch.randn(in_shape, dtype=DTYPE, device=device, generator=gen))
    sym_out.zero_()
    ref = _reference("all_gather", sym_in, None, nccl_group, world_size)
    dist.barrier(nccl_group)
    torch.cuda.synchronize()
    if variant == "multicast":
        nvl.all_gather_copy_engine_multicast(comm.obj, sym_in, sym_out)
    else:
        flags = nvl.make_ce_flags(cpu_group, world_size)
        nvl.all_gather_copy_engine_unicast(comm.obj, sym_in, sym_out, ce_flags=flags)
    torch.cuda.synchronize()
    dist.barrier(nccl_group)
    torch.testing.assert_close(sym_out.float(), ref, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
