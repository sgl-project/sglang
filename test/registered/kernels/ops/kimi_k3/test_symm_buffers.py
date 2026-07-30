"""Invariants of the named persistent symmetric buffers behind the K3 fused pull.

The NVLS pull reduces in place on its input's multicast alias, so every rank must
resolve the same ``(buffer, offset)``. These tests pin the properties the design
rests on:

* a slice's offset is a pure function of its row count -- in particular it does
  not depend on rank-local allocator state, which is exactly what broke when the
  buffer came from a per-forward symm-pool allocation instead;
* every rank resolves the same offset, including after rank-asymmetric memory
  churn (a real server has that: routing-dependent temporaries differ per rank);
* a buffer is created once per name, so no rendezvous runs on the hot path;

Usage::

    python test/registered/kernels/ops/kimi_k3/test_symm_buffers.py
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
from sglang.srt.layers import k3_ar_fusion as ar
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(
    est_time=60,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

HIDDEN = 7168
MOE_WIDTH = 3584 + HIDDEN
# The buffers size themselves from the server args on first use; keep the bound
# small here so the test's buffers stay small. What the tests check is the
# offsets and the reporting, not the number.
MAX_ROWS = 64


@cache_once
def _init_ctx():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    ps._TP = coord  # symm_buffer takes the group from the TP group
    server_args = ServerArgs(model_path="dummy")
    server_args.chunked_prefill_size = MAX_ROWS
    set_global_server_args_for_scheduler(server_args)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return coord.cpu_group


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _buf(name: str, rows: int, width: int):
    _init_ctx()
    return ar.symm_buffer(name, rows, width, torch.bfloat16)


def _all_gather_ints(values: list[int]) -> list[list[int]]:
    t = torch.tensor(values, dtype=torch.int64, device=_device())
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return [o.tolist() for o in out]


_CASES = ((ar.ATTN_O_PROJ, HIDDEN), (ar.MOE_LATENT_SHARED, MOE_WIDTH))


@pytest.mark.parametrize("name,width", _CASES)
@torch.inference_mode()
def test_buffer_is_created_once_and_multicast_capable(name: str, width: int):
    first = _buf(name, 1, width)
    mc = ar.get_mc_ptr(first)
    assert mc != 0
    # keyed by name: a second call reuses the same storage, so nothing is
    # allocated and no rendezvous runs again
    again = _buf(name, 1, width)
    assert again.data_ptr() == first.data_ptr()
    assert ar.get_mc_ptr(again) == mc


@pytest.mark.parametrize("name,width", _CASES)
@torch.inference_mode()
def test_offsets_are_uniform_across_ranks(name: str, width: int):
    base = ar.get_mc_ptr(_buf(name, 1, width))
    offsets = []
    for rows in (1, 5, 12, MAX_ROWS):
        view = _buf(name, rows, width)
        assert view.shape == (rows, width)
        offsets.append(ar.get_mc_ptr(view) - base)
    per_rank = _all_gather_ints(offsets)
    assert all(o == per_rank[0] for o in per_rank), per_rank
    # one slice per name, always from the buffer's base, so the offset carries no
    # allocator state at all
    assert offsets == [0, 0, 0, 0], offsets


@torch.inference_mode()
def test_offsets_survive_rank_asymmetric_churn():
    """Rank-local allocator state must not reach the offset.

    A real server's ranks have different VA histories (mxfp4's routing-dependent
    temporaries, for one). Under the old per-forward symm-pool allocation that was
    enough to make ranks pick different segments; here it must change nothing.
    """
    bases = [ar.get_mc_ptr(_buf(n, 1, w)) for n, w in _CASES]
    rank = dist.get_rank()
    churn = [
        torch.empty((7 + rank) << 20, dtype=torch.uint8, device=_device())
        for _ in range(rank + 1)
    ]
    del churn
    torch.cuda.synchronize()
    deltas = [ar.get_mc_ptr(_buf(n, 12, w)) - b for (n, w), b in zip(_CASES, bases)]
    per_rank = _all_gather_ints(deltas)
    assert all(d == per_rank[0] for d in per_rank), per_rank


@torch.inference_mode()
def test_slices_are_aligned_and_the_two_buffers_are_disjoint():
    a = _buf(ar.ATTN_O_PROJ, MAX_ROWS, HIDDEN)
    b = _buf(ar.MOE_LATENT_SHARED, MAX_ROWS, MOE_WIDTH)
    for t in (a, b):
        assert t.data_ptr() % 16 == 0
        assert t.is_contiguous()
    a_lo, a_hi = a.data_ptr(), a.data_ptr() + a.numel() * 2
    b_lo, b_hi = b.data_ptr(), b.data_ptr() + b.numel() * 2
    # the attention output and the MoE latent|shared buffer are live at the same
    # time, so they must not overlap
    assert a_hi <= b_lo or b_hi <= a_lo, (hex(a_lo), hex(a_hi), hex(b_lo), hex(b_hi))


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
