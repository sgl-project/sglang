"""K3 ROCm TP8: AITER all_gather_lastdim_add with one cross-GPU barrier.

Bit-identical to ``CustomAllreduce.all_gather_lastdim_add`` but without the
trailing end barrier (-1.6..1.8 us per call on 8x MI355X). The input is held
here until two calls later so the CUDA-graph allocator cannot give its memory
to a kernel that runs before the next collective's start barrier (by then
every peer has finished reading it). Graph capture only, inside AITER's
``capture()`` scope (which registers the input's peer addresses).
See ``jit/csrc/kimi_k3/comm/ag_one_barrier_hip.cuh``.
"""

from __future__ import annotations

import collections
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

WORLD_SIZE = 8

# Keeps each input alive until two calls later, past the next collective's
# start barrier, so the graph allocator cannot reuse it while peers read it.
_KEEPALIVE: collections.deque = collections.deque(maxlen=2)
_EMPTY: dict = {}


@cache_once
def _module():
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "k3_ag_one_barrier_hip",
        cuda_files=["kimi_k3/comm/ag_one_barrier_hip.cuh"],
        cuda_wrappers=[("run", "k3_ag_one_barrier::AgOneBarrier::run")],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        # AITER's custom_all_reduce module flushes denormals; match its adds
        extra_cuda_cflags=["-O3", "-fgpu-flush-denormals-to-zero"],
    )


def all_gather_lastdim_add_1sync(
    ca,
    y: torch.Tensor,
    add_b: torch.Tensor,
    add_c: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out = bf16(bf16(all_gather(y, dim=-1) + add_b) [+ add_c])."""
    assert torch.cuda.is_current_stream_capturing() and ca._IS_CAPTURING
    rows, ld = y.shape
    if out is None:
        out = torch.empty((rows, ld * WORLD_SIZE), dtype=y.dtype, device=y.device)
    if add_c is None:
        add_c = _EMPTY.get(y.device)
        if add_c is None:
            add_c = _EMPTY[y.device] = torch.empty(0, dtype=y.dtype, device=y.device)
    _module().run(ca._ptr, y, out, add_b, add_c)
    _KEEPALIVE.append(y)
    return out
