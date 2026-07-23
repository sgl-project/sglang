"""K3 MNNVL fused all-reduce dispatch (``SGLANG_K3_AR_FUSION``).

Glue between the model and the ``jit_kernel.kimi_k3.all_reduce`` kernels,
mirroring the NCCL-window design (pool + registered segments + find-window
dispatch), but backed by torch symmetric memory:

* :func:`symm_alloc` — allocation context that routes tensor allocations into
  the torch symm-mem pool (with the pause-the-graph-pool dance under CUDA
  graph capture, same as ``pynccl_allocator.SymmetricMemoryContext``).
* :func:`all_reduce` — in-place ``x = allreduce(x) [+ residual]``:
  small messages take the 1shot multicast-push (any tensor; reuses the
  group's CustomAllReduceV2 workspace), large ones take the in-place NVLS
  2shot on ``x``'s multicast address (rendezvous cached per segment).

Call-site contract when ``SGLANG_K3_AR_FUSION`` is on: ``enabled()`` was
checked once (which initializes the state), ``x`` is bf16 and contiguous,
tensors above the push threshold are allocated under :func:`symm_alloc`,
and the residual is identical on every rank (a fully reduced tensor such
as the attn-res prefix sum) or ``None``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

logger = logging.getLogger(__name__)

# push wins below ~0.5 MB on B200x8/GB300 (bs<=32 @ H=7168); NVLS 2shot wins
# above (measured: push 8.5us vs pull 11.0us at 448KB, 15.5 vs 11.0 at 896KB)
_PUSH_MAX_BYTES = 512 * 1024

# Latent width of the K3 latent|shared MoE buffer the fused-norm AR expects
# ([N, NORM_DIM] latent then [N, 2*NORM_DIM] shared). MUST match kNormDim in
# csrc/kimi_k3/comm/ar_fusion.cuh — the mod hardcodes this row width.
NORM_DIM = 3584


class _State:
    def __init__(self, comm: CustomAllReduceV2, world_size: int, group_name: str):
        self.comm = comm  # CustomAllReduceV2 (storage plane owner)
        self.world_size = world_size
        self.group_name = group_name


_STATE: Optional[_State] = None
_INITIALIZED = False


def _init_state() -> Optional[_State]:
    global _STATE, _INITIALIZED
    if _INITIALIZED:
        return _STATE
    _INITIALIZED = True
    if not envs.SGLANG_K3_AR_FUSION.get():
        return None
    from sglang.srt.distributed import get_tensor_model_parallel_world_size
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.parallel_state import get_tp_group

    if get_tensor_model_parallel_world_size() <= 1:
        return None
    group = get_tp_group()
    comm = group.ca_comm
    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or comm.mc_base_ptr == 0
    ):
        logger.warning(
            "SGLANG_K3_AR_FUSION requested but CustomAllReduceV2 with multicast "
            "is unavailable; falling back to the regular all-reduce path."
        )
        return None
    from sglang.jit_kernel.kimi_k3 import all_reduce as mod

    mod.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    _STATE = _State(comm, comm.world_size, group.cpu_group.group_name)
    logger.info("K3 all-reduce fusion enabled (world_size=%d)", comm.world_size)
    return _STATE


def enabled() -> bool:
    return _init_state() is not None


@contextmanager
def symm_alloc():
    """Route tensor allocations into the torch symm-mem pool.

    Only entered when the fusion is enabled (call-site contract). Under CUDA
    graph capture the graph's own pool is paused first (allocating to two
    pools at once is rejected), mirroring
    pynccl_allocator.SymmetricMemoryContext.
    """
    import torch.distributed._symmetric_memory as torch_symm_mem
    from torch._C import (
        _cuda_beginAllocateCurrentThreadToPool,
        _cuda_endAllocateToPool,
        _cuda_releasePool,
    )

    device_index = torch.cuda.current_device()
    pool = torch_symm_mem.get_mem_pool(torch.device("cuda", device_index))
    in_capture = torch.cuda.is_current_stream_capturing()
    graph_pool_id = None
    if in_capture:
        from sglang.srt.distributed.device_communicators import pynccl_allocator

        graph_pool_id = pynccl_allocator._graph_pool_id
        assert graph_pool_id is not None, "graph_pool_id not set under capture"
        _cuda_endAllocateToPool(device_index, graph_pool_id)
    _cuda_beginAllocateCurrentThreadToPool(device_index, pool.id)
    try:
        yield
    finally:
        _cuda_endAllocateToPool(device_index, pool.id)
        _cuda_releasePool(device_index, pool.id)
        if in_capture:
            _cuda_beginAllocateCurrentThreadToPool(device_index, graph_pool_id)


def _find_mc_ptr(state: _State, x: torch.Tensor) -> int:
    """Multicast VA of ``x`` (contract: allocated under :func:`symm_alloc`).

    Rendezvous per call: torch caches the handle per allocation (~2us on a
    hit), tracks segment lifetime itself, and the first touch of a fresh
    segment is a lockstep collective (allocation and dispatch are
    TP-uniform), so no extra Python-side registry is needed or safe.
    """
    import torch.distributed._symmetric_memory as torch_symm_mem

    hdl = torch_symm_mem.rendezvous(x, state.group_name)
    assert hdl is not None and hdl.multicast_ptr != 0, (
        "all_reduce input above the push threshold is not a "
        "symm-pool tensor (allocate it under k3_ar_fusion.symm_alloc())"
    )
    offset = x.data_ptr() - hdl.buffer_ptrs[state.comm.rank]
    return hdl.multicast_ptr + offset


def all_reduce(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]``; returns ``x``.

    Call-site contract (see module docstring): the state is initialized,
    ``x`` is bf16 and contiguous, and large inputs live in the symm pool.
    """
    from sglang.jit_kernel.kimi_k3 import all_reduce as mod

    state = _STATE
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    nbytes = x.numel() * 2
    if nbytes <= min(_PUSH_MAX_BYTES, state.comm.max_push_size):
        return mod.all_reduce_push_res(
            state.world_size, x, residual, ws_mc_base=state.comm.mc_base_ptr
        )
    else:
        return mod.all_reduce_pull_res(
            state.world_size, x, residual, input_mc_ptr=_find_mc_ptr(state, x)
        )


def all_reduce_low_sm(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    num_blocks: Optional[int] = None,
    unroll: Optional[int] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` pinned to the low-SM NVLS
    pull, with the launch geometry exposed; returns ``x``.

    For side-stream use (the shared-expert all-reduce): unlike
    :func:`all_reduce` it never dispatches to the push kernel — push fans
    out over many blocks and would steal SMs from the main-stream GEMMs it
    is meant to overlap. ``x`` must live in the symm pool at ANY size.
    ``num_blocks`` / ``unroll`` default to the tuned tables when None.
    """
    from sglang.jit_kernel.kimi_k3 import all_reduce as mod

    state = _STATE
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    return mod.all_reduce_pull_res(
        state.world_size,
        x,
        residual,
        input_mc_ptr=_find_mc_ptr(state, x),
        num_blocks=num_blocks,
        unroll=unroll,
    )


def all_reduce_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    num_tokens: int,
) -> torch.Tensor:
    """In-place ``x = allreduce(x)`` with a fused RMSNorm over the first
    ``num_tokens`` rows; returns ``x``.

    ``x`` is a 2D ``[rows, NORM_DIM]`` bf16 tensor (a latent-only
    ``[N, NORM_DIM]`` tensor with ``num_tokens = N``, or the flat K3
    latent|shared buffer viewed as ``[3N, NORM_DIM]`` with ``num_tokens =
    N``). Same dispatch and call-site contract as :func:`all_reduce`.
    """
    from sglang.jit_kernel.kimi_k3 import all_reduce as mod

    state = _STATE
    assert state is not None
    if x.shape[0] == 0:
        return x
    nbytes = x.numel() * 2
    if nbytes <= min(_PUSH_MAX_BYTES, state.comm.max_push_size):
        return mod.all_reduce_push_norm(
            state.world_size,
            x,
            weight,
            eps,
            num_norm_rows=num_tokens,
            ws_mc_base=state.comm.mc_base_ptr,
        )
    else:
        return mod.all_reduce_pull_norm(
            state.world_size,
            x,
            weight,
            eps,
            num_norm_rows=num_tokens,
            input_mc_ptr=_find_mc_ptr(state, x),
        )


def finalize_push_fits(num_tokens: int) -> bool:
    """Whether a [num_tokens, NORM_DIM] latent fits the 1shot push window
    (the finalize-fused AR is push-only; larger sizes keep the in-op
    finalize + :func:`all_reduce_norm` NVLS path)."""
    state = _STATE
    assert state is not None
    nbytes = num_tokens * NORM_DIM * 2
    return nbytes <= min(_PUSH_MAX_BYTES, state.comm.max_push_size)


def gemm_ag_up_fits(num_tokens: int) -> bool:
    """Whether the column-parallel up_proj + multicast all-gather tail
    (:func:`gemm_ag_up_proj`) covers this decode batch: TP8, the batch is
    within the kernel's win range, one [num_tokens, 896] staging slice fits
    a push slot, and the phase-counter array covers both launch grids."""
    from sglang.jit_kernel.kimi_k3 import gemm_ag as mod

    state = _STATE
    assert state is not None
    comm = state.comm
    return (
        0 < num_tokens <= mod.MAX_TOKENS
        and state.world_size == 8
        and num_tokens * (2 * NORM_DIM // 8) * 2 <= comm.max_push_size
        and comm.config.num_push_blocks >= 112  # GEMV grid (896 / 8)
    )


def gemm_ag_up_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
) -> torch.Tensor:
    """``up_proj(x) + b (+ c)`` via the column-parallel GEMV + multicast
    all-gather + fused add3 (one more push-workspace user). ``x`` is the
    normed [T, 3584] latent, ``weight`` the FULL replicated [7168, 3584]
    up_proj weight (the kernel slices out this rank's row block itself),
    ``b`` / ``c`` the [T, 7168] addends. Caller checked
    :func:`gemm_ag_up_fits`; same call-site contract as
    :func:`all_reduce`."""
    from sglang.jit_kernel.kimi_k3 import gemm_ag as mod

    state = _STATE
    assert state is not None
    return mod.gemm_ag_up_proj(
        state.world_size,
        x,
        weight,
        b,
        c,
        torch.empty_like(b),
        ws_mc_base=state.comm.mc_base_ptr,
    )


def finalize_all_reduce_push_norm(
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deferred MoE finalize fused into the 1shot push AR + RMSNorm on every
    row; ``out`` ([T, NORM_DIM] bf16) is output-only. Caller checked
    :func:`finalize_push_fits`; same call-site contract as
    :func:`all_reduce`."""
    from sglang.jit_kernel.kimi_k3 import all_reduce as mod

    state = _STATE
    assert state is not None
    return mod.finalize_all_reduce_push_norm(
        state.world_size,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
        ws_mc_base=state.comm.mc_base_ptr,
    )
