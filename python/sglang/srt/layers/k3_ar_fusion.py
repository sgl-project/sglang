"""K3 MNNVL fused all-reduce dispatch.

Auto-enabled on SM100/SM103 when CustomAllReduceV2 with multicast is
available; ``SGLANG_K3_AR_FUSION`` overrides in either direction (0 = off,
1 = attempt anywhere and warn when unavailable).

Glue between the model and the ``kernels.ops.kimi_k3.all_reduce`` kernels. Small
messages take the 1shot multicast-push through the group's CustomAllReduceV2
workspace; large ones take the in-place NVLS 2shot, which needs its input to be a
:func:`symm_buffer` slice and otherwise falls back to the regular all-reduce.

Call-site contract when the fusion is active: ``enabled()`` was
checked once (which initializes the state), ``x`` is bf16 and contiguous,
and the residual is identical on every rank (a fully reduced tensor such
as the attn-res prefix sum) or ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

import sglang.srt.runtime_context as ctx
from sglang.kernels.jit.utils import cache_once
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

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

# :func:`symm_buffer` names: o_proj's [rows, hidden_size] output, and the MoE
# [latent | shared] pair flattened to rows x (moe_hidden_size + hidden_size).
ATTN_O_PROJ = "attn_o_proj"
MOE_LATENT_SHARED = "moe_latent_shared"


class _State(NamedTuple):
    comm: CustomAllReduceV2
    world_size: int
    group_name: str


@cache_once
def _get_state() -> Optional[_State]:
    explicit = envs.SGLANG_K3_AR_FUSION.is_set()
    if explicit:
        if not envs.SGLANG_K3_AR_FUSION.get():
            return None
    else:
        # Auto-probe: only SM100/SM103, and not under --enable-symm-mem or EP
        # a2a (their allocator contexts misroute the buffers the pull needs).
        from sglang.srt.utils.common import get_device_sm

        if get_device_sm() not in (100, 103):
            return None
        symm_mem = ctx.get_exec().comm.enable_symm_mem
        a2a_backend = ctx.get_exec().moe.moe_a2a_backend
        if symm_mem or a2a_backend != "none":
            logger.info(
                "K3 all-reduce fusion auto-probe: skipping "
                "(enable_symm_mem=%s, moe_a2a_backend=%s; under symm-mem the "
                "allocator contexts conflict, and under EP a2a the model's "
                "symm-pool allocation contract does not hold on every AR "
                "call-site. Set SGLANG_K3_AR_FUSION=1 to force.)",
                symm_mem,
                a2a_backend,
            )
            return None
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.parallel_state import get_tp_group

    if get_parallel().tp_size <= 1:
        return None
    group = get_tp_group()
    comm = group.ca_comm
    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not comm.has_multicast
    ):
        if explicit:
            logger.warning(
                "SGLANG_K3_AR_FUSION requested but CustomAllReduceV2 with multicast "
                "is unavailable; falling back to the regular all-reduce path."
            )
        else:
            logger.info(
                "K3 all-reduce fusion auto-probe: CustomAllReduceV2 with "
                "multicast is unavailable; using the regular all-reduce path."
            )
        return None
    from sglang.kernels.ops.kimi_k3 import all_reduce as mod

    mod.register_comm(comm.obj)
    logger.info("K3 all-reduce fusion enabled (world_size=%d)", comm.world_size)
    return _State(comm, comm.world_size, group.cpu_group.group_name)


def enabled() -> bool:
    return _get_state() is not None


class _Buffer(NamedTuple):
    region: torch.Tensor  # flat [max_rows * width]
    base: int  # local VA of region[0]
    mc_base: int  # multicast VA of region[0]
    max_rows: int
    width: int


# Reverse pointer -> multicast lookup for get_mc_ptr().
_BUFS: List[_Buffer] = []


@cache_once
def _max_buffer_rows() -> int:
    """Rows to reserve per buffer: the largest batch the server args allow."""
    schedule = ctx.get_schedule()
    chunked = schedule.chunked_prefill_size
    if chunked is not None and chunked > 0:
        return int(chunked)
    return int(schedule.max_prefill_tokens or 0)


def _create_buffer(
    name: str, width: int, dtype: torch.dtype, group_name: str
) -> _Buffer:
    """One symmetric buffer, rendezvoused once. ``empty_strided_p2p`` skips the
    caching allocator, so this is right for a persistent buffer only."""
    from torch._C._distributed_c10d import _SymmetricMemory

    max_rows = _max_buffer_rows()
    # outside inference mode on purpose: created inside it the buffer would be an
    # inference tensor, and the in-place write from a no_grad path (CUDA graph
    # capture) is then rejected
    with torch.inference_mode(False):
        region = _SymmetricMemory.empty_strided_p2p(
            (max_rows * width,),
            [1],
            dtype,
            torch.device("cuda", torch.cuda.current_device()),
            group_name,
        ).view(max_rows, width)
    handle = _SymmetricMemory.rendezvous(region)
    assert handle is not None and handle.multicast_ptr != 0
    buf = _Buffer(
        region=region,
        base=int(region.data_ptr()),
        mc_base=int(handle.multicast_ptr),
        max_rows=max_rows,
        width=width,
    )
    _BUFS.append(buf)
    logger.info(
        "K3 symm buffer %r: %d rows x %d, %.1f MB (rendezvoused once on first "
        "use; no per-forward symmetric allocation)",
        name,
        max_rows,
        width,
        region.numel() * region.element_size() / (1024 * 1024),
    )
    return buf


def symm_buffer(
    name: str,
    rows: int,
    width: int,
    dtype: torch.dtype,
    group_name: Optional[str] = None,
):
    """``[rows, width]`` view of a named persistent symmetric buffer.

    The pull reduces in place on the input's multicast alias, so every rank must
    resolve the same ``(buffer, offset)``; a per-forward symm-pool allocation does
    not, because torch's caching allocator breaks ties on the absolute address and
    ``rendezvous`` validates sizes only.

    ``group_name`` defaults to the TP group, the one the fused all-reduce reduces
    over; a caller reducing over another group (SP-MoE's attention TP group) passes
    its own. Each name belongs to one group, so the name alone identifies it.
    """
    if group_name is None:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group_name = get_tp_group().cpu_group.group_name
    buf: _Buffer = ctx.get_buffer(
        f"k3_symm:{name}", lambda: _create_buffer(name, width, dtype, group_name)
    )
    assert 0 < rows <= buf.max_rows and width == buf.width and dtype == buf.region.dtype
    return buf.region[:rows]


def find_mc_ptr(x: torch.Tensor) -> Optional[int]:
    """Multicast VA of ``x`` if it lies inside a named buffer, else None."""
    ptr = int(x.data_ptr())
    end = ptr + x.numel() * x.element_size()
    for buf in _BUFS:
        if buf.base <= ptr and end <= buf.base + buf.region.nbytes:
            return buf.mc_base + (ptr - buf.base)
    return None


def get_mc_ptr(x: torch.Tensor) -> int:
    """Multicast VA of ``x``, which must be a :func:`symm_buffer` slice."""
    mc = find_mc_ptr(x)
    assert mc is not None, "K3 fused pull input is not a symm_buffer slice"
    return mc


def all_reduce(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]``; returns ``x``."""
    from sglang.kernels.ops.kimi_k3 import all_reduce as mod

    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    nbytes = x.numel() * 2
    if nbytes <= min(_PUSH_MAX_BYTES, state.comm.max_push_size):
        return mod.all_reduce_push_res(state.world_size, x, residual)
    return mod.all_reduce_pull_res(
        state.world_size, x, residual, input_mc_ptr=get_mc_ptr(x)
    )


def all_reduce_low_sm(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    num_blocks: Optional[int] = None,
    unroll: Optional[int] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]``, pinned to the low-SM NVLS pull.

    For side-stream use: push would fan out over many blocks and steal SMs from
    the GEMMs it overlaps. ``num_blocks`` / ``unroll`` default to the tuned tables.
    """
    from sglang.kernels.ops.kimi_k3 import all_reduce as mod

    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    return mod.all_reduce_pull_res(
        state.world_size,
        x,
        residual,
        input_mc_ptr=get_mc_ptr(x),
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
    ``num_tokens`` rows of the 2D ``[rows, NORM_DIM]`` input."""
    from sglang.kernels.ops.kimi_k3 import all_reduce as mod

    state = _get_state()
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
        )
    return mod.all_reduce_pull_norm(
        state.world_size,
        x,
        weight,
        eps,
        num_norm_rows=num_tokens,
        input_mc_ptr=get_mc_ptr(x),
    )


def finalize_push_fits(num_tokens: int) -> bool:
    """Whether a [num_tokens, NORM_DIM] latent fits the push window; the
    finalize-fused AR is push-only."""
    state = _get_state()
    assert state is not None
    nbytes = num_tokens * NORM_DIM * 2
    return nbytes <= min(_PUSH_MAX_BYTES, state.comm.max_push_size)


def gemm_ag_up_fits(num_tokens: int) -> bool:
    """Whether :func:`gemm_ag_up_proj` covers this decode batch: TP8, within the
    kernel's win range, and the staging slice fits a push slot."""
    from sglang.kernels.ops.kimi_k3 import gemm_ag as mod

    state = _get_state()
    assert state is not None
    comm = state.comm
    return (
        0 < num_tokens <= mod.MAX_TOKENS
        and state.world_size == 8
        and num_tokens * (2 * NORM_DIM // 8) * 2 <= comm.max_push_size
        # The GEMV producer reads a single phase counter, so its grid is not
        # bound to the counter array; the spin consumer still needs one block
        # per counter slot plus the cleanup block.
        and comm.config.num_push_blocks >= 2
    )


def gemm_ag_up_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
) -> torch.Tensor:
    """``up_proj(x) + b (+ c)`` via column-parallel GEMV + multicast all-gather +
    fused add3. ``weight`` is the FULL replicated up_proj (the kernel slices this
    rank's rows). Caller checked :func:`gemm_ag_up_fits`."""
    from sglang.kernels.ops.kimi_k3 import gemm_ag as mod

    state = _get_state()
    assert state is not None
    return mod.gemm_ag_up_proj(
        state.world_size,
        x,
        weight,
        b,
        c,
        torch.empty_like(b),
    )


def finalize_all_reduce_push_norm(
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deferred MoE finalize fused into the 1shot push AR + RMSNorm; ``out`` is
    output-only. Caller checked :func:`finalize_push_fits`."""
    from sglang.kernels.ops.kimi_k3 import all_reduce as mod

    state = _get_state()
    assert state is not None
    return mod.finalize_all_reduce_push_norm(
        state.world_size,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
    )
