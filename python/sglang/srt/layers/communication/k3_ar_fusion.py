"""K3 MNNVL fused all-reduce dispatch.

Auto-enabled on SM100/SM103 when the TP symmetric workspace has multicast; ``SGLANG_K3_AR_FUSION`` overrides in either direction (0 = off,
1 = attempt anywhere and warn when unavailable).

Glue between the model and the ``nvlink_comm`` all-reduce and the
``fused_all_reduce_{push,pull}`` norm kernels. Small messages take the 1-shot
push through the group's symmetric workspace; large ones take the in-place NVLS
2-shot, which needs its input to be a :func:`symm_buffer` slice and otherwise
falls back to the regular all-reduce.

Call-site contract when the fusion is active: ``enabled()`` was
checked once (which initializes the state), ``x`` is bf16 and contiguous,
and the residual is identical on every rank (a fully reduced tensor such
as the attn-res prefix sum) or ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, overload

import torch

import sglang.srt.runtime_context as ctx
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import fused_all_reduce, kimi_k3_comm, nvlink_comm
from sglang.srt.environ import envs
from sglang.srt.layers.communication import ar_fusion
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceLease,
    )

logger = logging.getLogger(__name__)

# push wins below ~0.5 MB on B200x8/GB300 (bs<=32 @ H=7168); NVLS 2shot wins
# above (measured: push 8.5us vs pull 11.0us at 448KB, 15.5 vs 11.0 at 896KB)
_PUSH_MAX_BYTES = 512 * 1024

# Latent width of the K3 latent|shared MoE buffer the fused-norm AR expects
# ([N, NORM_DIM] latent then [N, 2*NORM_DIM] shared).
NORM_DIM = 3584

# Fused-norm pull: 2 rows per group; CTAs by message size. Measured on GB300 x8.
_PULL_ROWS_PER_GROUP = 2
_PULL_SMALL_MAX_BYTES = 672 * 1024
_PULL_SMALL_NUM_BLOCKS = 8
_PULL_LARGE_NUM_BLOCKS = 24
# Side-stream pull: few CTAs so the routed GEMMs it overlaps keep their SMs.
# Measured on GB300 x8 at K3 widths (alone, not under overlap).
_LOW_SM_PULL_BLOCKS = 4
_LOW_SM_PULL_UNROLL = 8

# :func:`symm_buffer` names: o_proj's [rows, hidden_size] output, and the MoE
# [latent | shared] pair flattened to rows x (moe_hidden_size + hidden_size).
ATTN_O_PROJ = "attn_o_proj"
MOE_LATENT_SHARED = "moe_latent_shared"


class _State(NamedTuple):
    lease: WorkspaceLease
    group_name: str

    @property
    def push_slot_bytes(self) -> int:
        return self.lease.request.push_slot_bytes

    @property
    def rank(self) -> int:
        return self.lease.rank

    @property
    def world_size(self) -> int:
        return self.lease.world_size


@cache_once
def _get_state() -> Optional[_State]:
    explicit = (
        envs.SGLANG_K3_AR_FUSION.is_set()
        or envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get()
    )
    if explicit:
        if not envs.SGLANG_K3_AR_FUSION.get():
            return None
    else:
        # Auto-probe: only SM100/SM103, and not under --enable-symm-mem or EP
        # a2a (their allocator contexts misroute the buffers the pull needs).
        from sglang.srt.utils.device import get_device_sm

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
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        V2_PUSH_SLOT,
        WorkspaceRequest,
        request_group_workspace,
    )
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size <= 1:
        return None
    group = get_parallel().tp_group
    # pull barriers only; any request gets at least one semaphore per SM
    lease = request_group_workspace(
        group,
        WorkspaceRequest(
            owner="k3_ar_fusion", push_slot_bytes=V2_PUSH_SLOT, semaphores=1
        ),
    )
    if lease is None or not lease.has_multicast:
        if explicit:
            logger.warning(
                "SGLANG_K3_AR_FUSION requested but a multicast symmetric workspace "
                "is unavailable; falling back to the regular all-reduce path."
            )
        else:
            logger.info(
                "K3 all-reduce fusion auto-probe: a multicast symmetric workspace "
                "is unavailable; using the regular all-reduce path."
            )
        return None
    kimi_k3_comm.register_comm(lease.get_communicator, world_size=lease.world_size)
    logger.info("K3 all-reduce fusion enabled (world_size=%d)", lease.world_size)
    return _State(lease, group.cpu_group.group_name)


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
        "K3 symm buffer %r: %d rows x %d, %.1f MB (rendezvoused once; no "
        "per-forward symmetric allocation)",
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
        from sglang.srt.runtime_context import get_parallel

        group_name = get_parallel().tp_group.cpu_group.group_name
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


@cache_once
def register_sp_collective_comm() -> bool:
    """Register the TP workspace's communicator with ``sp_collective``, whose
    direct ops barrier on it; False when another communicator owns this TP size."""
    from sglang.kernels.ops.communication import sp_collective

    state = _get_state()
    assert state is not None
    try:
        sp_collective.register_comm(
            state.lease.get_communicator, world_size=state.world_size
        )
    except AssertionError:
        return False
    return True


def extend_rows(x: torch.Tensor, rows: int) -> Optional[torch.Tensor]:
    """The first ``rows`` rows of the named buffer ``x`` starts, or None when
    ``x`` does not start one or it reserves fewer rows."""
    ptr = int(x.data_ptr())
    for buf in _BUFS:
        if buf.base == ptr and x.shape[1] == buf.width and rows <= buf.max_rows:
            return buf.region[:rows]
    return None


def try_symm_buffer(
    name: str, rows: int, width: int, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """TP-group :func:`symm_buffer`, or None when ``rows`` exceeds its reserved rows."""
    if rows > _max_buffer_rows():
        return None
    return symm_buffer(name, rows, width, dtype)


def all_reduce(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]``; returns ``x``."""
    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    nbytes = x.numel() * 2
    if nbytes <= min(_PUSH_MAX_BYTES, state.push_slot_bytes):
        _k3_push_all_reduce(state.world_size, x, residual)
    else:
        _k3_pull_all_reduce(
            state.world_size,
            x,
            get_mc_ptr(x),
            residual=residual,
            num_blocks_hint=0,
            num_unroll=4,
        )
    return x


def all_reduce_low_sm(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]``, pinned to the low-SM NVLS pull.

    For side-stream use: push would fan out over many blocks and steal SMs from
    the GEMMs it overlaps.
    """
    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    _k3_pull_all_reduce(
        state.world_size,
        x,
        get_mc_ptr(x),
        residual=residual,
        num_blocks_hint=_LOW_SM_PULL_BLOCKS,
        num_unroll=_LOW_SM_PULL_UNROLL,
    )
    return x


@overload
def _rows(t: Literal[None]) -> Literal[None]: ...


@overload
def _rows(t: torch.Tensor) -> torch.Tensor: ...


def _rows(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.view(-1, NORM_DIM)


@register_custom_op(op_name="k3_push_all_reduce", mutates_args=["x"])
def _k3_push_all_reduce(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> None:
    rows = _rows(x)
    nvlink_comm.all_reduce_push(
        kimi_k3_comm.get_comm(world_size), rows, rows, _rows(residual)
    )


@register_custom_op(op_name="k3_pull_all_reduce", mutates_args=["x"])
def _k3_pull_all_reduce(
    world_size: int,
    x: torch.Tensor,
    x_mc: int,
    residual: Optional[torch.Tensor],
    num_blocks_hint: int,
    num_unroll: int,
) -> None:
    rows = _rows(x)
    nvlink_comm.all_reduce_pull(
        kimi_k3_comm.get_comm(world_size),
        rows,
        rows,
        _rows(residual),
        in_mc_ptr=x_mc,
        out_mc_ptr=x_mc,
        num_unroll=num_unroll,
        num_blocks_hint=num_blocks_hint,
    )


def all_reduce_norm_prefix(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    num_tokens: int,
) -> torch.Tensor:
    """In-place ``x = allreduce(x)`` with a fused RMSNorm over the first
    ``num_tokens`` rows of the 2D ``[rows, NORM_DIM]`` input."""
    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x
    nbytes = x.numel() * 2
    if nbytes <= min(_PUSH_MAX_BYTES, state.push_slot_bytes):
        _k3_push_all_reduce_norm_prefix(
            world_size=state.world_size,
            x=x,
            norm_weight=norm_weight,
            eps=eps,
            num_norm_rows=num_tokens,
        )
        return x
    num_blocks = (
        _PULL_SMALL_NUM_BLOCKS
        if nbytes <= _PULL_SMALL_MAX_BYTES
        else _PULL_LARGE_NUM_BLOCKS
    )
    _k3_pull_all_reduce_norm_prefix(
        world_size=state.world_size,
        x=x,
        x_mc=get_mc_ptr(x),
        norm_weight=norm_weight,
        eps=eps,
        num_norm_rows=num_tokens,
        num_blocks=num_blocks,
    )
    return x


@register_custom_op(op_name="k3_push_all_reduce_norm_prefix", mutates_args=["x"])
def _k3_push_all_reduce_norm_prefix(
    world_size: int,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
) -> None:
    fused_all_reduce.push.all_reduce_norm_prefix(
        kimi_k3_comm.get_comm(world_size),
        x,
        norm_weight,
        eps,
        num_norm_rows=num_norm_rows,
        cluster_size=ar_fusion.get_cluster_size(norm_weight.shape[0]),
    )


@register_custom_op(op_name="k3_pull_all_reduce_norm_prefix", mutates_args=["x"])
def _k3_pull_all_reduce_norm_prefix(
    world_size: int,
    x: torch.Tensor,
    x_mc: int,
    norm_weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
    num_blocks: int,
) -> None:
    fused_all_reduce.pull.all_reduce_norm_prefix(
        kimi_k3_comm.get_comm(world_size),
        x,
        x_mc,
        norm_weight,
        eps,
        num_norm_rows=num_norm_rows,
        num_blocks=num_blocks,
        rows_per_group=_PULL_ROWS_PER_GROUP,
    )


def moe_finalize_all_reduce_norm_fits(num_tokens: int) -> bool:
    """Whether a [num_tokens, NORM_DIM] latent fits the push window; the
    finalize-fused AR is push-only."""
    state = _get_state()
    assert state is not None
    nbytes = num_tokens * NORM_DIM * 2
    return nbytes <= min(_PUSH_MAX_BYTES, state.push_slot_bytes)


# Largest batch where the sharded up_proj tail beats the replicated GEMM + add3,
# measured on GB300 TP8 (the only TP size KimiK3MoE runs the tail at).
_SHARDED_UP_MAX_TOKENS = 256


def gemm_ag_up_fits(num_tokens: int) -> bool:
    """Whether :func:`gemm_ag_up_proj` covers this decode batch: within its
    row window (the fused-GEMV range only under SGLANG_K3_SHARDED_MOE=0), and
    the [world, M, 7168 / world] staging fits a push slot."""
    from sglang.kernels.ops.communication import gemm_ag as mod

    state = _get_state()
    assert state is not None
    max_tokens = (
        _SHARDED_UP_MAX_TOKENS
        if envs.SGLANG_K3_SHARDED_MOE.get()
        else mod.GEMV_MAX_TOKENS
    )
    return (
        0 < num_tokens <= max_tokens
        and num_tokens * (2 * NORM_DIM // state.world_size) * 2 <= state.push_slot_bytes
    )


# Warp split-K's row limit for a slice GEMM; TGV above. Both beat cuBLAS at
# every decode M, measured on GB300 with CUDA graphs.
_SLICE_WARP_SPLITK_MAX_TOKENS = 32


def slice_gemm(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``x @ weight.T`` for this rank's row block of a TP-sharded weight: the
    sharded MoE front's [block | gate_up] (fp32 out) and the up_proj slice."""
    from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    m, n = x.shape[0], weight.shape[0]
    if not get_bf16_gemm_backend().is_cutedsl():
        if out_dtype == x.dtype:
            return torch.nn.functional.linear(x, weight)
        return torch.mm(x, weight.t(), out_dtype=out_dtype)
    out = x.new_empty((m, n), dtype=out_dtype)
    # warp split-K compiles per M: only captured batches (a fixed set of M) take it,
    # and its 16- or 32-column output tiles must divide N (not the TP16 front's 1048)
    if m <= _SLICE_WARP_SPLITK_MAX_TOKENS and n % 16 == 0 and get_is_capture_mode():
        from sglang.kernels.ops.gemm.dense_bf16_gemm_warp_splitk import (
            default_tactic,
            run_warp_splitk_dense,
        )

        tactic = default_tactic(m, n, weight.shape[1])
        return run_warp_splitk_dense(x, weight.t(), out, pdl=True, tactic=tactic)
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm_out

    return cutedsl_bf16_gemm_out(x, weight, out)


def gemm_ag_up_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
) -> torch.Tensor:
    """``up_proj(x) + b (+ c)`` with the up_proj sharded by output column over
    TP. ``weight`` is the FULL replicated up_proj. Caller checked
    :func:`gemm_ag_up_fits`."""
    from sglang.kernels.ops.communication import gemm_ag as mod

    state = _get_state()
    assert state is not None
    if x.shape[0] > mod.GEMV_MAX_TOKENS:
        rows = weight.shape[0] // state.world_size
        rank = state.rank
        local = slice_gemm(x, weight[rank * rows : (rank + 1) * rows])
        return mod.gather_add3(state.world_size, local, b, c, torch.empty_like(b))
    return mod.gemm_ag_up_proj(
        state.world_size,
        x,
        weight,
        b,
        c,
        torch.empty_like(b),
    )


def moe_front_all_gather_fits(num_tokens: int, n_local: int) -> bool:
    """Whether :func:`moe_front_all_gather` covers a ``[num_tokens, n_local]``
    fp32 block: it fits a push slot."""
    state = _get_state()
    assert state is not None
    return (
        num_tokens > 0
        and n_local % 4 == 0
        and num_tokens * n_local * 4 <= state.push_slot_bytes
    )


def moe_front_all_gather(local: torch.Tensor) -> torch.Tensor:
    """``[M, world * n_local]`` fp32 column all-gather of this rank's
    ``[M, n_local]`` block, whose rows may be strided (rank-major columns).
    Caller checked :func:`moe_front_all_gather_fits`."""
    from sglang.kernels.ops.communication import moe_front_ag as mod

    state = _get_state()
    assert state is not None
    out = local.new_empty((local.shape[0], local.shape[1] * state.world_size))
    return mod.moe_front_all_gather(state.world_size, local, out)


def moe_finalize_all_reduce_norm(
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deferred MoE finalize fused into the 1-shot push AR + RMSNorm; ``out`` is
    output-only. Caller checked :func:`moe_finalize_all_reduce_norm_fits`."""
    state = _get_state()
    assert state is not None
    _k3_push_moe_finalize_all_reduce_norm(
        world_size=state.world_size,
        out=out,
        gemm2_out=gemm2_out,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        expert_weights=expert_weights,
        norm_weight=norm_weight,
        eps=eps,
    )
    return out


@register_custom_op(
    op_name="k3_push_moe_finalize_all_reduce_norm", mutates_args=["out"]
)
def _k3_push_moe_finalize_all_reduce_norm(
    world_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> None:
    fused_all_reduce.push.moe_finalize_all_reduce_norm(
        kimi_k3_comm.get_comm(world_size),
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        norm_weight,
        eps,
        cluster_size=ar_fusion.get_cluster_size(norm_weight.shape[0]),
    )
