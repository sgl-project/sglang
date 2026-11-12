# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K3 MNNVL fused all-reduce dispatch.

Auto-enabled on SM100/SM103 (SM107 with ``SGLANG_ENABLE_K3_SM107``) when
CustomAllReduceV2 with multicast is available; ``SGLANG_K3_AR_FUSION``
overrides in either direction (0 = off, 1 = attempt anywhere and warn when
unavailable).

Glue between the model and the
``kernels.ops.communication.all_reduce_residual`` kernels. Small messages take
the 1shot multicast-push through the group's CustomAllReduceV2
workspace; mid-size ones (``SGLANG_K3_AR_PUSH2``, default off) the barrier-free
2shot push through the same workspace; large ones take the in-place NVLS 2shot,
which needs its input to be a :func:`symm_buffer` slice and otherwise falls back
to the regular all-reduce.

Call-site contract when the fusion is active: ``enabled()`` was
checked once (which initializes the state), ``x`` is bf16 and contiguous,
and the residual is identical on every rank (a fully reduced tensor such
as the attn-res prefix sum) or ``None``.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

import sglang.srt.runtime_context as ctx
from sglang.kernels.jit.utils import cache_once
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

logger = logging.getLogger(__name__)

# The SM10x parts the K3 collectives were measured on.
_K3_MEASURED_SMS = (100, 103)
_SM107_LOGGED = False


def k3_sm_supported() -> bool:
    """SM100/SM103, or SM107 (VR200) under SGLANG_ENABLE_K3_SM107: the
    multimem / cluster feature set the K3 paths gated here need. Uncached so
    tests can patch get_device_sm."""
    global _SM107_LOGGED
    from sglang.srt.utils.device import get_device_sm

    sm = get_device_sm()
    if sm in _K3_MEASURED_SMS:
        return True
    if sm != 107 or not envs.SGLANG_ENABLE_K3_SM107.get():
        return False
    if not _SM107_LOGGED:
        _SM107_LOGGED = True
        logger.info(
            "K3 SM107 port: SM100/SM103-gated K3 paths enabled on sm=107 "
            "(SGLANG_ENABLE_K3_SM107=1)"
        )
    return True


# push wins below ~0.5 MB on B200x8/GB300 (bs<=32 @ H=7168); NVLS 2shot wins
# above (measured at 448 KB and 896 KB)
_PUSH_MAX_BYTES = 512 * 1024

# With the 2shot push band on (SGLANG_K3_AR_PUSH2), the 1shot push keeps only
# the sizes where it still beats it (GB300 TP8 over MNNVL, CUDA graph): the
# residual AR 1shot is faster at 168 KB and slower from 252 KB, the latent-norm
# AR 1shot faster at 252 KB and slower at 336 KB. The finalize-fused latent AR
# keeps the whole window (the fused 1shot is faster than the fused 2shot at 72 rows).
_PUSH_RES_MAX_BYTES = 240 * 1024
_PUSH_NORM_MAX_BYTES = 288 * 1024
# 2shot fan-out: unicast for the flat residual AR (faster than multicast at
# 240 rows), one multicast store for the row-owner norm kernels (even at 240
# rows, faster at 36).
_PUSH2_RES_MC = False
_PUSH2_NORM_MC = True

# SGLANG_K3_AR_LARGE_PULL_TUNING: (num_blocks, unroll) for pull messages above
# _LARGE_PULL_BYTES, from a TP4 NVLS sweep; the stock fallbacks leave much of the
# NVLS bandwidth unused there. The shared-expert AR overlaps the routed GEMMs, so it gets a
# smaller grid than the critical-path ones.
_LARGE_PULL_BYTES = 4 * 1024 * 1024
_LARGE_PULL_RES = (32, 8)
_LARGE_PULL_NORM = (128, 2)
_LARGE_PULL_LOW_SM_BLOCKS = 16


def _large_pull(nbytes: int) -> bool:
    return nbytes > _LARGE_PULL_BYTES and envs.SGLANG_K3_AR_LARGE_PULL_TUNING.get()


def _sm107_window_env(name: str) -> Optional[int]:
    """The positive int in env ``name`` on SM107, else None (unset or off SM107)."""
    raw = os.environ.get(name, "")
    if not raw:
        return None
    from sglang.srt.utils.device import get_device_sm

    if get_device_sm() != 107:
        return None
    window = int(raw)
    if window <= 0:
        raise ValueError(f"{name}={raw} must be > 0")
    return window


@lru_cache(maxsize=1)
def _push_max_bytes() -> int:
    """The 1shot-push window of :func:`all_reduce` (push_res vs pull_res):
    _PUSH_MAX_BYTES, or on SM107 SGLANG_K31_AR_PUSH_MAX_BYTES (default unset;
    a measured push vs NVLS pull crossover). The push plane's slot
    size (``comm.max_push_size``) still caps it at every call."""
    window = _sm107_window_env("SGLANG_K31_AR_PUSH_MAX_BYTES")
    if window is None:
        return _PUSH_MAX_BYTES
    logger.info(
        "K3 AR push window on SM107: %d bytes (SGLANG_K31_AR_PUSH_MAX_BYTES; "
        "default %d), capped by the push slot",
        window,
        _PUSH_MAX_BYTES,
    )
    return window


@lru_cache(maxsize=1)
def _norm_push_max_bytes() -> int:
    """The 1shot-push window of :func:`all_reduce_norm` (push_norm vs
    pull_norm, the latent|shared MoE buffer), measured separately from the
    residual AR's: on SM107 SGLANG_K31_AR_NORM_PUSH_MAX_BYTES, else the
    :func:`_push_max_bytes` window. Capped by the push slot at every call."""
    window = _sm107_window_env("SGLANG_K31_AR_NORM_PUSH_MAX_BYTES")
    if window is None:
        return _push_max_bytes()
    logger.info(
        "K3 AR norm push window on SM107: %d bytes "
        "(SGLANG_K31_AR_NORM_PUSH_MAX_BYTES; default the all_reduce window %d), "
        "capped by the push slot",
        window,
        _push_max_bytes(),
    )
    return window


# --- o_proj AR riders (SM107, each default-off) ---
# A2: the o_proj residual AR at >= SGLANG_K31_AR_TWOSHOT_MIN_BYTES takes
#     CustomAllReduceV2's 2shot push plus a bf16 residual add
#     (_o_proj_twoshot_add).
# A3: the o_proj AR without a residual gets its own push window,
#     SGLANG_K31_AR_NORES_PUSH_MAX_BYTES; above it SGLANG_K31_AR_NORES_ABOVE
#     (pull | twoshot, default pull) picks the NVLS pull or A2's 2shot path.
# all_reduce_norm is never rerouted: the finalize-fused AR is push-only.


@lru_cache(maxsize=1)
def _twoshot_min_bytes() -> Optional[int]:
    """A2: the smallest o_proj residual AR (bytes) that takes the 2shot push +
    bf16 add, on SM107 only; None = off (unset, or not SM107)."""
    window = _sm107_window_env("SGLANG_K31_AR_TWOSHOT_MIN_BYTES")
    if window is not None:
        logger.info(
            "K3 AR o_proj residual 2shot window on SM107: >= %d bytes take "
            "CustomAllReduceV2 2shot_push + bf16 residual add "
            "(SGLANG_K31_AR_TWOSHOT_MIN_BYTES)",
            window,
        )
    return window


@lru_cache(maxsize=1)
def _nores_push_max_bytes() -> int:
    """A3: the 1shot-push window of the o_proj AR without a residual: on
    SM107 SGLANG_K31_AR_NORES_PUSH_MAX_BYTES, else the :func:`all_reduce`
    window. Capped by the push slot at every call."""
    window = _sm107_window_env("SGLANG_K31_AR_NORES_PUSH_MAX_BYTES")
    if window is None:
        return _push_max_bytes()
    logger.info(
        "K3 AR o_proj no-residual push window on SM107: %d bytes, above it %s "
        "(SGLANG_K31_AR_NORES_PUSH_MAX_BYTES / SGLANG_K31_AR_NORES_ABOVE)",
        window,
        _nores_above(),
    )
    return window


@lru_cache(maxsize=1)
def _nores_above() -> str:
    """A3: where an o_proj AR without a residual goes above its push window:
    "pull" (default, as before) or "twoshot" (SM107 only)."""
    raw = os.environ.get("SGLANG_K31_AR_NORES_ABOVE", "").strip() or "pull"
    if raw not in ("pull", "twoshot"):
        raise ValueError(f"SGLANG_K31_AR_NORES_ABOVE={raw} must be pull or twoshot")
    if raw == "twoshot":
        from sglang.srt.utils.device import get_device_sm

        if get_device_sm() != 107:
            return "pull"
    return raw


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
    explicit = (
        envs.SGLANG_K3_AR_FUSION.is_set()
        or envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get()
    )
    if explicit:
        if not envs.SGLANG_K3_AR_FUSION.get():
            return None
    else:
        # Auto-probe: only SM100/SM103 (+SM107, k3_sm_supported), and not under
        # --enable-symm-mem or EP a2a (their allocator contexts misroute the
        # buffers the pull needs).
        if not k3_sm_supported():
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
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size <= 1:
        return None
    group = get_parallel().tp_group
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
    from sglang.kernels.ops.communication import all_reduce_residual as mod

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


def symm_buffer_max_rows() -> int:
    """Rows every :func:`symm_buffer` holds (the largest prefill batch)."""
    return _max_buffer_rows()


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


def _push_fits(state: _State, nbytes: int, window: int = _PUSH_MAX_BYTES) -> bool:
    """The 1shot push window."""
    return nbytes <= min(window, state.comm.max_push_size)


def _pick_push(
    state: _State, nbytes: int, push_max: int, *, rows: bool, window: int
) -> str:
    """``"push"`` (1shot), ``"push2"`` or ``"pull"`` for an ``nbytes`` AR under
    SGLANG_K3_AR_PUSH2: the 1shot push up to ``push_max``, the 2shot push band
    past it, and the 1shot push again for whatever the band does not take
    inside the 1shot ``window``."""
    push = _push_fits(state, nbytes, window)
    if push and nbytes <= push_max:
        return "push"
    if _push2_fits(state, nbytes, rows=rows):
        return "push2"
    return "push" if push else "pull"


def _push2_fits(state: _State, nbytes: int, *, rows: bool = False) -> bool:
    """The 2shot push band (past the 1shot window): ``nbytes`` within
    ``SGLANG_K3_AR_PUSH2_MAX_KB`` and its two owner-chunk halves (whole latent
    rows for the norm variants) within a push slot."""
    from sglang.kernels.ops.communication import all_reduce_residual as mod

    if not envs.SGLANG_K3_AR_PUSH2.get():
        return False
    slot_bytes = mod.push2_slot_bytes(
        nbytes, state.world_size, row_bytes=NORM_DIM * 2 if rows else 0
    )
    return (
        nbytes <= envs.SGLANG_K3_AR_PUSH2_MAX_KB.get() * 1024
        and slot_bytes <= state.comm.max_push_size
    )


def _o_proj_twoshot(
    state: _State, x: torch.Tensor, nbytes: int, has_residual: bool
) -> bool:
    """Whether this o_proj AR takes the 2shot push (A2 with a residual, A3's
    "twoshot" without). Depends only on bytes, env, SM and the communicator's
    config, so every rank decides alike."""
    if has_residual:
        min_bytes = _twoshot_min_bytes()
        if min_bytes is None or nbytes < min_bytes:
            return False
    elif _nores_above() != "twoshot" or nbytes <= min(
        _nores_push_max_bytes(), state.comm.max_push_size
    ):
        return False
    comm = state.comm
    covered = comm.should_custom_ar(x) and comm.uses_two_shot_push(nbytes)
    _log_twoshot_once(has_residual, covered, x.shape[0], nbytes)
    return covered


def _o_proj_twoshot_add(
    state: _State, x: torch.Tensor, residual: Optional[torch.Tensor]
) -> torch.Tensor:
    """The o_proj AR on the 2shot push: CustomAllReduceV2 reduces ``x`` into a
    new tensor, which is written back into ``x`` (plus ``residual``, viewed
    like ``x`` as push_res reads it). The reduction and the residual add
    round to bf16 separately; push_res rounds once."""
    reduced = state.comm.custom_all_reduce(x)
    if residual is None:
        return x.copy_(reduced)
    return torch.add(reduced, residual.view_as(x), out=x)


_TWOSHOT_LOGGED: set = set()


def _log_twoshot_once(
    has_residual: bool, covered: bool, rows: int, nbytes: int
) -> None:
    key = (has_residual, covered)
    if key in _TWOSHOT_LOGGED:
        return
    _TWOSHOT_LOGGED.add(key)
    what = "residual (A2)" if has_residual else "no-residual (A3)"
    if covered:
        logger.info(
            "K3 AR o_proj %s 2shot push engaged: %d rows, %d bytes",
            what,
            rows,
            nbytes,
        )
    else:
        logger.warning(
            "K3 AR o_proj %s 2shot push NOT engaged at %d rows, %d bytes: "
            "CustomAllReduceV2 does not pick 2shot_push for it; keeping the "
            "push/pull window",
            what,
            rows,
            nbytes,
        )


def all_reduce(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    o_proj: bool = False,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]``; returns ``x``. ``o_proj``:
    the attention o_proj AR (kimi_k3._k3_all_reduce_add), the only call the
    SM107 riders A2/A3 (SGLANG_K31_AR_TWOSHOT_MIN_BYTES,
    SGLANG_K31_AR_NORES_PUSH_MAX_BYTES / _ABOVE) may reroute."""
    from sglang.kernels.ops.communication import all_reduce_residual as mod

    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    nbytes = x.numel() * 2
    window = _push_max_bytes()
    if o_proj:
        if _o_proj_twoshot(state, x, nbytes, residual is not None):
            return _o_proj_twoshot_add(state, x, residual)
        if residual is None:
            window = _nores_push_max_bytes()
    if envs.SGLANG_K3_AR_PUSH2.get():
        algo = _pick_push(state, nbytes, _PUSH_RES_MAX_BYTES, rows=False, window=window)
        if algo == "push2":
            return mod.all_reduce_push2_res(
                state.world_size, x, residual, use_mc=_PUSH2_RES_MC
            )
        if algo == "push":
            return mod.all_reduce_push_res(state.world_size, x, residual)
    elif nbytes <= min(window, state.comm.max_push_size):
        return mod.all_reduce_push_res(state.world_size, x, residual)
    num_blocks, unroll = _LARGE_PULL_RES if _large_pull(nbytes) else (None, None)
    return mod.all_reduce_pull_res(
        state.world_size,
        x,
        residual,
        input_mc_ptr=get_mc_ptr(x),
        num_blocks=num_blocks,
        unroll=unroll,
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
    from sglang.kernels.ops.communication import all_reduce_residual as mod

    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x if residual is None else x + residual
    if _large_pull(x.numel() * x.element_size()):
        num_blocks = max(num_blocks or 0, _LARGE_PULL_LOW_SM_BLOCKS)
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
    from sglang.kernels.ops.communication import all_reduce_residual as mod

    state = _get_state()
    assert state is not None
    if x.shape[0] == 0:
        return x
    nbytes = x.numel() * 2
    if envs.SGLANG_K3_AR_PUSH2.get():
        algo = _pick_push(
            state,
            nbytes,
            _PUSH_NORM_MAX_BYTES,
            rows=True,
            window=_norm_push_max_bytes(),
        )
        if algo == "push2":
            return mod.all_reduce_push2_norm(
                state.world_size,
                x,
                weight,
                eps,
                num_norm_rows=num_tokens,
                use_mc=_PUSH2_NORM_MC,
            )
        push = algo == "push"
    else:
        push = nbytes <= min(_norm_push_max_bytes(), state.comm.max_push_size)
    if push:
        return mod.all_reduce_push_norm(
            state.world_size,
            x,
            weight,
            eps,
            num_norm_rows=num_tokens,
        )
    num_blocks, unroll = _LARGE_PULL_NORM if _large_pull(nbytes) else (None, None)
    return mod.all_reduce_pull_norm(
        state.world_size,
        x,
        weight,
        eps,
        num_norm_rows=num_tokens,
        input_mc_ptr=get_mc_ptr(x),
        num_blocks=num_blocks,
        unroll=unroll,
    )


@lru_cache(maxsize=1)
def _finalize_push_max_bytes() -> int:
    """The window of the finalize-fused push AR + RMSNorm
    (:func:`finalize_all_reduce_push_norm`): _PUSH_MAX_BYTES, or on SM107
    SGLANG_K31_AR_FINALIZE_PUSH_MAX_BYTES (default unset). It is the push_norm cluster kernel with the deferred MoE finalize
    in its staging pass, so it fits wherever :func:`all_reduce_norm` pushes;
    the push slot (``comm.max_push_size``) still caps it at every call."""
    window = _sm107_window_env("SGLANG_K31_AR_FINALIZE_PUSH_MAX_BYTES")
    if window is None:
        return _PUSH_MAX_BYTES
    logger.info(
        "K3 AR finalize push window on SM107: %d bytes "
        "(SGLANG_K31_AR_FINALIZE_PUSH_MAX_BYTES; default %d), capped by the "
        "push slot",
        window,
        _PUSH_MAX_BYTES,
    )
    return window


def _finalize_push2(state: _State, nbytes: int) -> bool:
    """Whether the finalize-fused AR of ``nbytes`` takes the 2shot push band:
    past its 1shot window, under SGLANG_K3_AR_PUSH2 + _FINALIZE."""
    return (
        envs.SGLANG_K3_AR_PUSH2_FINALIZE.get()
        and not _push_fits(state, nbytes, _finalize_push_max_bytes())
        and _push2_fits(state, nbytes, rows=True)
    )


def finalize_push_fits(num_tokens: int) -> bool:
    """Whether a [num_tokens, NORM_DIM] latent fits the 1shot push window or
    (with ``SGLANG_K3_AR_PUSH2`` and ``SGLANG_K3_AR_PUSH2_FINALIZE``) the 2shot
    push band; the finalize-fused AR is push-only."""
    state = _get_state()
    assert state is not None
    nbytes = num_tokens * NORM_DIM * 2
    if _finalize_push2(state, nbytes):
        return True
    fits = nbytes <= min(_finalize_push_max_bytes(), state.comm.max_push_size)
    if fits and nbytes > _PUSH_MAX_BYTES:
        _log_finalize_above_default_once(num_tokens, nbytes)
    return fits


_FINALIZE_ABOVE_DEFAULT_LOGGED = False


def _log_finalize_above_default_once(num_tokens: int, nbytes: int) -> None:
    global _FINALIZE_ABOVE_DEFAULT_LOGGED
    if not _FINALIZE_ABOVE_DEFAULT_LOGGED:
        _FINALIZE_ABOVE_DEFAULT_LOGGED = True
        logger.info(
            "K3 AR finalize fusion engaged above %d bytes: %d rows, %d bytes",
            _PUSH_MAX_BYTES,
            num_tokens,
            nbytes,
        )


# Largest batch where the sharded up_proj tail beats the replicated GEMM + add3,
# measured on GB300 TP8 (the only TP size KimiK3MoE runs the tail at).
_SHARDED_UP_MAX_TOKENS = 256

# Under SGLANG_K3_SHARDED_LATENT_UP: TP world -> largest batch where the sharded
# up_proj tail is faster than the replicated GEMM + add3 (measured on GB300 with
# CUDA graphs and cold weights, per layer with the latent AR in front; TP2's
# window is smaller because its slice is half the weight).
# TP8 reads half of TP4's slice bytes; it covers DFLASH verify batches up to
# bs 85 (M 510) at 6 tokens.
_SHARDED_LATENT_UP_MAX_TOKENS = {2: 168, 4: 240, 8: 512}


def gemm_ag_up_fits(num_tokens: int) -> bool:
    """Whether :func:`gemm_ag_up_proj` covers this decode batch: within its
    row window (under SGLANG_K3_SHARDED_LATENT_UP this TP world's win range
    and SGLANG_K3_SHARDED_LATENT_UP_MAX_TOKENS; else the fused-GEMV range only
    under SGLANG_K3_SHARDED_MOE=0), and the [world, M, 7168 / world] staging
    fits a push slot."""
    from sglang.kernels.ops.communication import gemm_ag as mod

    state = _get_state()
    assert state is not None
    comm = state.comm
    if envs.SGLANG_K3_SHARDED_LATENT_UP.get():
        max_tokens = min(
            _SHARDED_LATENT_UP_MAX_TOKENS.get(state.world_size, 0),
            envs.SGLANG_K3_SHARDED_LATENT_UP_MAX_TOKENS.get(),
        )
    else:
        max_tokens = (
            _SHARDED_UP_MAX_TOKENS
            if envs.SGLANG_K3_SHARDED_MOE.get()
            else mod.GEMV_MAX_TOKENS
        )
    return (
        0 < num_tokens <= max_tokens
        and num_tokens * (2 * NORM_DIM // state.world_size) * 2 <= comm.max_push_size
        # The GEMV producer reads a single phase counter, so its grid is not
        # bound to the counter array; the consumers stride over the output with
        # at most one block per counter slot plus the cleanup block.
        and comm.config.num_push_blocks >= 2
    )


# Warp split-K's row limit for a slice GEMM; TGV above. Both beat cuBLAS at
# every decode M, measured on GB300 with CUDA graphs.
_SLICE_WARP_SPLITK_MAX_TOKENS = 32


# The slice_gemm shapes a slice table may route: the TP8 sharded MoE front
# (fp32 out) and the latent-up slice.
_SM107_SLICE_SHAPES = frozenset({(2096, 7168), (896, 3584)})


@lru_cache(maxsize=1)
def _sm107_slice_table() -> dict:
    """{(n, k): ((m_lo, m_hi, spec), ...)}: the "slice" rows of
    SGLANG_K31_SM107_SMALLM_TABLE (kimi_k3), on SM107 under
    SGLANG_K31_SM107_GEMM_TACTICS only; spec is a TGV tactic id or
    {"wsk": [output_tile, token_tile, k_tile, stages, b_loader_warps]}."""
    path = os.environ.get("SGLANG_K31_SM107_SMALLM_TABLE", "")
    if not path or not envs.SGLANG_K31_SM107_GEMM_TACTICS.get():
        return {}
    from sglang.srt.utils.device import get_device_sm

    if get_device_sm() != 107:
        return {}
    with open(path) as f:
        section = json.load(f).get("slice", {})
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
        _TGV_CUTE_EXT_TACTIC_CONFIGS,
    )

    num_tactics = len(_TGV_CUTE_EXT_TACTIC_CONFIGS)
    table = {}
    for key, ranges in section.items():
        n, k = (int(v) for v in key.split(","))
        if (n, k) not in _SM107_SLICE_SHAPES:
            raise ValueError(
                f"SM107 slice table: shape {(n, k)} is not a slice_gemm shape "
                f"{sorted(_SM107_SLICE_SHAPES)}"
            )
        rows = []
        for m_lo, m_hi, spec in ranges:
            if isinstance(spec, dict):
                wsk = spec.get("wsk")
                ok = (
                    set(spec) == {"wsk"}
                    and isinstance(wsk, list)
                    and len(wsk) == 5
                    and all(type(v) is int for v in wsk)
                )
                spec = tuple(wsk) if ok else None
            else:
                ok = type(spec) is int and 0 <= spec < num_tactics
            if not (
                ok and type(m_lo) is int and type(m_hi) is int and 1 <= m_lo <= m_hi
            ):
                raise ValueError(
                    f"SM107 slice table: bad row {(m_lo, m_hi, spec)} for {(n, k)}"
                )
            if type(spec) is tuple and m_hi > _SLICE_WARP_SPLITK_MAX_TOKENS:
                raise ValueError(
                    f"SM107 slice table: warp split-K above M 32 for {(n, k)}"
                )
            rows.append((m_lo, m_hi, spec))
        rows.sort(key=lambda r: r[0])
        for a, b in zip(rows, rows[1:]):
            if b[0] <= a[1]:
                raise ValueError(f"SM107 slice table: {a} overlaps {b} for {(n, k)}")
        table[(n, k)] = tuple(rows)
    logger.info(
        "K3 SM107 slice GEMM table on (SGLANG_K31_SM107_SMALLM_TABLE=%s): %s",
        path,
        {f"{n}x{k}": [list(r) for r in v] for (n, k), v in table.items()},
    )
    return table


def _sm107_slice_spec(m: int, n: int, k: int):
    """The slice table's spec for this GEMM, or None."""
    for m_lo, m_hi, spec in _sm107_slice_table().get((n, k), ()):
        if m_lo <= m <= m_hi:
            return spec
    return None


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
    # SM107 small-M override (default off; captured batches only, like warp split-K)
    spec = _sm107_slice_spec(m, n, weight.shape[1]) if get_is_capture_mode() else None
    if spec is not None:
        if type(spec) is tuple:
            from sglang.kernels.ops.gemm.dense_bf16_gemm_warp_splitk import (
                WarpSplitKTactic,
                run_warp_splitk_dense,
            )

            tactic = WarpSplitKTactic(*spec)
            return run_warp_splitk_dense(x, weight.t(), out, pdl=True, tactic=tactic)
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import _run_tgv

        return _run_tgv(x, weight.t(), None, out, pdl=True, tactic=spec)
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


def _up_proj_slice(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for this rank's [7168 / tp, 3584] up_proj row block under
    SGLANG_K3_SHARDED_LATENT_UP: warp split-K up to its row limit at any M
    (eager included), TGV above."""
    m, n = x.shape[0], weight.shape[0]
    if not k3_sm_supported():
        return torch.nn.functional.linear(x, weight)
    if m <= _SLICE_WARP_SPLITK_MAX_TOKENS:
        from sglang.kernels.ops.gemm.dense_bf16_gemm_warp_splitk import (
            default_tactic,
            run_warp_splitk_dense,
        )

        out = x.new_empty((m, n))
        tactic = default_tactic(m, n, weight.shape[1])
        return run_warp_splitk_dense(x, weight.t(), out, True, tactic)
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm

    return cutedsl_bf16_gemm(x, weight)


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
        rank = state.comm.rank
        block = weight[rank * rows : (rank + 1) * rows]
        if envs.SGLANG_K3_SHARDED_LATENT_UP.get():
            local = _up_proj_slice(x, block)
        else:
            local = slice_gemm(x, block)
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
    fp32 block: it fits a push slot, and the plane has a work block plus the
    counter-update block."""
    state = _get_state()
    assert state is not None
    comm = state.comm
    return (
        num_tokens > 0
        and n_local % 4 == 0
        and num_tokens * n_local * 4 <= comm.max_push_size
        and comm.config.num_push_blocks >= 2
    )


def moe_front_all_gather(local: torch.Tensor) -> torch.Tensor:
    """``[M, world * n_local]`` fp32 column all-gather of this rank's
    ``[M, n_local]`` block, whose rows may be strided (rank-major columns).
    Caller checked :func:`moe_front_all_gather_fits`."""
    from sglang.kernels.ops.communication import moe_front_ag as mod

    state = _get_state()
    assert state is not None
    out = local.new_empty((local.shape[0], local.shape[1] * state.world_size))
    return mod.moe_front_all_gather(
        state.world_size,
        local,
        out,
        widest_block=envs.SGLANG_K3_SHARDED_MOE_FRONT.get(),
    )


def finalize_all_reduce_push_norm(
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deferred MoE finalize fused into the push AR (1shot in its window;
    2shot past it under SGLANG_K3_AR_PUSH2 + _FINALIZE) + RMSNorm; ``out`` is
    output-only. Caller checked :func:`finalize_push_fits`."""
    from sglang.kernels.ops.communication import all_reduce_residual as mod

    state = _get_state()
    assert state is not None
    if _finalize_push2(state, out.numel() * 2):
        return mod.finalize_all_reduce_push2_norm(
            state.world_size,
            out,
            gemm2_out,
            expanded_idx_to_permuted_idx,
            expert_weights,
            weight,
            eps,
            use_mc=_PUSH2_NORM_MC,
        )
    return mod.finalize_all_reduce_push_norm(
        state.world_size,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
    )
