"""Deferred-fold GDN ReplaySSM target verify on FlashInfer's u-cache kernels.

Every GDN layer keeps, per mamba slot, an fp16 checkpoint ``S0`` plus a 32-entry
ring of ``(k_hat, u, g)`` records; a slot's live window is the ``hist_len`` records
starting at ``cache_base``. One FlashInfer launch per layer and verify:

* outputs every draft position from ``S0`` plus the window (no state write),
* appends the T draft records after the window,
* and, for rows with ``hist_len >= flush_min``, folds the window into ``S0`` in place.

The cursors are shared by all GDN layers of a slot and committed once per verify
from the accept count (bonus token included), with the ``hist_len`` the verify saw:
a folded row restarts its window at the old window's end with the accepted
records, any other row extends it. A fresh slot only needs ``hist_len = 0``.

Under SGLANG_ENABLE_GDN_UCACHE_REQUEST_RINGS the rings and cursors are request-lifetime
scratch indexed by the request's row in the request table (``by_request_row``), as the
ReplaySSM spec rings are: only live requests verify, so the radix tree's cached slots
carry no rings. The state ``S0`` stays at the Mamba slot; only ring addresses change.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# FlashInfer u-cache ring geometry (gdn_replay_mtp_umma.RING_SLOTS / W_RING).
RING_SLOTS = 32
_RING_MASK = RING_SLOTS - 1
_MAX_WINDOW = 16
_CURSOR_BLOCK = 256


class GdnUcacheBuffers(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer rings and their cursors, indexed by the mamba state slot, or by the
    request row under ``by_request_row``."""

    # [layers, slots, H, 32, K] normalized keys: bf16, or fp16 under
    # SGLANG_ENABLE_GDN_UCACHE_FP16_KEY_RING (gdn_ucache_key_ring_dtype).
    k_ring: torch.Tensor
    # [layers, slots, HV, 32, V] fp16 beta-scaled corrections (FlashInfer takes bf16 or fp16;
    # fp16 keeps 3 more mantissa bits through the deferred fold).
    u_ring: torch.Tensor
    # [layers, slots, HV, 32] fp32 window-relative cumulative log decay.
    g_ring: torch.Tensor
    # [slots] int32 committed window length, in [0, 16].
    hist_len: torch.Tensor
    # [slots] int32 window origin in the ring, in [0, 32).
    cache_base: torch.Tensor
    # Rows whose window holds at least this many records fold it in the verify.
    flush_min: int
    # The "slots" dimension above is the request table's rows (row 0 = padding), not Mamba slots.
    by_request_row: bool = False


class GdnUcacheRows(msgspec.Struct, frozen=True, kw_only=True):
    """One verify's per-row view of the cursors; padded rows have slot -1."""

    slots: torch.Tensor
    hist_len: torch.Tensor
    cache_base: torch.Tensor
    # int32 ring row of each batch row (by_request_row only; 0 on padded rows).
    ring_rows: Optional[torch.Tensor] = None


class GdnUcacheFlushOptions(msgspec.Struct, frozen=True, kw_only=True):
    """Launch options of the SM107 flush kernel; all off by default, none changes a result."""

    # Read the packed post-conv q / k / v and the strided b / a views in place.
    strided_qkv: bool = False
    # Programmatic dependent launch behind the conv.
    pdl: bool = False
    # Rows that fold their window first in the launch order.
    flush_first: bool = False

    def kernel_kwargs(self) -> dict:
        kw = {}
        if self.strided_qkv:
            kw["strided_inputs"] = True
        if self.pdl:
            kw["pdl"] = True
        return kw


def gdn_ucache_flush_options(device: torch.device) -> GdnUcacheFlushOptions:
    opts = GdnUcacheFlushOptions(
        strided_qkv=envs.SGLANG_ENABLE_GDN_UCACHE_STRIDED_QKV.get(),
        pdl=envs.SGLANG_ENABLE_GDN_UCACHE_PDL.get(),
        flush_first=envs.SGLANG_ENABLE_GDN_UCACHE_FLUSH_FIRST.get(),
    )
    if opts != GdnUcacheFlushOptions():
        # FlashInfer's SM100 kernel takes none of them.
        capability = torch.cuda.get_device_capability(device)
        if capability != (10, 7):
            raise ValueError(
                f"GDN u-cache flush options {opts} need the SM107 u-cache kernel, "
                f"got SM{capability[0]}{capability[1]}"
            )
    return opts


def gdn_ucache_flush_min(max_num_draft_tokens: int) -> int:
    # The laziest threshold that still fits T appended records in the window.
    return _MAX_WINDOW - max_num_draft_tokens + 1


def gdn_ucache_ring_bytes_per_slot(
    *,
    num_layers: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> int:
    k_bytes = num_k_heads * RING_SLOTS * head_k_dim * 2
    u_bytes = num_v_heads * RING_SLOTS * head_v_dim * 2
    g_bytes = num_v_heads * RING_SLOTS * 4
    return (k_bytes + u_bytes + g_bytes) * num_layers


def gdn_ucache_key_ring_dtype(device) -> torch.dtype:
    if not envs.SGLANG_ENABLE_GDN_UCACHE_FP16_KEY_RING.get():
        return torch.bfloat16
    # Only the vendored SM107 kernel takes an fp16 k ring; FlashInfer's SM100 one asserts bf16.
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 7):
        raise ValueError(
            "SGLANG_ENABLE_GDN_UCACHE_FP16_KEY_RING needs the SM107 u-cache kernel, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return torch.float16


def gdn_ucache_request_rings() -> bool:
    return envs.SGLANG_ENABLE_GDN_UCACHE_REQUEST_RINGS.get()


def allocate_gdn_ucache(
    *,
    num_layers: int,
    num_slots: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    max_num_draft_tokens: int,
    device: str,
    by_request_row: bool = False,
) -> GdnUcacheBuffers:
    # Rings are zeroed once; the kernels never read a record outside the window.
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    return GdnUcacheBuffers(
        k_ring=zeros(
            (num_layers, num_slots, num_k_heads, RING_SLOTS, head_k_dim),
            gdn_ucache_key_ring_dtype(device),
        ),
        u_ring=zeros(
            (num_layers, num_slots, num_v_heads, RING_SLOTS, head_v_dim),
            torch.float16,
        ),
        g_ring=zeros((num_layers, num_slots, num_v_heads, RING_SLOTS), torch.float32),
        hist_len=zeros((num_slots,), torch.int32),
        cache_base=zeros((num_slots,), torch.int32),
        flush_min=gdn_ucache_flush_min(max_num_draft_tokens),
        by_request_row=by_request_row,
    )


# Mamba slot 0 is the reserved dummy target of padded rows, so it is treated as
# padding (-1): the kernel then skips the row and leaves its output zero.
@triton.jit
def _gather_rows_kernel(
    slots_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    slots_out_ptr,
    hist_out_ptr,
    base_out_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    hist = tl.load(hist_pool_ptr + slot, mask=live, other=0)
    base = tl.load(base_pool_ptr + slot, mask=live, other=0)
    tl.store(slots_out_ptr + offs, tl.where(live, slot, -1).to(tl.int32), in_range)
    tl.store(hist_out_ptr + offs, hist, in_range)
    tl.store(base_out_ptr + offs, base, in_range)


@triton.jit
def _commit_cursors_kernel(
    slots_ptr,
    num_accept_tokens_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    num_rows,
    flush_min,
    RING_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    hist = tl.load(hist_pool_ptr + slot, mask=live, other=0)
    base = tl.load(base_pool_ptr + slot, mask=live, other=0)
    num_accept = tl.load(num_accept_tokens_ptr + offs, mask=live, other=0).to(tl.int32)
    folded = hist >= flush_min
    new_base = tl.where(folded, (base + hist) & RING_MASK, base)
    new_hist = tl.where(folded, num_accept, hist + num_accept)
    tl.store(base_pool_ptr + slot, new_base, live)
    tl.store(hist_pool_ptr + slot, new_hist, live)


# by_request_row: liveness still comes from the state slot (padded rows: slot <= 0), the
# cursors and ring row from the request row.
@triton.jit
def _gather_request_rows_kernel(
    slots_ptr,
    rows_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    slots_out_ptr,
    hist_out_ptr,
    base_out_ptr,
    ring_out_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    row = tl.load(rows_ptr + offs, mask=live, other=0).to(tl.int64)
    hist = tl.load(hist_pool_ptr + row, mask=live, other=0)
    base = tl.load(base_pool_ptr + row, mask=live, other=0)
    tl.store(slots_out_ptr + offs, tl.where(live, slot, -1).to(tl.int32), in_range)
    tl.store(hist_out_ptr + offs, hist, in_range)
    tl.store(base_out_ptr + offs, base, in_range)
    tl.store(ring_out_ptr + offs, tl.where(live, row, 0).to(tl.int32), in_range)


@triton.jit
def _commit_request_cursors_kernel(
    slots_ptr,
    rows_ptr,
    num_accept_tokens_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    num_rows,
    flush_min,
    RING_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    row = tl.load(rows_ptr + offs, mask=live, other=0).to(tl.int64)
    hist = tl.load(hist_pool_ptr + row, mask=live, other=0)
    base = tl.load(base_pool_ptr + row, mask=live, other=0)
    num_accept = tl.load(num_accept_tokens_ptr + offs, mask=live, other=0).to(tl.int32)
    folded = hist >= flush_min
    new_base = tl.where(folded, (base + hist) & RING_MASK, base)
    new_hist = tl.where(folded, num_accept, hist + num_accept)
    tl.store(base_pool_ptr + row, new_base, live)
    tl.store(hist_pool_ptr + row, new_hist, live)


@triton.jit
def _reset_cursors_kernel(
    slots_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    zero = tl.zeros([BLOCK], dtype=tl.int32)
    tl.store(hist_pool_ptr + slot, zero, live)
    tl.store(base_pool_ptr + slot, zero, live)


@triton.jit
def _zero_padded_rows_kernel(
    slots_ptr,
    out_ptr,
    row_numel,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(slots_ptr + row) < 0:
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            out_ptr + row.to(tl.int64) * row_numel + offs,
            tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
            offs < row_numel,
        )


# Stable partition of the verify's rows, folding rows (hist_len >= flush_min) first: the
# flush kernel's row_order (flush_first_row_order's permutation, in one launch).
@triton.jit
def _flush_first_order_kernel(
    hist_ptr,
    order_ptr,
    num_rows,
    flush_min,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    valid = offs < num_rows
    hist = tl.load(hist_ptr + offs, mask=valid, other=0)
    folds = (valid & (hist >= flush_min)).to(tl.int32)
    rest = (valid & (hist < flush_min)).to(tl.int32)
    num_folds = tl.sum(folds, axis=0)
    pos = tl.where(
        folds == 1,
        tl.cumsum(folds, axis=0) - 1,
        num_folds + tl.cumsum(rest, axis=0) - 1,
    )
    tl.store(order_ptr + pos, offs.to(tl.int32), mask=valid)


# Copies the packed [q | k | v] token rows of one strided view into three
# contiguous buffers in a single launch (the FlashInfer verify reads contiguous q/k/v).
@triton.jit
def _split_qkv_kernel(
    src_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    src_row_stride,
    Q_COLS: tl.constexpr,
    K_COLS: tl.constexpr,
    V_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = Q_COLS + K_COLS + V_COLS
    x = tl.load(src_ptr + token * src_row_stride + cols, mask=cols < total)
    tl.store(q_ptr + token * Q_COLS + cols, x, mask=cols < Q_COLS)
    k_cols = cols - Q_COLS
    tl.store(k_ptr + token * K_COLS + k_cols, x, mask=(k_cols >= 0) & (k_cols < K_COLS))
    v_cols = cols - Q_COLS - K_COLS
    tl.store(v_ptr + token * V_COLS + v_cols, x, mask=(v_cols >= 0) & (v_cols < V_COLS))


# Refreshes every layer's fp32 gate-parameter copy from its bf16 source in one
# launch; the int64 tables hold the source and destination addresses.
@triton.jit
def _refresh_gate_params_kernel(
    src_ptrs,
    dst_ptrs,
    numel,
    BLOCK: tl.constexpr,
):
    entry = tl.program_id(0)
    src = tl.load(src_ptrs + entry).to(tl.pointer_type(tl.bfloat16))
    dst = tl.load(dst_ptrs + entry).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK)
    mask = offs < numel
    tl.store(dst + offs, tl.load(src + offs, mask=mask).to(tl.float32), mask=mask)


def gather_gdn_ucache_rows(
    *,
    slots: torch.Tensor,
    buffers: GdnUcacheBuffers,
    request_rows: Optional[torch.Tensor] = None,
) -> GdnUcacheRows:
    if buffers.by_request_row:
        return _gather_gdn_ucache_request_rows(
            slots=slots, request_rows=request_rows, buffers=buffers
        )
    num_rows = slots.shape[0]
    rows = GdnUcacheRows(
        slots=torch.empty(num_rows, dtype=torch.int32, device=slots.device),
        hist_len=torch.empty(num_rows, dtype=torch.int32, device=slots.device),
        cache_base=torch.empty(num_rows, dtype=torch.int32, device=slots.device),
    )
    if num_rows > 0:
        _gather_rows_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
            slots,
            buffers.hist_len,
            buffers.cache_base,
            rows.slots,
            rows.hist_len,
            rows.cache_base,
            num_rows,
            BLOCK=_CURSOR_BLOCK,
        )
    return rows


def _gather_gdn_ucache_request_rows(
    *,
    slots: torch.Tensor,
    request_rows: Optional[torch.Tensor],
    buffers: GdnUcacheBuffers,
) -> GdnUcacheRows:
    num_rows = slots.shape[0]
    if request_rows is None or request_rows.shape[0] < num_rows:
        raise RuntimeError(
            "GDN u-cache rings by request row: the verify needs its request rows."
        )
    device = slots.device
    rows = GdnUcacheRows(
        slots=torch.empty(num_rows, dtype=torch.int32, device=device),
        hist_len=torch.empty(num_rows, dtype=torch.int32, device=device),
        cache_base=torch.empty(num_rows, dtype=torch.int32, device=device),
        ring_rows=torch.empty(num_rows, dtype=torch.int32, device=device),
    )
    if num_rows > 0:
        _gather_request_rows_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
            slots,
            request_rows,
            buffers.hist_len,
            buffers.cache_base,
            rows.slots,
            rows.hist_len,
            rows.cache_base,
            rows.ring_rows,
            num_rows,
            BLOCK=_CURSOR_BLOCK,
        )
    return rows


def commit_gdn_ucache_cursors(
    *,
    buffers: GdnUcacheBuffers,
    slots: torch.Tensor,
    num_accept_tokens: torch.Tensor,
    request_rows: Optional[torch.Tensor] = None,
) -> None:
    """Advance the cursors of the verified slots (their request rows under
    ``by_request_row``); must follow that verify and precede the next one, with no
    other cursor write in between."""
    num_rows = slots.shape[0]
    if num_rows == 0:
        return
    if buffers.by_request_row:
        if request_rows is None or request_rows.shape[0] < num_rows:
            raise RuntimeError(
                "GDN u-cache rings by request row: the commit needs its request rows."
            )
        _commit_request_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
            slots,
            request_rows,
            num_accept_tokens,
            buffers.hist_len,
            buffers.cache_base,
            num_rows,
            buffers.flush_min,
            RING_MASK=_RING_MASK,
            BLOCK=_CURSOR_BLOCK,
        )
        return
    _commit_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        slots,
        num_accept_tokens,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        buffers.flush_min,
        RING_MASK=_RING_MASK,
        BLOCK=_CURSOR_BLOCK,
    )


def reset_gdn_ucache_cursors(*, buffers: GdnUcacheBuffers, slots: torch.Tensor) -> None:
    """Empty the windows of the slots a target extend writes, so the extend and
    the verifies after it read only the checkpoint ``S0``.

    The slot may carry its previous owner's cursors (a recycled slot, a prefix-cache
    copy-on-write or load-back), while the state it receives is always canonical:
    prefix-cached states are prefill-written, decode-time states never enter the
    tree. ``slots`` are state-pool (physical) ids; ``slot <= 0`` rows are padding
    and left untouched. One launch, no host sync."""
    global _reset_logged
    num_rows = slots.shape[0]
    if num_rows == 0:
        return
    if not _reset_logged:
        _reset_logged = True
        logger.info(
            "GDN u-cache: target extends restart their slots' windows (first: %d rows)",
            num_rows,
        )
    _reset_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        slots,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        BLOCK=_CURSOR_BLOCK,
    )


def reset_gdn_ucache_request_rows(
    *, buffers: GdnUcacheBuffers, request_rows: torch.Tensor
) -> None:
    """``reset_gdn_ucache_cursors`` under ``by_request_row``: empty the windows of the
    request rows a target extend writes (row 0 is the padding row and left untouched)."""
    global _reset_logged
    assert buffers.by_request_row
    num_rows = request_rows.shape[0]
    if num_rows == 0:
        return
    if not _reset_logged:
        _reset_logged = True
        logger.info(
            "GDN u-cache: target extends restart their request rows' windows (first: %d rows)",
            num_rows,
        )
    _reset_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        request_rows,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        BLOCK=_CURSOR_BLOCK,
    )


def _load_flush_kernel(device: torch.device) -> Callable:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 7):
        # The vendored copy of flashinfer#6791 (gdn_ucache_rubin/__init__.py), not the
        # image's FlashInfer: 0.7.2 has no Rubin build, and a later one may differ.
        from sglang.kernels.ops.attention.linear.gdn_ucache_rubin.gdn_replay_mtp_umma_rubin import (
            gated_delta_rule_mtp_ucache_flush_umma_rubin,
        )

        return gated_delta_rule_mtp_ucache_flush_umma_rubin
    if capability == (10, 0):
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
            gated_delta_rule_mtp_ucache_flush_umma,
        )

        return gated_delta_rule_mtp_ucache_flush_umma
    raise RuntimeError(
        f"GDN u-cache verify needs SM100 or SM107, got SM{capability[0]}{capability[1]}"
    )


def gdn_ucache_pd_dtype_tag(kv_cache_dtype_str: str) -> str:
    """The KV dtype string a PD handshake compares between prefill and decode; under
    SGLANG_ENABLE_GDN_UCACHE_PD it names the GDN state dtype too, since the state
    transfer copies bytes and bf16 and fp16 slots have the same size."""
    if not envs.SGLANG_ENABLE_GDN_UCACHE_PD.get():
        return kv_cache_dtype_str
    from sglang.srt.runtime_context import get_exec

    return f"{kv_cache_dtype_str}+gdn_state_{get_exec().mamba.mamba_ssm_dtype}"


def gdn_ucache_blocks_state_backup(mamba_pool) -> bool:
    """Whether a host backup of a slot's checkpoint would lose state: under the u-cache
    verify the checkpoint lags the slot's unfolded ring window, which no backup copies."""
    global _backup_refusal_logged
    if mamba_pool.gdn_ucache is None:
        return False
    if not _backup_refusal_logged:
        _backup_refusal_logged = True
        logger.warning(
            "GDN u-cache verify: a retracted request's state cannot be backed up "
            "(its checkpoint lags the ring window); the request is aborted."
        )
    return True


_backup_refusal_logged = False
_engaged_logged = False
_reset_logged = False


class GdnUcacheVerifier:
    """Runs the per-layer verify launches of one target model.

    Per forward, the first GDN layer gathers the cursors, zeroes the padded rows of
    the shared output buffer and refreshes every layer's fp32 gate parameters; each
    layer then costs one q/k/v split plus the FlashInfer launch (no split with
    SGLANG_ENABLE_GDN_UCACHE_STRIDED_QKV, see GdnUcacheFlushOptions)."""

    def __init__(
        self,
        *,
        device: torch.device,
        first_layer_id: int,
        by_request_row: bool = False,
        num_ring_rows: int = 0,
        num_state_slots: int = 0,
    ):
        global _engaged_logged
        from sglang.srt.runtime_context import get_exec

        self._flush_kernel = _load_flush_kernel(device)
        self._options = gdn_ucache_flush_options(device)
        # SGLANG_ENABLE_GDN_UCACHE_FUSED_CONV: the vendored SM107 kernel's conv prologue, which
        # its strided, dependent launch carries
        self.fused_conv_capable = (
            self._options.strided_qkv
            and self._options.pdl
            and torch.cuda.get_device_capability(device) == (10, 7)
        )
        self._kernel_kwargs = self._options.kernel_kwargs()
        self._by_request_row = by_request_row
        if by_request_row:
            # Only the vendored SM107 kernel takes ring_indices; FlashInfer's SM100 one does not.
            capability = torch.cuda.get_device_capability(device)
            if capability != (10, 7):
                raise ValueError(
                    "SGLANG_ENABLE_GDN_UCACHE_REQUEST_RINGS needs the SM107 u-cache kernel, "
                    f"got SM{capability[0]}{capability[1]}"
                )
            logger.info(
                "SGLANG_ENABLE_GDN_UCACHE_REQUEST_RINGS: the u-cache rings and cursors are "
                "indexed by request row (%d rows), the GDN state by Mamba slot (%d slots)",
                num_ring_rows,
                num_state_slots,
            )
        if not _engaged_logged:
            _engaged_logged = True
            logger.info(
                "GDN u-cache verify engaged: %s.%s, first GDN layer %d, PD decode %s, "
                "radix extend-time checkpoints %s, flush options %s",
                self._flush_kernel.__module__,
                self._flush_kernel.__name__,
                first_layer_id,
                envs.SGLANG_ENABLE_GDN_UCACHE_PD.get(),
                get_exec().mamba.enable_mamba_extra_buffer,
                self._options,
            )
        self._device = device
        self._first_layer_id = first_layer_id
        # layer_id -> fp32 (A_log, dt_bias) passed to FlashInfer, allocated eagerly.
        self._gate_params: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # (bf16 source, fp32 destination) pairs refreshed by one launch per forward.
        self._gate_refresh: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._gate_refresh_tables: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._rows: Optional[GdnUcacheRows] = None
        # This verify's flush-first row order, shared by all layers.
        self._row_order: Optional[torch.Tensor] = None
        self._strided_logged = False
        # Shared by all layers: a layer's output is consumed (gated norm and
        # out_proj) before the next GDN layer's verify overwrites it.
        self._output: Optional[torch.Tensor] = None
        self._qkv: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def target_verify(
        self,
        *,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        k_ring: torch.Tensor,
        u_ring: torch.Tensor,
        g_ring: torch.Tensor,
        buffers: GdnUcacheBuffers,
        cache_indices: torch.Tensor,
        num_draft_tokens: int,
        request_rows: Optional[torch.Tensor] = None,
        fused_conv=None,
        fused_norm=None,
    ) -> torch.Tensor:
        """Returns the verify output in ``value``'s shape; padded rows are zero.
        ``request_rows`` (the batch's request-table rows) index the rings under
        ``by_request_row``. ``fused_conv`` (a GdnFusedConv, ``fused_conv_capable`` only): the
        launch computes query / key / value, views of one [B, T, D] conv output, itself.
        ``fused_norm`` (a GdnFusedNorm, same kernel): the output is the gated RMSNorm of it."""
        num_v_heads, head_v_dim = value.shape[-2], value.shape[-1]
        num_k_heads, head_k_dim = key.shape[-2], key.shape[-1]
        batch = value.numel() // (num_draft_tokens * num_v_heads * head_v_dim)
        token_shape = (batch, num_draft_tokens)
        output = self._output_buffer(
            batch=batch, row_shape=(num_draft_tokens, num_v_heads, head_v_dim)
        )
        if layer_id == self._first_layer_id:
            self._rows = gather_gdn_ucache_rows(
                slots=cache_indices[:batch],
                buffers=buffers,
                request_rows=None if request_rows is None else request_rows[:batch],
            )
            self._refresh_gate_params()
            if self._options.flush_first:
                self._row_order = _flush_first_order(
                    hist_len=self._rows.hist_len, flush_min=buffers.flush_min
                )
            # Last before the flush: under SGLANG_ENABLE_GDN_UCACHE_PDL its warps 0/1 read the row
            # order and cursors before griddepcontrol.wait, so the kernel just before it may write
            # only what they never read (padded output rows).
            self._zero_padded_rows(output)
        rows = self._rows
        assert rows is not None and rows.slots.shape[0] == batch
        A_log_fp32, dt_bias_fp32 = self._fp32_gate_params(layer_id, A_log, dt_bias)
        if self._options.strided_qkv:
            # The kernel reads the views in place (or copies them itself if it cannot).
            q, k, v = query, key, value
        else:
            q, k, v = self._contiguous_qkv(query, key, value)
        q = q.reshape(*token_shape, num_k_heads, head_k_dim)
        k = k.reshape(*token_shape, num_k_heads, head_k_dim)
        v = v.reshape(*token_shape, num_v_heads, head_v_dim)
        a = a.reshape(*token_shape, num_v_heads)
        b = b.reshape(*token_shape, num_v_heads)
        if self._options.strided_qkv and not self._strided_logged:
            self._strided_logged = True
            _log_strided_inputs(q=q, k=k, v=v, a=a, b=b)
        self._flush_kernel(
            A_log=A_log_fp32,
            a=a,
            dt_bias=dt_bias_fp32,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=rows.slots,
            scale=head_k_dim**-0.5,
            output=output,
            k_cache=k_ring,
            u_cache=u_ring,
            g_cache=g_ring,
            hist_len=rows.hist_len,
            cache_base=rows.cache_base,
            flush_min=buffers.flush_min,
            restart_hist_on_flush=False,
            **self._kernel_kwargs,
            **({"row_order": self._row_order} if self._options.flush_first else {}),
            **({"ring_indices": rows.ring_rows} if self._by_request_row else {}),
            **({"fused_conv": fused_conv} if fused_conv is not None else {}),
            **({"fused_norm": fused_norm} if fused_norm is not None else {}),
        )
        return output.reshape(value.shape)

    def warm_up(
        self,
        *,
        max_batch_size: int,
        num_draft_tokens: int,
        state: torch.Tensor,
        k_ring: torch.Tensor,
        u_ring: torch.Tensor,
        g_ring: torch.Tensor,
        flush_min: int,
        fused_conv: bool = False,
        fused_norm_max_batch: int = 0,
    ) -> None:
        """Compiles the kernels, allocates the shared buffers and builds
        FlashInfer's per-batch-size helper tensors eagerly, so CUDA graph capture
        creates none of them; with ``fused_conv`` / ``fused_norm_max_batch`` (> 0), those builds
        too."""
        num_k_heads, head_k_dim = k_ring.shape[1], k_ring.shape[3]
        num_v_heads, head_v_dim = u_ring.shape[1], u_ring.shape[3]
        device = state.device
        tokens = (max_batch_size, num_draft_tokens)
        self._output_buffer(
            batch=max_batch_size, row_shape=(num_draft_tokens, num_v_heads, head_v_dim)
        )
        self._qkv_buffers(
            num_tokens=max_batch_size * num_draft_tokens,
            cols=(
                num_k_heads * head_k_dim,
                num_k_heads * head_k_dim,
                num_v_heads * head_v_dim,
            ),
        )

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        if self._options.strided_qkv:
            # The served layout (the strides are compiled in): q / k / v of packed
            # [q | k | v] token rows, b / a of [b | a] rows.
            qk_cols, v_cols = num_k_heads * head_k_dim, num_v_heads * head_v_dim
            packed = zeros(*tokens, 2 * qk_cols + v_cols)
            q = packed[..., :qk_cols].unflatten(-1, (num_k_heads, head_k_dim))
            k = packed[..., qk_cols : 2 * qk_cols].unflatten(-1, (num_k_heads, head_k_dim))
            v = packed[..., 2 * qk_cols :].unflatten(-1, (num_v_heads, head_v_dim))
            gates = zeros(*tokens, 2 * num_v_heads)
            b, a = gates[..., :num_v_heads], gates[..., num_v_heads:]
        else:
            q = k = zeros(*tokens, num_k_heads, head_k_dim)
            v = zeros(*tokens, num_v_heads, head_v_dim)
            a = b = zeros(*tokens, num_v_heads)
        gate_param = zeros(num_v_heads, dtype=torch.float32)
        cursor = zeros(max_batch_size, dtype=torch.int32)
        padded = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
        order = torch.arange(max_batch_size, dtype=torch.int32, device=device)
        ring_rows = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        for batch in range(1, max_batch_size + 1):
            self._flush_kernel(
                A_log=gate_param,
                a=a[:batch],
                dt_bias=gate_param,
                q=q[:batch],
                k=k[:batch],
                v=v[:batch],
                b=b[:batch],
                initial_state_source=state,
                initial_state_indices=padded[:batch],
                k_cache=k_ring,
                u_cache=u_ring,
                g_cache=g_ring,
                hist_len=cursor[:batch],
                cache_base=cursor[:batch],
                flush_min=flush_min,
                restart_hist_on_flush=False,
                **self._kernel_kwargs,
                **({"row_order": order[:batch]} if self._options.flush_first else {}),
                **({"ring_indices": ring_rows[:batch]} if self._by_request_row else {}),
            )
        if (fused_conv or fused_norm_max_batch > 0) and self.fused_conv_capable:
            self._warm_up_fused(
                q=q, k=k, v=v, a=a, b=b, gate_param=gate_param, state=state,
                k_ring=k_ring, u_ring=u_ring, g_ring=g_ring, cursor=cursor,
                padded=padded, order=order, ring_rows=ring_rows, flush_min=flush_min,
                fused_conv=fused_conv, fused_norm_max_batch=fused_norm_max_batch,
            )

    def _warm_up_fused(
        self, *, q, k, v, a, b, gate_param, state, k_ring, u_ring, g_ring, cursor, padded,
        order, ring_rows, flush_min, fused_conv, fused_norm_max_batch,
    ) -> None:
        """The conv-prologue / gated-norm builds (SGLANG_ENABLE_GDN_UCACHE_FUSED_CONV / _NORM) at
        every batch size that takes them. Every row is padded, so the launches touch no buffer."""
        from sglang.kernels.ops.attention.linear.gdn_ucache_rubin.gdn_replay_mtp_umma_rubin import (
            GdnFusedConv,
            GdnFusedNorm,
            fused_conv_unsupported_reason,
        )

        max_batch, T = q.shape[0], q.shape[1]
        H, HV = q.shape[2], v.shape[2]
        D = (2 * H + HV) * q.shape[3]
        device = q.device
        bf16 = torch.bfloat16
        x_all = torch.zeros(max_batch * T * D, dtype=bf16, device=device)
        win = torch.zeros(1, D, T + 2, dtype=bf16, device=device)
        win_view = win.as_strided((1, T, D, 3), ((T + 2) * D, 1, T + 2, 1))
        weight = torch.zeros(D, 4, dtype=bf16, device=device)
        conv_state = torch.zeros(1, D, 3, dtype=bf16, device=device)
        win_rows = torch.zeros(max_batch, dtype=torch.int32, device=device)
        z_all = torch.zeros(max_batch * T, HV, q.shape[3], dtype=bf16, device=device)
        norm_w = torch.zeros(q.shape[3], dtype=bf16, device=device)
        built = 0
        for batch in range(1, max_batch + 1):
            fc = GdnFusedConv(
                x=x_all[: batch * T * D].view(batch, T, D),
                weight=weight,
                conv_state=conv_state,
                window=win_view,
                window_indices=win_rows[:batch],
            )
            if not fused_conv or fused_conv_unsupported_reason(
                fc, num_k_heads=H, num_v_heads=HV
            ) is not None:
                fc = None
            fn = (
                GdnFusedNorm(z=z_all[: batch * T], weight=norm_w, eps=1e-6)
                if batch <= fused_norm_max_batch
                and gdn_ucache_fused_norm_applies(batch * T * HV, device)
                else None
            )
            if fc is None and fn is None:
                continue
            built += 1
            self._flush_kernel(
                A_log=gate_param,
                a=a[:batch],
                dt_bias=gate_param,
                q=q[:batch],
                k=k[:batch],
                v=v[:batch],
                b=b[:batch],
                initial_state_source=state,
                initial_state_indices=padded[:batch],
                k_cache=k_ring,
                u_cache=u_ring,
                g_cache=g_ring,
                hist_len=cursor[:batch],
                cache_base=cursor[:batch],
                flush_min=flush_min,
                restart_hist_on_flush=False,
                **({"fused_conv": fc} if fc is not None else {}),
                **({"fused_norm": fn} if fn is not None else {}),
                **self._kernel_kwargs,
                **({"row_order": order[:batch]} if self._options.flush_first else {}),
                **({"ring_indices": ring_rows[:batch]} if self._by_request_row else {}),
            )
        logger.info(
            "SGLANG_ENABLE_GDN_UCACHE_FUSED_CONV / _NORM (%s / up to batch %d): flush builds warmed "
            "at %d of %d batch sizes",
            fused_conv,
            fused_norm_max_batch,
            built,
            max_batch,
        )

    def _output_buffer(self, *, batch: int, row_shape: tuple) -> torch.Tensor:
        out = self._output
        if out is None or out.shape[0] < batch or tuple(out.shape[1:]) != row_shape:
            _assert_not_capturing("the verify output buffer")
            out = torch.zeros(
                (batch, *row_shape), dtype=torch.bfloat16, device=self._device
            )
            self._output = out
        return out[:batch]

    def _zero_padded_rows(self, output: torch.Tensor) -> None:
        # Valid rows are fully written by every layer; padded rows are skipped by
        # the kernel and must read as zero.
        rows = self._rows
        batch = output.shape[0]
        if batch == 0:
            return
        row_numel = output[0].numel()
        block = 4096
        _zero_padded_rows_kernel[(batch, triton.cdiv(row_numel, block))](
            rows.slots, output, row_numel, BLOCK=block
        )

    def _qkv_buffers(
        self, *, num_tokens: int, cols: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bufs = self._qkv
        if (
            bufs is None
            or bufs[0].shape[0] < num_tokens
            or tuple(buf.shape[1] for buf in bufs) != cols
        ):
            _assert_not_capturing("the q/k/v split buffers")
            bufs = tuple(
                torch.empty((num_tokens, c), dtype=torch.bfloat16, device=self._device)
                for c in cols
            )
            self._qkv = bufs
        return tuple(buf[:num_tokens] for buf in bufs)

    def _contiguous_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.is_contiguous() and key.is_contiguous() and value.is_contiguous():
            return query, key, value
        q_cols, k_cols, v_cols = (
            x.shape[-2] * x.shape[-1] for x in (query, key, value)
        )
        num_tokens = query.numel() // q_cols
        row_stride = query.stride(-3)
        elem = query.element_size()
        packed = (
            key.data_ptr() == query.data_ptr() + q_cols * elem
            and value.data_ptr() == key.data_ptr() + k_cols * elem
            and key.stride(-3) == row_stride
            and value.stride(-3) == row_stride
            and all(
                x.stride(-1) == 1 and x.stride(-2) == x.shape[-1]
                for x in (query, key, value)
            )
        )
        if not packed:
            return query.contiguous(), key.contiguous(), value.contiguous()
        q, k, v = self._qkv_buffers(
            num_tokens=num_tokens, cols=(q_cols, k_cols, v_cols)
        )
        block = 1024
        _split_qkv_kernel[(num_tokens, triton.cdiv(q_cols + k_cols + v_cols, block))](
            query,
            q,
            k,
            v,
            row_stride,
            Q_COLS=q_cols,
            K_COLS=k_cols,
            V_COLS=v_cols,
            BLOCK=block,
        )
        return q, k, v

    def _fp32_gate_params(
        self, layer_id: int, A_log: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self._gate_params.get(layer_id)
        if params is None:
            _assert_not_capturing("layer gate parameters (run an eager forward first)")
            params = (_fp32_view_or_buffer(A_log), _fp32_view_or_buffer(dt_bias))
            self._gate_params[layer_id] = params
            for buf, src in zip(params, (A_log, dt_bias)):
                if buf.data_ptr() == src.data_ptr():
                    continue
                buf.copy_(src)
                if src.dtype == torch.bfloat16 and src.is_contiguous():
                    self._gate_refresh.append((src, buf))
                    self._gate_refresh_tables = None
        # Other non-aliased sources (not bf16 contiguous) are refreshed per call.
        for buf, src in zip(params, (A_log, dt_bias)):
            if buf.data_ptr() != src.data_ptr() and not (
                src.dtype == torch.bfloat16 and src.is_contiguous()
            ):
                buf.copy_(src)
        return params

    def _refresh_gate_params(self) -> None:
        # In-place weight updates reach graph replays: the refresh is captured.
        if not self._gate_refresh:
            return
        if self._gate_refresh_tables is None:
            _assert_not_capturing("the gate refresh tables")
            self._gate_refresh_tables = tuple(
                torch.tensor(
                    [t.data_ptr() for t in group],
                    dtype=torch.int64,
                    device=self._device,
                )
                for group in zip(*self._gate_refresh)
            )
        numel = self._gate_refresh[0][0].numel()
        assert all(src.numel() == numel for src, _ in self._gate_refresh)
        src_ptrs, dst_ptrs = self._gate_refresh_tables
        _refresh_gate_params_kernel[(len(self._gate_refresh),)](
            src_ptrs, dst_ptrs, numel, BLOCK=triton.next_power_of_2(numel)
        )


def gdn_ucache_fused_norm_applies(rows: int, device) -> bool:
    """SGLANG_ENABLE_GDN_UCACHE_FUSED_NORM: the flush epilogue reduces a 128-wide row as 16
    threads x 8 elements, which is the stock gated-norm launch's order once it holds >= 2 rows per
    program (verify B >= 3); smaller verifies keep the separate norm."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block

    return calc_rows_per_block(rows, device) >= 2


def _flush_first_order(*, hist_len: torch.Tensor, flush_min: int) -> torch.Tensor:
    num_rows = hist_len.shape[0]
    order = torch.empty(num_rows, dtype=torch.int32, device=hist_len.device)
    if num_rows > 0:
        _flush_first_order_kernel[(1,)](
            hist_len,
            order,
            num_rows,
            flush_min,
            BLOCK=max(16, triton.next_power_of_2(num_rows)),
        )
    return order


def _log_strided_inputs(*, q, k, v, a, b) -> None:
    from sglang.kernels.ops.attention.linear.gdn_ucache_rubin.gdn_replay_mtp_umma_rubin import (
        strided_inputs_layout,
    )

    strides = dict(q=q.stride(), v=v.stride(), a=a.stride())
    if strided_inputs_layout(q, k, v, a, b) is None:
        logger.warning(
            "SGLANG_ENABLE_GDN_UCACHE_STRIDED_QKV: the flush kernel copies q / k / v / a / b "
            "(views it cannot address in place: strides %s)",
            strides,
        )
    else:
        logger.info(
            "SGLANG_ENABLE_GDN_UCACHE_STRIDED_QKV: the flush kernel reads q / k / v / a / b "
            "in place (strides %s)",
            strides,
        )


def _assert_not_capturing(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"GDN u-cache verify: {what} must be prepared before CUDA graph capture."
        )


def _fp32_view_or_buffer(param: torch.Tensor) -> torch.Tensor:
    if param.dtype == torch.float32 and param.is_contiguous():
        return param.detach()
    return torch.empty(param.shape, dtype=torch.float32, device=param.device)
