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

With radix decode inserts, a row whose accepted tokens cross a track point gets
``fold(S0, window up to that point)`` written into its track slot after the
commit (FlashInfer's ``gdn_prefix_materialize``), so every state the radix tree
holds has an empty window.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import msgspec
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# FlashInfer u-cache ring geometry (gdn_replay_mtp_umma.RING_SLOTS / W_RING).
RING_SLOTS = 32
_RING_MASK = RING_SLOTS - 1
_MAX_WINDOW = 16
_CURSOR_BLOCK = 256


class GdnUcacheBuffers(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer rings and per-slot cursors, indexed by the mamba state slot."""

    # [layers, slots, H, 32, K] bf16 normalized keys.
    k_ring: torch.Tensor
    # [layers, slots, HV, 32, V] bf16 beta-scaled corrections.
    u_ring: torch.Tensor
    # [layers, slots, HV, 32] fp32 window-relative cumulative log decay.
    g_ring: torch.Tensor
    # [slots] int32 committed window length, in [0, 16].
    hist_len: torch.Tensor
    # [slots] int32 window origin in the ring, in [0, 32).
    cache_base: torch.Tensor
    # Rows whose window holds at least this many records fold it in the verify.
    flush_min: int


class GdnUcacheRows(msgspec.Struct, frozen=True, kw_only=True):
    """One verify's per-row view of the cursors; padded rows have slot -1."""

    slots: torch.Tensor
    hist_len: torch.Tensor
    cache_base: torch.Tensor


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
) -> GdnUcacheBuffers:
    # Rings are zeroed once; the kernels never read a record outside the window.
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    return GdnUcacheBuffers(
        k_ring=zeros(
            (num_layers, num_slots, num_k_heads, RING_SLOTS, head_k_dim),
            torch.bfloat16,
        ),
        u_ring=zeros(
            (num_layers, num_slots, num_v_heads, RING_SLOTS, head_v_dim),
            torch.bfloat16,
        ),
        g_ring=zeros((num_layers, num_slots, num_v_heads, RING_SLOTS), torch.float32),
        hist_len=zeros((num_slots,), torch.int32),
        cache_base=zeros((num_slots,), torch.int32),
        flush_min=gdn_ucache_flush_min(max_num_draft_tokens),
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


# One program per plan layer (each recomputes the cheap row scan). Entry
# e = layer * num_rows + row; the active list holds every layer's live entries
# first, then -1. Reads the post-commit cursors.
@triton.jit
def _plan_track_kernel(
    slots_ptr,
    num_accept_tokens_ptr,
    track_slots_ptr,
    steps_to_track_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    src_out_ptr,
    dst_out_ptr,
    base_out_ptr,
    count_out_ptr,
    active_out_ptr,
    num_active_out_ptr,
    num_rows,
    layer_slot_stride,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    num_layers = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    dst = tl.load(track_slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    step = tl.load(steps_to_track_ptr + offs, mask=in_range, other=-1).to(tl.int32)
    live = in_range & (slot > 0) & (dst > 0) & (step >= 0)
    hist = tl.load(hist_pool_ptr + slot, mask=live, other=0)
    base = tl.load(base_pool_ptr + slot, mask=live, other=0)
    num_accept = tl.load(num_accept_tokens_ptr + offs, mask=live, other=0).to(tl.int32)
    # This verify's accepted records start at window index hist - num_accept on
    # folded and unfolded rows alike; the snapshot covers them through `step`.
    count = hist - num_accept + step + 1
    live_i32 = live.to(tl.int32)
    num_live = tl.sum(live_i32, axis=0)
    rank = tl.cumsum(live_i32, axis=0) - 1
    entry = layer * num_rows + offs
    layer_offset = layer.to(tl.int64) * layer_slot_stride
    src_flat = tl.where(live, layer_offset + slot, -1).to(tl.int32)
    dst_flat = tl.where(live, layer_offset + dst, -1).to(tl.int32)
    tl.store(src_out_ptr + entry, src_flat, in_range)
    tl.store(dst_out_ptr + entry, dst_flat, in_range)
    tl.store(base_out_ptr + entry, tl.where(live, base, 0), in_range)
    tl.store(count_out_ptr + entry, tl.where(live, count, -1), in_range)
    tl.store(active_out_ptr + layer * num_live + rank, entry, live)
    tail = num_layers * num_live + layer * (num_rows - num_live) + offs - num_live
    tl.store(
        active_out_ptr + tail,
        tl.full([BLOCK], -1, tl.int32),
        in_range & (offs >= num_live),
    )
    if layer == 0:
        tl.store(num_active_out_ptr, num_live * num_layers)
        # A track slot holds a folded state, so its own window is empty.
        zero = tl.zeros([BLOCK], dtype=tl.int32)
        tl.store(hist_pool_ptr + dst, zero, live)
        tl.store(base_pool_ptr + dst, zero, live)


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
    *, slots: torch.Tensor, buffers: GdnUcacheBuffers
) -> GdnUcacheRows:
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


def commit_gdn_ucache_cursors(
    *,
    buffers: GdnUcacheBuffers,
    slots: torch.Tensor,
    num_accept_tokens: torch.Tensor,
) -> None:
    """Advance the cursors of the verified slots; must follow that verify and
    precede the next one, with no other cursor write in between."""
    num_rows = slots.shape[0]
    if num_rows == 0:
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
    num_rows = slots.shape[0]
    if num_rows == 0:
        return
    _reset_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        slots,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        BLOCK=_CURSOR_BLOCK,
    )


def _load_prefix_materialize() -> Callable:
    # FlashInfer fixes PM's dtypes when flashinfer.gdn_kernels is first imported
    # (GDN_UCACHE_STATE_DTYPE, set by the server-args hook); the pool is fp16 state
    # with bf16 rings.
    try:
        from flashinfer.gdn_kernels.gdn_prefix_materialize import (
            gdn_prefix_materialize,
            ring_ty_torch,
            state_ty_torch,
        )
    except ImportError as e:
        raise RuntimeError(
            "--enable-gdn-replayssm-ucache with radix decode inserts needs "
            "FlashInfer >= 0.7.3 (flashinfer.gdn_kernels.gdn_prefix_materialize)."
        ) from e
    if (state_ty_torch(), ring_ty_torch()) != (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "gdn_prefix_materialize resolved state/ring dtypes "
            f"{state_ty_torch()}/{ring_ty_torch()}, need float16/bfloat16: set "
            "GDN_UCACHE_STATE_DTYPE=fp16 (and leave GDN_UCACHE_RING_DTYPE unset) "
            "before flashinfer.gdn_kernels is first imported."
        )
    return gdn_prefix_materialize


class _TrackPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Static gdn_prefix_materialize arguments for up to ``rows`` batch rows
    (times the plan layers); every tensor is [rows * layers] int32."""

    rows: int
    src: torch.Tensor
    dst: torch.Tensor
    base: torch.Tensor
    count: torch.Tensor
    active: torch.Tensor
    # [1] int32: live entries in ``active``.
    num_active: torch.Tensor


class GdnUcacheTrackMaterializer:
    """Writes the decode-time radix track snapshots of a u-cache pool.

    After a verify's cursor commit, every row whose accepted tokens cross a track
    point gets ``fold(S0, window up to that point)`` in its track slot, for all GDN
    layers, and the track slot's cursors are zeroed. One plan launch builds the
    FlashInfer arguments on the device (no host sync, graph-capturable); then one
    FlashInfer launch over ``[layers * slots]`` views of the layer-major pools, or
    one per layer if a pool is not contiguous."""

    def __init__(self, *, state: torch.Tensor, buffers: GdnUcacheBuffers):
        self._materialize = _load_prefix_materialize()
        self._buffers = buffers
        self._num_layers, self._num_slots = state.shape[:2]
        pools = (state, buffers.k_ring, buffers.u_ring, buffers.g_ring)
        assert all(t.shape[:2] == (self._num_layers, self._num_slots) for t in pools), [
            t.shape for t in pools
        ]
        self._layer_pools = pools
        # Flattening [layers, slots] into one pool dim needs no gap between layers.
        self._flat_pools = (
            tuple(t.view(-1, *t.shape[2:]) for t in pools)
            if all(t.is_contiguous() for t in pools)
            else None
        )
        self._plan_layers = self._num_layers if self._flat_pools is not None else 1
        self._plan: Optional[_TrackPlan] = None

    @property
    def single_launch(self) -> bool:
        return self._flat_pools is not None

    def materialize(
        self,
        *,
        slots: torch.Tensor,
        num_accept_tokens: torch.Tensor,
        track_slots: torch.Tensor,
        steps_to_track: torch.Tensor,
    ) -> None:
        """``slots`` and ``track_slots`` are physical ids (``<= 0`` skips the row),
        ``steps_to_track`` the verify step whose state is tracked (``-1`` skips).
        Must follow ``commit_gdn_ucache_cursors`` of the same verify and precede
        the next verify."""
        num_rows = slots.shape[0]
        if num_rows == 0:
            return
        plan = self._plan_for(num_rows)
        num_entries = num_rows * self._plan_layers
        _plan_track_kernel[(self._plan_layers,)](
            slots,
            num_accept_tokens,
            track_slots,
            steps_to_track,
            self._buffers.hist_len,
            self._buffers.cache_base,
            plan.src,
            plan.dst,
            plan.base,
            plan.count,
            plan.active,
            plan.num_active,
            num_rows,
            self._num_slots,
            BLOCK=triton.next_power_of_2(plan.rows),
        )
        args = dict(
            src_slots=plan.src[:num_entries],
            dst_slots=plan.dst[:num_entries],
            cache_base=plan.base[:num_entries],
            count=plan.count[:num_entries],
            active_request_indices=plan.active[:num_entries],
            num_active=plan.num_active,
        )
        if self._flat_pools is not None:
            self._launch(self._flat_pools, args)
        else:
            for layer in range(self._num_layers):
                self._launch(tuple(t[layer] for t in self._layer_pools), args)

    def warm_up(self, *, max_batch_size: int) -> None:
        """Allocates the plan for ``max_batch_size`` rows and compiles both kernels
        on an all-padding batch, so CUDA graph capture compiles and allocates
        nothing."""
        self._plan_for(max_batch_size)
        device = self._buffers.hist_len.device
        padded = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
        self.materialize(
            slots=padded,
            num_accept_tokens=padded,
            track_slots=padded,
            steps_to_track=padded,
        )

    def _launch(self, pools: tuple, args: dict) -> None:
        state, k_ring, u_ring, g_ring = pools
        self._materialize(state, k_cache=k_ring, u_cache=u_ring, g_cache=g_ring, **args)

    def _plan_for(self, num_rows: int) -> _TrackPlan:
        plan = self._plan
        if plan is not None and plan.rows >= num_rows:
            return plan
        _assert_not_capturing("the track materialize plan (warm it up first)")
        device = self._buffers.hist_len.device
        entries = num_rows * self._plan_layers

        def buf(n):
            return torch.empty(n, dtype=torch.int32, device=device)

        self._plan = _TrackPlan(
            rows=num_rows,
            src=buf(entries),
            dst=buf(entries),
            base=buf(entries),
            count=buf(entries),
            active=buf(entries),
            num_active=buf(1),
        )
        return self._plan


def _load_flush_kernel(device: torch.device) -> Callable:
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 7)):
        raise RuntimeError(
            f"GDN u-cache verify needs SM100 or SM107, got SM{capability[0]}{capability[1]}"
        )
    if capability == (10, 7):
        try:
            from flashinfer.gdn_kernels.gdn_replay_mtp_umma_rubin import (
                gated_delta_rule_mtp_ucache_flush_umma_rubin,
            )

            return gated_delta_rule_mtp_ucache_flush_umma_rubin
        except ImportError:
            # FlashInfer 0.7.3 ships only the SM100 build, which also runs on SM107
            # (slower); the SM107 build arrives with flashinfer-ai/flashinfer#6791.
            logger.warning(
                "GDN u-cache verify: FlashInfer has no SM107 u-cache build; "
                "using the SM100 build. Upgrade FlashInfer for the tuned kernel."
            )
    try:
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
            gated_delta_rule_mtp_ucache_flush_umma,
        )
    except ImportError as e:
        raise RuntimeError(
            "--enable-gdn-replayssm-ucache needs FlashInfer >= 0.7.3 "
            "(flashinfer.gdn_kernels.gdn_replay_mtp_umma)."
        ) from e
    return gated_delta_rule_mtp_ucache_flush_umma


class GdnUcacheVerifier:
    """Runs the per-layer verify launches of one target model.

    Per forward, the first GDN layer gathers the cursors, zeroes the padded rows of
    the shared output buffer and refreshes every layer's fp32 gate parameters; each
    layer then costs one q/k/v split plus the FlashInfer launch."""

    def __init__(self, *, device: torch.device, first_layer_id: int):
        self._flush_kernel = _load_flush_kernel(device)
        self._device = device
        self._first_layer_id = first_layer_id
        # layer_id -> fp32 (A_log, dt_bias) passed to FlashInfer, allocated eagerly.
        self._gate_params: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # (bf16 source, fp32 destination) pairs refreshed by one launch per forward.
        self._gate_refresh: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._gate_refresh_tables: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._rows: Optional[GdnUcacheRows] = None
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
    ) -> torch.Tensor:
        """Returns the verify output in ``value``'s shape; padded rows are zero."""
        num_v_heads, head_v_dim = value.shape[-2], value.shape[-1]
        num_k_heads, head_k_dim = key.shape[-2], key.shape[-1]
        batch = value.numel() // (num_draft_tokens * num_v_heads * head_v_dim)
        token_shape = (batch, num_draft_tokens)
        output = self._output_buffer(
            batch=batch, row_shape=(num_draft_tokens, num_v_heads, head_v_dim)
        )
        if layer_id == self._first_layer_id:
            self._rows = gather_gdn_ucache_rows(
                slots=cache_indices[:batch], buffers=buffers
            )
            self._zero_padded_rows(output)
            self._refresh_gate_params()
        rows = self._rows
        assert rows is not None and rows.slots.shape[0] == batch
        A_log_fp32, dt_bias_fp32 = self._fp32_gate_params(layer_id, A_log, dt_bias)
        q, k, v = self._contiguous_qkv(query, key, value)
        self._flush_kernel(
            A_log=A_log_fp32,
            a=a.reshape(*token_shape, num_v_heads),
            dt_bias=dt_bias_fp32,
            q=q.view(*token_shape, num_k_heads, head_k_dim),
            k=k.view(*token_shape, num_k_heads, head_k_dim),
            v=v.view(*token_shape, num_v_heads, head_v_dim),
            b=b.reshape(*token_shape, num_v_heads),
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
    ) -> None:
        """Compiles the kernels, allocates the shared buffers and builds
        FlashInfer's per-batch-size helper tensors eagerly, so CUDA graph capture
        creates none of them."""
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

        qk = zeros(*tokens, num_k_heads, head_k_dim)
        v = zeros(*tokens, num_v_heads, head_v_dim)
        gate = zeros(*tokens, num_v_heads)
        gate_param = zeros(num_v_heads, dtype=torch.float32)
        cursor = zeros(max_batch_size, dtype=torch.int32)
        padded = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
        for batch in range(1, max_batch_size + 1):
            self._flush_kernel(
                A_log=gate_param,
                a=gate[:batch],
                dt_bias=gate_param,
                q=qk[:batch],
                k=qk[:batch],
                v=v[:batch],
                b=gate[:batch],
                initial_state_source=state,
                initial_state_indices=padded[:batch],
                k_cache=k_ring,
                u_cache=u_ring,
                g_cache=g_ring,
                hist_len=cursor[:batch],
                cache_base=cursor[:batch],
                flush_min=flush_min,
                restart_hist_on_flush=False,
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


def _assert_not_capturing(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"GDN u-cache verify: {what} must be prepared before CUDA graph capture."
        )


def _fp32_view_or_buffer(param: torch.Tensor) -> torch.Tensor:
    if param.dtype == torch.float32 and param.is_contiguous():
        return param.detach()
    return torch.empty(param.shape, dtype=torch.float32, device=param.device)
