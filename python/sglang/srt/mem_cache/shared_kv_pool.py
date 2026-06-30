"""SharedKVPool — one physical byte buffer shared by ≥2 sub-pools.

In short:

* One `uint8` buffer (`SharedKVPool._raw`) is split dynamically between
  sub-pools by `MultiEndedAllocator`s growing from opposite ends; `free` does
  eager compaction so each pool's allocated byte range stays hole-free.
* Per-slot layout is **slot/envelope-major** — a slot holds its data for all of
  that pool's layers in one contiguous byte envelope — so freeing a slot vacates
  a contiguous region the peer can grow into.
* Everything above the allocator stores **virtual** slot IDs (immutable for the
  slot's lifetime); the allocator keeps per-sub-pool `virtual_to_physical` /
  `physical_to_virtual` tables. On compaction `p_src → p_dst` only those two
  tables change — no reference rewriting. There is **no** `relocation_log` /
  `SlotBacktrack` / binder machinery.

Wires up the hybrid-Mamba family (`init_shared_mamba_pools`) and hybrid-SWA
family (`SharedSWAKVPool` / `init_shared_swa_pools`). N>2 sub-pools / DSV4 and
the disagg/spec bits are not implemented here yet.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import triton
from torch.profiler import record_function

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    move_kv_cache_native,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.triton_ops.cache_move import store_cache_4d_kernel
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _prod(iterable) -> int:
    out = 1
    for x in iterable:
        out *= int(x)
    return out


def _store_dtype_for(kv_cache_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return torch.uint8
    return kv_cache_dtype


# ---------------------------------------------------------------------------
# Sub-pool specs (pure per-slot layout math; no allocator/binder state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):
    """Abstract per-slot layout of one sub-pool in a `SharedKVPool`."""

    name: str
    layer_num: int
    grow_direction: str  # "up" or "down"

    def __post_init__(self):
        assert self.grow_direction in (
            "up",
            "down",
        ), f"grow_direction must be 'up' or 'down'; got {self.grow_direction!r}"
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"

    @abstractmethod
    def entry_bytes(self) -> int:
        """Bytes consumed by one slot across all `layer_num` layers of this pool."""
        raise NotImplementedError

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """The storage dtype of this sub-pool's KV data.

        Used by ``MultiEndedAllocator`` to pass to its base init's
        ``dtype`` field (informational; matches the upstream allocator's
        ``self.dtype`` attribute). Subclasses with a single dtype return
        it directly; subclasses with multiple dtypes (e.g., Mamba's
        ``conv_dtype`` and ``temporal_dtype``) return the most
        representative one — by convention the dtype of the dominant
        state buffer (conv for Mamba) — and document the choice.
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` may differ from
    `head_dim` (matches `MHATokenToKVPool`); falls back to `head_dim` if None."""

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert (
            self.v_head_dim > 0
        ), f"v_head_dim must be positive; got {self.v_head_dim}"

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

    # Page-major layer-major byte math.
    # Within each page's ``page_size * entry_bytes`` block, K and V for each
    # layer are grouped: page bytes =
    #   [L0_K * page_size | L0_V * page_size | L1_K * page_size | ...].
    # Across pages the layout stays envelope (page_bytes per page, preserving
    # byte-frontier coordination between sub-pools).
    #
    # At ``page_size == 1`` this collapses to today's slot-major envelope —
    # one page is one slot and the within-page block IS the per-slot
    # ``[L0_K | L0_V | L1_K | L1_V | ...]`` envelope. Byte addresses are
    # therefore byte-identical to slot-based envelope at ps=1.

    def page_bytes(self, page_size: int) -> int:
        """Bytes per page (``page_size`` slots)."""
        return page_size * self.entry_bytes()

    def layer_k_offset_in_page(self, layer_id: int, page_size: int) -> int:
        """Byte offset (within one page) of layer ``layer_id``'s K block.

        Layer L's K block starts at ``L * page_size * (k_row + v_row)``. At
        ps=1 this is ``L * (k_row + v_row)`` — same as today's envelope.
        """
        return layer_id * page_size * (self.k_row_bytes() + self.v_row_bytes())

    def layer_v_offset_in_page(self, layer_id: int, page_size: int) -> int:
        """Byte offset (within one page) of layer ``layer_id``'s V block.
        Immediately after that layer's K block."""
        return (
            self.layer_k_offset_in_page(layer_id, page_size)
            + page_size * self.k_row_bytes()
        )

    def get_dtype(self) -> torch.dtype:
        """Storage dtype of the MHA K/V buffers — single dtype shared by
        both K and V (matches ``MHATokenToKVPool``'s contract)."""
        return self.store_dtype


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool. `layer_num` = number of
    Mamba layers whose state is held (≡ `MambaPool` `num_mamba_layers`)."""

    conv_state_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, "conv_state_shapes must be non-empty"

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def entry_bytes(self) -> int:
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total

    def get_dtype(self) -> torch.dtype:
        """Mamba has two distinct dtypes: ``conv_dtype`` for conv state
        buffers and ``temporal_dtype`` for the SSM temporal state. We
        return ``conv_dtype`` as the representative — it's the dominant
        state (one tensor per ``conv_state_shapes`` entry; temporal is
        single) and matches the convention of ``MambaPool.dtype`` in
        upstream. The temporal dtype is separately accessible via
        ``temporal_dtype`` for callers that need it.
        """
        return self.conv_dtype


# ---------------------------------------------------------------------------
# SharedKVPool — the byte buffer + the strided per-sub-pool views
# ---------------------------------------------------------------------------


class SharedKVPool:
    """One physical `uint8` byte buffer shared by 2 (eventually ≥2) sub-pools.

    Each sub-pool exposes its per-layer K/V or conv/temporal tensors as strided
    views into the raw buffer (envelope layout, anchored at byte 0). Allocators
    coordinate to keep their byte ranges disjoint; this class does not track usage.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpec],
        device: str,
        enable_memory_saver: bool,
        page_size: int = 1,
    ):
        assert page_size >= 1, f"page_size must be >= 1; got {page_size}"
        assert len(sub_pool_specs) == 2, (
            f"SharedKVPool currently supports exactly 2 sub-pools; got "
            f"{len(sub_pool_specs)} (N>2 is not yet implemented)"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == 2, f"sub-pool names must be unique; got {names}"
        directions = sorted(s.grow_direction for s in sub_pool_specs)
        assert directions == ["down", "up"], (
            f"SharedKVPool needs one grow-up and one grow-down sub-pool; "
            f"got {directions}"
        )

        self.device = device
        self.total_bytes = total_bytes
        self.sub_pool_specs = sub_pool_specs
        self._page_size = page_size  # feeds _build_mha_views stride math
        self._specs_by_name: Dict[str, SubPoolSpec] = {
            s.name: s for s in sub_pool_specs
        }

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._raw = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        self._raw.zero_()  # unset slots read as zeros (matches non-shared behavior)

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}
        # MHA views: (k_buffer, v_buffer); Mamba views: (conv_state_list, temporal_state)
        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}

        # Slot-0 padding-write safety: every pool's slot-0 dummy writes land in
        # raw bytes [0, entry_i). Their union is [0, entry_max). Each pool's first
        # allocatable slot index is chosen so its real data starts at ≥ entry_max.
        entry_max = max(s.entry_bytes() for s in sub_pool_specs)

        for spec in sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            min_slot_index = (entry_max + entry_bytes - 1) // entry_bytes  # ceil
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"SharedKVPool: sub-pool {spec.name!r} fits only {max_slots} "
                    f"slots in {total_bytes} bytes, but min_slot_index={min_slot_index} "
                    f"leaves no room for real data. Increase total_bytes."
                )
            anchor = 0  # all anchors are 0 (uniform view construction)
            self._max_slots[spec.name] = max_slots
            self._anchor_bytes[spec.name] = anchor
            self._min_slot_index[spec.name] = min_slot_index
            if isinstance(spec, MHASubPoolSpec):
                self._mha_views[spec.name] = self._build_mha_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
            elif isinstance(spec, MambaSubPoolSpec):
                self._mamba_views[spec.name] = self._build_mamba_views(
                    spec, anchor, max_slots
                )
            else:  # pragma: no cover
                raise TypeError(f"unsupported SubPoolSpec type: {type(spec)}")

        logger.info(
            "[shared-pool] SharedKVPool allocated: total_bytes=%.2f GB (=%d B), "
            "%d sub-pool(s)",
            total_bytes / GB,
            total_bytes,
            len(sub_pool_specs),
        )
        for s in sub_pool_specs:
            logger.info(
                "[shared-pool]   sub-pool %r: kind=%s, layer_num=%d, grow=%s, "
                "entry_bytes=%d, max_slots=%d, min_slot_index=%d (slots [0,%d) reserved)",
                s.name,
                type(s).__name__,
                s.layer_num,
                s.grow_direction,
                s.entry_bytes(),
                self._max_slots[s.name],
                self._min_slot_index[s.name],
                self._min_slot_index[s.name],
            )

    # -- introspection --

    def spec(self, name: str) -> SubPoolSpec:
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MHASubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MambaSubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        return s

    def max_slots(self, name: str) -> int:
        return self._max_slots[name]

    def min_slot_index(self, name: str) -> int:
        return self._min_slot_index[name]

    def anchor_bytes(self, name: str) -> int:
        anchor = self._anchor_bytes[name]
        assert anchor == 0, f"current design assumes all anchors are 0; got {anchor}"
        return anchor

    def mha_views_for(self, name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._mha_views[name]

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    # -- view construction (envelope layout) --

    def _build_mha_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K/V views over the shared buffer in the page-major envelope
        layout. The strided-view math lives in the layout module; the sub-pool's
        ``anchor_bytes`` places its region inside the shared buffer."""
        return build_page_major_mha_views(
            self._raw,
            layer_num=spec.layer_num,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=max_slots // page_size,
            anchor_bytes=anchor_bytes,
        )

    def _build_mamba_views(
        self, spec: MambaSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-slot envelope: [conv[0] rows × layers][conv[1] rows × layers]...
        [temporal rows × layers]. Each returned view has shape
        (num_layers, max_slots, *inner_shape) — matches `MambaPool.State.conv[i]`
        / `.temporal` conventions.
        """
        return build_page_major_mamba_views(
            self._raw,
            layer_num=spec.layer_num,
            conv_state_shapes=spec.conv_state_shapes,
            conv_dtype=spec.conv_dtype,
            temporal_state_shape=spec.temporal_state_shape,
            temporal_dtype=spec.temporal_dtype,
            max_slots=max_slots,
            anchor_bytes=anchor_bytes,
        )


# ---------------------------------------------------------------------------
# SharedMHATokenToKVPool — MHA pool whose buffers are views into a SharedKVPool
# ---------------------------------------------------------------------------


class SharedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool whose `k_buffer` / `v_buffer` are strided views into a
    `SharedKVPool`. Buffer lifetime is owned by the SharedKVPool;
    relocation uses the native move (strided views break the tiled Triton kernel
    that assumes stride == row bytes).

    `set_kv_buffer` receives PHYSICAL slot ids (the full-physical
    `KVWriteLoc.full_loc` / swa-physical `swa_out_cache_loc` resolved in the
    attention metadata) and writes them directly — the pool never translates.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        spec = shared_buffer.mha_spec(sub_pool_name)
        k_buffer, v_buffer = shared_buffer.mha_views_for(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)

        self._shared_buffer = shared_buffer
        self._sub_pool_name = sub_pool_name
        self._k_views = k_buffer
        self._v_views = v_buffer
        # page_size for the 4-D view stride math in `_create_buffers` /
        # `move_kv_cache` (the K/V strided views are TOKEN-granular).
        self._page_size = page_size

        super().__init__(
            size=max_slots - 1,  # -1 for reserved slot 0
            page_size=page_size,
            dtype=spec.store_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=spec.layer_num,
            device=shared_buffer.device,
            enable_memory_saver=False,  # buffer owned by SharedKVPool
            v_head_dim=spec.v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,  # strided views — force native move
        )

    # -- buffer lifecycle overrides --

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views
        # data_ptrs / data_strides are populated for any external code that
        # inspects them; we force the native move path so they are not consumed.
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in (self.k_buffer + self.v_buffer)],
            device=self.device,
        )

    def _clear_buffers(self):
        # Lifetime owned by SharedKVPool; do not delete the views.
        pass

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # tgt_loc / src_loc are PHYSICAL slot ids (passed by the allocator's
        # _compact_pending). Force the native move (strided views).
        if tgt_loc.numel() == 0:
            return
        # Pass page_size so move_kv_cache_native takes the 4-D
        # branch and splits token ids into (page_id, tok_in_page) when
        # operating on layer-major views.
        with record_function("SharedMHA.move_kv_cache"):
            move_kv_cache_native(
                self.k_buffer,
                self.v_buffer,
                tgt_loc,
                src_loc,
                page_size=self._page_size,
            )

    def get_kv_size_bytes(self):
        # The shared buffer's total size is logged by SharedKVPool once;
        # per-sub-pool accounting would double-count.
        return 0, 0

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        # `dcp_kv_mask` is forwarded unconditionally by the parent
        # `HybridLinearKVPool.set_kv_buffer` (default None). Decode context
        # parallelism is not supported under the shared KV pool yet, so fail
        # loud rather than silently ignore a real mask (which would write
        # tokens this rank does not own).
        assert dcp_kv_mask is None, (
            "SharedMHATokenToKVPool.set_kv_buffer: decode context parallel "
            "(dcp_kv_mask) is not supported with --enable-shared-kv-pool."
        )
        # Collapsed wrapper chain: this method does all the work in ONE
        # frame and calls `store_cache_4d_kernel` directly, bypassing the
        # Python launcher for the hot path. The standalone `store_cache_4d`
        # wrapper in `utils.py` stays for any non-shared-pool callers (and
        # as a validated reference impl) — it is deliberately bypassed here
        # for latency, since the contract is guaranteed by `_build_mha_views`.
        #
        # Always-bypass-super() write path.
        # Why bypass `super().set_kv_buffer(...)`? The parent
        # `MHATokenToKVPool.set_kv_buffer` routes through
        # `_set_kv_buffer_impl` which calls `k_cache.view(-1, row_dim)`
        # before invoking the `store_cache` Triton kernel. PyTorch's
        # `view()` requires `stride[i] == shape[i+1] * stride[i+1]` for the
        # dimensions being merged. Our 4-D layer-major view
        # `(num_pages, page_size, head_num, head_dim)` has
        # `stride[0] = page_bytes/itemsize` which doesn't satisfy that
        # merge rule at page_size > 1; we use the same code path at
        # page_size == 1 for consistency.
        with record_function("SharedMHA.set_kv_buffer"):
            # `loc` is PHYSICAL token ids: the write location is fully resolved in
            # the attention metadata (`KVWriteLoc.full_loc` / the SWA pool's
            # `swa_out_cache_loc`) before reaching here, so the pool does NO v2p
            # translate and holds no allocator / location state.

            # Step 1: replicate the parent's dtype-cast logic inline.
            if cache_k.dtype != self.dtype:
                if k_scale is not None:
                    cache_k.div_(k_scale)
                if v_scale is not None:
                    cache_v.div_(v_scale)
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            # Step 3: write into the 4-D layer-major view via a single-launch
            # Triton `store_cache_4d_kernel` (called directly; the Python
            # wrapper `store_cache_4d` in `utils.py` is bypassed for latency
            # since its validation asserts are guaranteed by `_build_mha_views`).
            layer_id = (
                layer.layer_id if layer_id_override is None else layer_id_override
            ) - self.start_layer
            k_view = self.k_buffer[layer_id]
            v_view = self.v_buffer[layer_id]
            ps = self._page_size
            # Inline store_cache_4d's body (kernel launch only). The contract —
            # 4-D K/V views with stride(-1)==1 and stride(-2)==head_dim, 3-D
            # cache_k/cache_v with matching batch dim — is guaranteed by
            # `_build_mha_views` (K/V) and the model forward (cache_k/cache_v).
            # Skipping the wrapper's validation asserts saves ~10-30 µs/call.
            N = loc.numel()
            if N == 0:
                return
            head_num = k_view.shape[2]
            head_dim = k_view.shape[3]
            v_head_dim = v_view.shape[3]
            K_ROW_DIM = head_num * head_dim
            V_ROW_DIM = head_num * v_head_dim
            BLOCK = 128
            row_dim_max = K_ROW_DIM if K_ROW_DIM > V_ROW_DIM else V_ROW_DIM
            store_cache_4d_kernel[(N, triton.cdiv(row_dim_max, BLOCK), 2)](
                k_view,
                v_view,
                cache_k,
                cache_v,
                loc,
                k_view.stride(0),
                k_view.stride(1),
                v_view.stride(0),
                v_view.stride(1),
                cache_k.stride(0),
                cache_v.stride(0),
                K_ROW_DIM=K_ROW_DIM,
                V_ROW_DIM=V_ROW_DIM,
                PAGE_SIZE=ps,
                BLOCK=BLOCK,
                num_warps=4,
            )


# ---------------------------------------------------------------------------
# SharedMambaPool — Mamba state pool whose buffers are views into a SharedKVPool
# ---------------------------------------------------------------------------


class SharedMambaPool(MambaPool):
    """Mamba state pool for the shared buffer — the VIRTUAL view.

    `conv_state` / `temporal_state` are strided views into a `SharedKVPool`,
    but this pool's public surface is virtual: every index-bearing method
    (`clear_slots`, `copy_from`, `get_cpu_copy`, `load_cpu_copy`) receives
    **virtual** slot ids (the upstream `req.mamba_pool_idx` contract). It does not
    own slot lifecycle and does not know the virtual<->physical mapping — it
    borrows the allocator's `translate` to resolve a virtual id the moment before
    touching the buffer, holding no v2p logic of its own.

    The PHYSICAL view — real slot alloc/free, the free-list/sizing, and the
    bidirectional virtual<->physical mapping — lives in the attached
    `SharedMambaSlotAllocator`. Mirrors upstream's `MambaPool` (state) /
    `MambaSlotAllocator` (slots) split, and the shared KV pool/allocator split
    where the allocator owns `translate_kv_loc`.

    Does NOT call `super().__init__()` (that allocates fresh tensors). Replicates
    the minimal `MambaPool` state against the shared buffer so inherited methods
    work.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedKVPool,
        sub_pool_name: str,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        spec = shared_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)
        conv_views, temporal_view = shared_buffer.mamba_views_for(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)

        self._shared_buffer = shared_buffer
        self._sub_pool_name = sub_pool_name

        # Replicate the state MambaPool.__init__ would have set.
        self._max_size = max_slots - 1  # -1 for reserved slot 0
        self.size = self._max_size
        self.device = shared_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num
        # Note: layer_transfer_counter is owned by HybridReqToTokenPool / MHA pools,
        # NOT MambaPool — don't add it here.
        # GDN/KDA ReplaySSM is not supported under the shared KV pool. Replicate
        # the parent `MambaPool`'s disabled-state attributes so inherited code
        # paths that read them unconditionally (e.g. `HybridReqToTokenPool.alloc`
        # and `MambaRadixCache`, both guarded by `replayssm_write_pos is not
        # None`) see a well-defined "disabled" pool instead of an AttributeError.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False

        assert (
            conv_views[0].shape[0] == self.num_mamba_layers
        ), f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        assert (
            conv_views[0].shape[1] == self._max_size + 1
        ), f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"

        # Optional per-draft-token intermediate buffers — different outer size
        # (spec_state_size+1), so NOT in the shared byte buffer; allocate locally.
        temporal_state_shape = spec.temporal_state_shape
        conv_state_shape = spec.conv_state_shapes
        conv_dtype = spec.conv_dtype
        ssm_dtype = spec.temporal_dtype
        if speculative_num_draft_tokens is not None:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        self.num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=shared_buffer.device,
                )
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            self.num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            cshape[0],
                            cshape[1],
                        ),
                        dtype=conv_dtype,
                        device=shared_buffer.device,
                    )
                    for cshape in conv_state_shape
                ]
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)

        self.mem_usage = shared_buffer.total_bytes / GB
        logger.info(
            "[shared-pool] SharedMambaPool(%s) wrapped shared buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )

    # Pure PHYSICAL store: this pool holds no v<->physical mapping. The public
    # state ops (copy_from / clear_slots / get_cpu_copy / load_cpu_copy) are
    # inherited from MambaPool and operate on PHYSICAL slot ids. Callers resolve
    # virtual->physical via the slot allocator before calling — the
    # scheduler/backend/radix/compaction paths via
    # `SharedHybridReqToTokenPool.translate_mamba_indices`, the HiCache offload
    # path via `HybridLinearKVPool` (which threads the same translate).

    def _copy_from_physical(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        """Physical-slot copy — used by the allocator's `_compact_pending`
        (which already holds physical ids)."""
        MambaPool.copy_from(self, src_index, dst_index)


# ---------------------------------------------------------------------------
# SharedMambaSlotAllocator — the PHYSICAL view (slot lifecycle + v<->p mapping)
# ---------------------------------------------------------------------------


class SharedMambaSlotAllocator:
    """Mamba slot allocator for the shared pool — the PHYSICAL view.

    Owns the physical side: real slot alloc/free, the free-list/sizing, and the
    bidirectional virtual<->physical mapping (the ``translate``). Presents the
    upstream ``MambaSlotAllocator`` interface that ``HybridReqToTokenPool``
    (``alloc``/``free``/``clear``/``available_size``/``alloc_group_*``) and the
    scheduler's invariant checker (``free_slots``/``size``) drive, backed by the
    ``MultiEndedAllocator`` that id-owns the shared mamba sub-pool's slot space.
    Because it owns ``translate`` (mirroring the KV allocator's
    ``translate_kv_loc``), ``SharedMambaPool`` (the VIRTUAL view) needs no v2p
    logic and just borrows it. Same pool(state)/allocator(slots) split upstream
    made when it separated ``MambaPool`` from ``MambaSlotAllocator``.

    ``alloc()`` returns VIRTUAL ids and does NOT clear state: clearing is
    deferred to ``SharedMambaPool.clear_slots`` via the upstream
    ``req.mamba_needs_clear`` path (``_execute_deferred_mamba_cow_and_clear``),
    exactly as the non-shared ``MambaSlotAllocator`` + ``MambaPool.clear_slots``
    pair works.
    """

    def __init__(self, mea, max_size: int, device: str):
        self._multi_ended_allocator = mea  # MultiEndedAllocator — id-owner + v2p table
        self._max_size = max_size  # slot capacity (excludes reserved slot 0)
        self._device = device
        self._alloc_iter = None  # active alloc_group batch iterator

    # -- translation (owns the v<->p mapping; mirror of allocator.translate_kv_loc) --

    def translate(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """VIRTUAL mamba slot ids -> PHYSICAL slot ids. The mamba sub-allocator
        is page_size==1 (1 slot == 1 page == 1 token), so this is a direct v2p
        gather (no page math)."""
        return self._multi_ended_allocator.virtual_to_physical[virtual_ids]

    @property
    def virtual_to_physical(self) -> torch.Tensor:
        return self._multi_ended_allocator.virtual_to_physical

    # -- sizing / free-list (leak/invariant checker + alloc planner) --

    @property
    def size(self) -> int:
        return self._max_size

    def available_size(self) -> int:
        """Slot-conservation free count (``max - allocated``) — the leak-check
        view (``available + evictable + protected + session_held == size``).
        NOT the planner value; use ``schedulable_available_size`` for that."""
        return self._max_size - self._multi_ended_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        """Byte-coordinated free count — the ``>= N => alloc(N) succeeds``
        contract used by ``common.alloc_req_slots``. Uses the MEA's
        realizable-with-compaction view (peer drainable holes credited), so a
        mamba-state admission is not starved when the shared gap is choked by
        the full peer's uncompacted holes — ``alloc`` flushes the peer before
        extending, so the credited capacity is obtainable."""
        return self._multi_ended_allocator.schedulable_available_size()

    @property
    def free_slots(self) -> torch.Tensor:
        """Physical free-list (watermark-derived) read by the scheduler's
        invariant checker. Mamba sub-allocator is page_size==1, so pages==slots;
        assert it so a future paged-mamba change is caught, not silently wrong."""
        a = self._multi_ended_allocator
        assert a.page_size == 1, (
            "SharedMambaSlotAllocator.free_slots assumes page_size==1; got "
            f"{a.page_size}. Mamba state is per-request, orthogonal to paging."
        )
        if a.grow_direction == "up":
            start, end = a.watermark_physical, a.num_pages
        else:
            start, end = a.min_page_index, a.watermark_physical + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.arange(start, end, dtype=torch.int64, device=self._device)

    # -- slot management (delegates to the MultiEndedAllocator) --

    def alloc(self, need_size: int):
        # alloc_group fast path: single-slot draws from the prefetched batch.
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._multi_ended_allocator.alloc(
            need_size
        )  # VIRTUAL ids; clearing deferred

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch (match_prefix amortization); ``alloc(1)`` then
        draws from it. Mirror of ``MambaSlotAllocator.alloc_group_begin``."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._multi_ended_allocator.alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self._multi_ended_allocator.free(torch.cat(remaining))
        self._alloc_iter = None

    def is_slot_allocated(self, slot) -> bool:
        return self._multi_ended_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        return self._multi_ended_allocator.allocator_state_str()


# ---------------------------------------------------------------------------
# SharedHybridReqToTokenPool — HybridReqToTokenPool whose MambaPool is shared
# ---------------------------------------------------------------------------


class SharedHybridReqToTokenPool(HybridReqToTokenPool):
    """`HybridReqToTokenPool` whose `mamba_pool` is a `SharedMambaPool` aliasing a
    shared byte buffer. Everything else (alloc/get_mamba_indices/free_mamba_cache/
    ping-pong) is inherited unchanged — `req.mamba_pool_idx`,
    `req_index_to_mamba_index_mapping` and `TreeNode.mamba_value` now hold VIRTUAL
    mamba ids, which is exactly what they should hold. Adds `translate_mamba_indices`
    for the attention backend's per-batch virtual->physical translation.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedKVPool,
        mamba_sub_pool_name: str,
        size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        self._shared_buffer = shared_buffer
        self._mamba_sub_pool_name = mamba_sub_pool_name
        # mamba_size matches SharedKVPool.max_slots - 1 (reserve slot 0).
        self._shared_mamba_size = shared_buffer.max_slots(mamba_sub_pool_name) - 1
        super().__init__(
            size=size,
            mamba_size=self._shared_mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            enable_overlap_schedule=enable_overlap_schedule,
            start_layer=start_layer,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
    ):
        # `mamba_envelope_layout`, `speculative_eagle_topk`,
        # `enable_linear_replayssm`, and `linear_replayssm_cache_len` are
        # accepted to
        # match the parent `HybridReqToTokenPool._init_mamba_pool` signature but
        # NOT forwarded to `SharedMambaPool`. The shared pool's Mamba conv/
        # temporal state are always envelope-strided VIEWS into the shared
        # buffer (built in `init_shared_mamba_pools`), so the standalone-pool
        # envelope flag does not apply. `speculative_eagle_topk`: in the base
        # `MambaPool` it only sizes the conv-state
        # ALLOCATION (the linear-vs-tree draft-chain shape), whereas the shared
        # pool's conv/temporal state are VIEWS into the shared buffer whose shape
        # is fixed by the `MambaSubPoolSpec` (built in `init_shared_mamba_pools`).
        # So there is nothing to allocate here. (A future shared-pool + eagle
        # spec-decode config would thread it through the sub-pool spec instead.)
        # Parent's contract: `mamba_size` is the source of truth for the mamba
        # pool's slot count. Under the shared pool that source of truth is
        # `SharedKVPool.max_slots("mamba") - 1` (= self._shared_mamba_size,
        # which __init__ passed as `mamba_size`). Re-assert the equality so a
        # future signature drift in the parent surfaces here, not later.
        assert mamba_size == self._shared_mamba_size, (
            f"SharedHybridReqToTokenPool._init_mamba_pool: mamba_size={mamba_size} "
            f"!= shared_buffer.max_slots({self._mamba_sub_pool_name!r}) - 1 "
            f"= {self._shared_mamba_size}"
        )
        # `cache_params` is consumed indirectly via the SharedKVPool's
        # MambaSubPoolSpec (built by init_shared_mamba_pools from the same
        # cache_params). Sanity-check the layer count matches.
        assert len(cache_params.layers) >= len(mamba_layer_ids), (
            f"cache_params.layers ({len(cache_params.layers)}) cannot supply "
            f"{len(mamba_layer_ids)} mamba layer ids"
        )
        # SharedMambaPool reads conv/temporal shapes from its sub-pool spec
        # (shared_buffer.mamba_spec(mamba_sub_pool_name)).
        self.mamba_pool = SharedMambaPool(
            shared_buffer=self._shared_buffer,
            sub_pool_name=self._mamba_sub_pool_name,
            spec_state_size=mamba_spec_state_size,
            mamba_layer_ids=mamba_layer_ids,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        # `mamba_allocator` (the PHYSICAL view — a `SharedMambaSlotAllocator`
        # wrapping the mamba `MultiEndedAllocator`) is wired in by the factory
        # `init_shared_mamba_pools` AFTER the composite allocator has created and
        # bound that MultiEndedAllocator (it does not exist yet at pool-init
        # time). Set to None here so the attribute exists; nothing reads it
        # before the factory wiring completes.
        self.mamba_allocator = None
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        # `mamba_ckpt_pool` (phase-6 upstream addition): an OPTIONAL int8 mamba
        # checkpoint pool the radix cache uses to hold cached prefix states in
        # int8 instead of the active bf16 pool. The parent sets it via
        # `maybe_init_int8_mamba_checkpoint_pool` (returns None unless the int8
        # checkpoint server-arg is on). The shared pool keeps its mamba states in
        # the shared byte buffer and does NOT use a separate int8 checkpoint pool,
        # so set None — the parent's reset path guards on `is not None`
        # (memory_pool.py: `if self.mamba_ckpt_pool is not None`), so None is the
        # "feature off" state and the radix cache falls back to the normal path.
        # (Enabling int8 checkpoint WITH the shared pool would be a separate
        # feature to thread through `init_shared_mamba_pools`.)
        self.mamba_ckpt_pool = None
        self.device = device
        # Mirror the parent's sizing: indexed by req_pool_idx, so by the
        # req_to_token buffer's first dim — which is `self.size + 1`, NOT
        # `self.size` (ReqToTokenPool reserves index 0 as the padding row;
        # see ReqToTokenPool.__init__'s `_alloc_size = size + 1`). Using
        # `self.size` directly here would under-size the mapping by one row.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    # int64 to match the parent: the alloc-path `index_put` source
                    # `torch.stack(mamba_ping_pong_track_buffers)` is int64 and is
                    # NOT cast (unlike `mamba_index_tensor`, which is cast to
                    # int32), so an int32 destination raises
                    # "Index put requires the source and destination dtypes match"
                    # on the first radix prefill (enable_mamba_extra_buffer is set
                    # by the radix cache, not chunk — hence radix-only).
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def translate_mamba_indices(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """Virtual per-request mamba ids -> physical slot ids. Called once per
        batch by the linear-attention backend's metadata build. The v<->p mapping
        lives in the allocator (the physical view) — delegate to it."""
        return self.mamba_allocator.translate(virtual_ids).to(torch.int32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class SharedPoolBundle(NamedTuple):
    shared_kv_pool: SharedKVPool
    token_to_kv_pool: object  # HybridLinearKVPool
    token_to_kv_pool_allocator: object  # SharedMambaTokenToKVPoolAllocator
    req_to_token_pool: object  # SharedHybridReqToTokenPool


def init_shared_mamba_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    is_draft_worker: bool,
    use_mla_backend: bool,
    mamba_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    mamba2_cache_params,
    model_context_len: int,
    extra_max_context_len: int,
    max_total_num_tokens: int,
    max_mamba_cache_size: int,
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    speculative_num_draft_tokens: Optional[int],
    disable_overlap_schedule: bool,
    need_sort: bool,
    mamba_full_memory_ratio: Optional[float] = None,  # informational only
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
) -> SharedPoolBundle:
    """Build the Mamba-hybrid shared-pool stack: `SharedKVPool` (full + mamba
    sub-pools), `SharedHybridReqToTokenPool` (with its `SharedMambaPool`),
    `SharedMHATokenToKVPool` injected into a `HybridLinearKVPool`, and the
    `SharedMambaTokenToKVPoolAllocator`."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.multi_ended_allocator import (
        SharedMambaTokenToKVPoolAllocator,
    )

    assert not use_mla_backend, "shared KV pool does not support MLA-hybrid-Mamba yet"
    # The full sub-pool is page-aware (via `MultiEndedAllocator(page_size=...)`);
    # the mamba sub-pool stays page=1 because the Mamba state is per-request,
    # orthogonal to per-token paging.
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # Layout convention: the full-attn KV sub-pool sits at the HIGH-byte end
    # (grow-down, watermark retreats toward the gap) and the peer (mamba)
    # sits at the LOW-byte end (grow-up, just above the slot-0 dummy sink).
    # The two pools grow toward each other through the shared middle gap. All
    # frontier/available_size math is direction-agnostic, so the assignment is
    # a convention, not a correctness constraint (see MultiEndedAllocator).
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
    )
    cp = mamba2_cache_params
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=len(mamba_layer_ids),
        conv_state_shapes=tuple(tuple(int(x) for x in s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
        temporal_dtype=cp.dtype.temporal,
        grow_direction="up",
    )
    total_bytes = (
        max_total_num_tokens * full_spec.entry_bytes()
        + max_mamba_cache_size * mamba_spec.entry_bytes()
    )
    shared_pool = SharedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    req_to_token_pool = SharedHybridReqToTokenPool(
        shared_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        # Mirror model_runner_kv_cache_mixin._init_pools: the parent's
        # `mamba_spec_state_size` is `max_num_reqs` — it sizes the spec-decode
        # intermediate-state buffers' outer dimension (one slot per concurrent
        # request).
        mamba_spec_state_size=max_num_reqs,
        max_context_len=model_context_len + extra_max_context_len,
        device=device,
        enable_memory_saver=enable_memory_saver,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layer_ids,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        enable_overlap_schedule=not disable_overlap_schedule,
        start_layer=start_layer,
    )
    shared_full_kv_pool = SharedMHATokenToKVPool(
        shared_buffer=shared_pool,
        sub_pool_name="full",
        page_size=page_size,
        start_layer=start_layer,
        end_layer=end_layer,
    )
    full_attn_layer_ids_for_pool = (
        [0] if is_draft_worker else list(full_attention_layer_ids)
    )
    token_to_kv_pool = HybridLinearKVPool(
        page_size=page_size,
        size=max_total_num_tokens,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        full_attention_layer_ids=full_attn_layer_ids_for_pool,
        device=device,
        mamba_pool=req_to_token_pool.mamba_pool,
        enable_memory_saver=enable_memory_saver,
        use_mla=use_mla_backend,
        start_layer=start_layer,
        full_kv_pool=shared_full_kv_pool,
    )
    allocator = SharedMambaTokenToKVPoolAllocator(
        shared_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
    )

    # Build the mamba slot allocator (PHYSICAL view). The composite above created
    # the mamba `MultiEndedAllocator` (`allocator.mamba_allocator`); it drives
    # compaction directly via the pool's `_copy_from_physical` (a kvcache ref).
    # Wrap that MEA in a `SharedMambaSlotAllocator` that presents the
    # `MambaSlotAllocator` interface (alloc/free/clear/sizing/group) the inherited
    # `HybridReqToTokenPool` + scheduler drive, and owns the virtual->physical
    # `translate`. `_shared_mamba_size` excludes the reserved slot 0.
    mamba_slot_allocator = SharedMambaSlotAllocator(
        allocator.mamba_allocator,
        max_size=req_to_token_pool._shared_mamba_size,
        device=device,
    )
    # The mamba pool is a pure PHYSICAL store: it holds no v<->physical mapping.
    # `translate` lives ONLY here, and callers resolve virtual->physical before
    # touching the pool — the scheduler/backend/radix/compaction paths via
    # `req_to_token_pool.translate_mamba_indices` (which delegates to this
    # allocator), and the HiCache offload path via the HybridLinearKVPool's
    # `_mamba_translate` hook wired below.
    req_to_token_pool.mamba_allocator = mamba_slot_allocator
    token_to_kv_pool._mamba_translate = mamba_slot_allocator.translate

    logger.info(
        "[shared-pool] ============================================================"
    )
    logger.info("[shared-pool] SHARED KV POOL ENABLED -- path=Mamba hybrid")
    logger.info(
        "[shared-pool]   full_layers=%d, mamba_layers=%d, head_num=%d, head_dim=%d, "
        "page_size=%d, is_draft_worker=%s",
        len(full_attention_layer_ids),
        len(mamba_layer_ids),
        head_num,
        head_dim,
        page_size,
        is_draft_worker,
    )
    logger.info(
        "[shared-pool]   total_bytes=%d, max_total_num_tokens=%d, max_mamba_cache_size=%d, "
        "max_num_reqs=%d, speculative_num_draft_tokens=%s",
        total_bytes,
        max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        speculative_num_draft_tokens,
    )
    if mamba_full_memory_ratio is not None:
        logger.info(
            "[shared-pool]   mamba_full_memory_ratio=%s governs the total budget only, "
            "not the runtime split.",
            mamba_full_memory_ratio,
        )
    logger.info(
        "[shared-pool] ============================================================"
    )
    return SharedPoolBundle(
        shared_kv_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )


# ---------------------------------------------------------------------------
# SharedSWAKVPool — hybrid SWA on the shared byte buffer
# ---------------------------------------------------------------------------


class SharedSWAKVPool(SWAKVPool):
    """Shared-buffer replacement for `SWAKVPool`.

    Composes two `SharedMHATokenToKVPool` instances (full + swa) that alias
    the same physical byte buffer. Exposes the same interface as `SWAKVPool`
    so downstream attention/kernel code is unchanged.

    Inherits from `SWAKVPool` purely for the typing/contract relationship —
    `isinstance(kvcache, SWAKVPool)` (and `BaseSWAKVPool`) is checked across
    attention backends, disagg, models/utils. We do NOT call the parent
    `__init__`: it would build static-partition `MHATokenToKVPool` instances,
    which is exactly what the shared pool replaces. The attribute layout the
    parent sets is replicated here against the shared buffer.

    Rather than maintaining an explicit `full_to_swa_index_mapping` tensor,
    this architecture exposes `translate_loc_from_full_to_swa` directly
    through the swa sub-allocator's
    `virtual_to_physical` table — the per-sub-pool v2p IS the mapping.
    `register_mapping(...)` becomes a no-op (the API surface is kept for
    `BaseSWAKVPool` ABC compatibility).
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedKVPool,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_memory_saver: bool = False,
    ):
        # NOTE: do NOT call `super().__init__(...)`. The SWAKVPool body would
        # allocate two static-partition MHA pools; we replace those with views
        # into the shared buffer here.
        self.shared_buffer = shared_buffer
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer if start_layer is not None else 0
        # Propagate page_size through to the inner SharedMHATokenToKVPool
        # views (which translate virtual TOKEN ids → physical TOKEN ids via
        # page math when page_size > 1).
        self.page_size = page_size
        self.layer_transfer_counter = None

        # The parent class exposes `size` / `size_swa` as plain attributes
        # (set in its __init__). Match that contract — these values are
        # constants of the SharedKVPool, fixed at allocation time.
        self.size = shared_buffer.max_slots("full") - 1
        self.size_swa = shared_buffer.max_slots("swa") - 1

        full_spec = shared_buffer.mha_spec("full")
        swa_spec = shared_buffer.mha_spec("swa")
        # `dtype` is read from MHASubPoolSpec.store_dtype; both sub-pools share
        # the same store_dtype in the standard configurations we support
        # (asymmetric store_dtype across full/swa is not a supported case).
        assert full_spec.store_dtype == swa_spec.store_dtype, (
            "SharedSWAKVPool: full and swa sub-pools must share store_dtype; got "
            f"full={full_spec.store_dtype}, swa={swa_spec.store_dtype}"
        )
        self.dtype = full_spec.store_dtype
        self.head_num = full_spec.head_num
        self.head_dim = full_spec.head_dim
        self.device = shared_buffer.device

        self.full_kv_pool = SharedMHATokenToKVPool(
            shared_buffer=shared_buffer,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.swa_kv_pool = SharedMHATokenToKVPool(
            shared_buffer=shared_buffer,
            sub_pool_name="swa",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # for disagg with nvlink — currently disabled in shared-pool, but keep
        # the attributes present so any caller reading them doesn't AttributeError.
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        # {global_layer_id: (per-pool index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (idx, False)
        for idx, gid in enumerate(swa_attention_layer_ids):
            self.layers_mapping[gid] = (idx, True)

        # `full_to_swa_index_mapping` is the "is the non-shared SWA mapping
        # registered?" signal in `SWAKVPool.set_kv_buffer` /
        # `translate_loc_from_full_to_swa`. Under shared mode we leave it
        # `None` and provide our own overrides that consult the swa
        # sub-allocator's v2p table instead.
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        # The shared buffer's total size is logged by SharedKVPool — set a
        # cosmetic 0 here to avoid double-counting in any aggregator.
        self.mem_usage = 0.0

        # Allocator handles wired in via `attach_allocators` from the composite
        # allocator's __init__.
        self._full_allocator = None
        self._swa_allocator = None

        logger.info(
            "[shared-pool] SharedSWAKVPool wrapped shared buffer: "
            "full_layers=%d (max_slots=%d), swa_layers=%d (max_slots=%d), "
            "head_num=%d, head_dim=%d",
            self.full_layer_nums,
            shared_buffer.max_slots("full"),
            self.swa_layer_nums,
            shared_buffer.max_slots("swa"),
            self.head_num,
            self.head_dim,
        )

    # -- allocator wiring --

    def attach_allocators(self, *, full_allocator, swa_allocator) -> None:
        """Wire the two `MultiEndedAllocator`s whose `virtual_to_physical`
        tables this pool uses to translate slot ids."""
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator

    # -- BaseSWAKVPool ABC surface --

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        # No-op in shared mode (allocator's swa-side v2p IS the mapping). Keep
        # `full_to_swa_index_mapping` None so the parent's `set_kv_buffer`
        # dispatch routes through our overrides.
        return

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        """Virtual token ids -> swa-physical token ids (int32).

        Differs from non-shared ``SWAKVPool.translate_loc_from_full_to_swa``
        in INPUT semantics (virtual, not full-physical), but the OUTPUT is
        the same swa-physical-token-id int32 contract the downstream
        consumers expect.

        For ``page_size == 1``: direct v2p lookup (the v2p table is
        slot-granular == token-granular).
        For ``page_size > 1``: page math —
        ``virt_pages = kv_indices // page_size``,
        ``offsets = kv_indices % page_size``,
        ``swa_phys_pages = swa.v2p_page[virt_pages]``,
        result ``= swa_phys_pages * page_size + offsets``.
        Mirrors ``SharedSWATokenToKVPoolAllocator.translate_loc_from_full_to_swa``.
        """
        assert self._swa_allocator is not None, (
            "SharedSWAKVPool.translate_loc_from_full_to_swa called before "
            "attach_allocators"
        )
        ps = self._swa_allocator.page_size
        if ps == 1:
            return self._swa_allocator.virtual_to_physical[kv_indices].to(torch.int32)
        virt_pages = kv_indices // ps
        offsets = kv_indices % ps
        swa_phys_pages = self._swa_allocator.virtual_to_physical[virt_pages]
        return (swa_phys_pages * ps + offsets).to(torch.int32)

    def get_state_buf_infos(self):
        return self.swa_kv_pool.get_contiguous_buf_infos()

    # -- size/info --

    def get_kv_size_bytes(self):
        # The shared buffer's bytes are logged by SharedKVPool; don't
        # double-count by returning per-side sizes here.
        return 0, 0

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    # -- buffer accessors (verbatim from SWAKVPool, but without _wait_for_layer
    # double-counting — counter wait is delegated to the inner SharedMHATokenToKVPool
    # via register_layer_transfer_counter) --

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_kv_buffer(pool_layer_id)

    # -- kv writing --

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        """Route to the right sub-pool, mirroring `SWAKVPool.set_kv_buffer`.

        `loc_info` bundles the full (virtual) write loc and the pre-translated
        SWA write loc (`forward_metadata.swa_out_cache_loc`), produced once per
        forward by the attention backend — eager via `init_forward_metadata`,
        cuda graph via `cuda_graph_swa_out_cache_loc` (refilled at replay). SWA
        layers write the already-swa-physical `swa_loc` directly; full layers
        write the full-physical `full_loc` carried in the write metadata. Both
        are PHYSICAL — the pool never translates."""
        _, swa_loc, full_loc = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            # `swa_loc` is ALREADY swa-physical (the backend's `swa_out_cache_loc`
            # rail); the pool writes physical locs directly. Routed through the
            # SharedMHATokenToKVPool override (NOT the grandparent
            # `MHATokenToKVPool`) because its 4-D LAYER_MAJOR view can't take the
            # parent's `k_cache.view(-1, row_dim)`.
            assert swa_loc is not None, (
                "SharedSWAKVPool.set_kv_buffer: SWA layer received no swa_loc; the "
                "attention backend must bundle forward_metadata.swa_out_cache_loc."
            )
            self.swa_kv_pool.set_kv_buffer(
                None,
                swa_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=pool_layer_id,
            )
            return
        # Full layer. The full-PHYSICAL write loc is carried in the write metadata
        # (`KVWriteLoc.full_loc`, from the attention backend's
        # `ForwardMetadata.out_cache_loc_full_physical` — translated once per
        # forward and tombstone-clamped by `translate_kv_loc`). The shared pool
        # always precomputes it (eager + cuda-graph capture), so it must be present.
        assert full_loc is not None, (
            "SharedSWAKVPool.set_kv_buffer: full layer received no full_loc; "
            "ForwardMetadata.out_cache_loc_full_physical must be precomputed for "
            "the shared KV pool."
        )
        self.full_kv_pool.set_kv_buffer(
            None,
            full_loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=pool_layer_id,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Should never be called on the composite — compaction operates
        # per-sub-pool via `SharedMHATokenToKVPool.move_kv_cache` directly
        # (each `MultiEndedAllocator._compact_pending` calls
        # `getattr(self._kvcache, "move_kv_cache", None)` where
        # `self._kvcache` is the per-sub-pool view, not this composite).
        raise NotImplementedError(
            "SharedSWAKVPool.move_kv_cache should not be called; compaction "
            "operates per-sub-pool via SharedMHATokenToKVPool.move_kv_cache."
        )

    # -- HiCache shims (translate virtual->physical, then delegate) --

    @staticmethod
    def _virt_tokens_to_phys_tokens(
        virt_tokens: torch.Tensor, allocator
    ) -> torch.Tensor:
        """Translate virtual TOKEN ids → physical TOKEN ids on the given
        sub-allocator. Page-aware: when ``allocator.page_size > 1``, applies
        the `virt_page * page_size + offset` math.

        Returns ``-1`` for any input whose virtual page is unbound (i.e.
        ``v2p_page[virt_page] == -1``) — propagated as ``-1 * page_size +
        offset``, but callers (HiCache) filter out negatives via
        ``swa_phys >= 0`` so this is safe.
        """
        ps = allocator.page_size
        if ps == 1:
            return allocator.virtual_to_physical[virt_tokens]
        virt_pages = virt_tokens // ps
        offsets = virt_tokens % ps
        phys_pages = allocator.virtual_to_physical[virt_pages]
        return phys_pages * ps + offsets

    def get_cpu_copy(self, indices, mamba_indices=None):
        assert self._full_allocator is not None
        assert self._swa_allocator is not None
        # `indices` are virtual TOKEN ids; translate per sub-pool with the
        # same page math as `translate_loc_from_full_to_swa` so the produced
        # physical token ids are correct at any page_size.
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
        full_cpu = self.full_kv_pool.get_cpu_copy(full_phys)
        valid = swa_phys >= 0
        swa_cpu = None
        if bool(valid.any().item()):
            swa_cpu = self.swa_kv_pool.get_cpu_copy(swa_phys[valid])
        return {"full": full_cpu, "swa": swa_cpu}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        assert self._full_allocator is not None
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        self.full_kv_pool.load_cpu_copy(kv_cache_cpu["full"], full_phys)
        if kv_cache_cpu.get("swa") is not None:
            assert self._swa_allocator is not None
            swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
            self.swa_kv_pool.load_cpu_copy(kv_cache_cpu["swa"], swa_phys)


# ---------------------------------------------------------------------------
# Factory — SWA bundle
# ---------------------------------------------------------------------------


class SharedSWAPoolBundle(NamedTuple):
    shared_kv_pool: SharedKVPool
    token_to_kv_pool: object  # SharedSWAKVPool
    token_to_kv_pool_allocator: object  # SharedSWATokenToKVPoolAllocator


def init_shared_swa_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    swa_attention_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    full_max_total_num_tokens: int,
    swa_max_total_num_tokens: int,
    enable_memory_saver: bool,
    need_sort: bool,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
) -> SharedSWAPoolBundle:
    """Build the SWA-hybrid shared-pool stack: `SharedKVPool` (full + swa
    sub-pools), `SharedSWAKVPool` (composite KV cache), and
    `SharedSWATokenToKVPoolAllocator`."""
    from sglang.srt.mem_cache.multi_ended_allocator import (
        SharedSWATokenToKVPoolAllocator,
    )

    # Both sub-allocators are page-aware (one virtual ID space at PAGE
    # granularity, two physical-holding sub-pools that compact pages
    # independently). The
    # kernel-once-in-virtual-space discipline in
    # `SharedSWATokenToKVPoolAllocator.alloc_extend` preserves the upstream
    # tail-page-reuse contract across both sub-pools.
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    assert (
        len(full_attention_layer_ids) > 0
    ), "SWA-hybrid with zero full-attention layers is degenerate"
    assert (
        len(swa_attention_layer_ids) > 0
    ), "SWA-hybrid with zero SWA-attention layers is degenerate"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # Layout convention (mirrors the mamba builder): full-attn KV at the
    # HIGH-byte end (grow-down), the SWA peer at the LOW-byte end (grow-up).
    # Both grow toward the shared middle gap; the math is direction-agnostic.
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=len(swa_attention_layer_ids),
        head_num=swa_head_num,
        head_dim=swa_head_dim,
        v_head_dim=swa_v_head_dim,
        store_dtype=store_dtype,
        grow_direction="up",
    )
    total_bytes = (
        full_max_total_num_tokens * full_spec.entry_bytes()
        + swa_max_total_num_tokens * swa_spec.entry_bytes()
    )
    shared_pool = SharedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, swa_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    token_to_kv_pool = SharedSWAKVPool(
        shared_buffer=shared_pool,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        page_size=page_size,
        start_layer=start_layer,
        end_layer=end_layer,
        enable_memory_saver=enable_memory_saver,
    )
    allocator = SharedSWATokenToKVPoolAllocator(
        shared_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        full_max_total_num_tokens=full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
    )

    logger.info(
        "[shared-pool] ============================================================"
    )
    logger.info("[shared-pool] SHARED KV POOL ENABLED -- path=SWA hybrid")
    logger.info(
        "[shared-pool]   full_layers=%d, swa_layers=%d, head_num=%d, head_dim=%d, "
        "v_head_dim=%d, swa_head_num=%d, swa_head_dim=%d, swa_v_head_dim=%d, "
        "page_size=%d",
        len(full_attention_layer_ids),
        len(swa_attention_layer_ids),
        head_num,
        head_dim,
        v_head_dim,
        swa_head_num,
        swa_head_dim,
        swa_v_head_dim,
        page_size,
    )
    logger.info(
        "[shared-pool]   total_bytes=%d (=%.2f GB), full_max_total_num_tokens=%d, "
        "swa_max_total_num_tokens=%d, joint_available=%d slots",
        total_bytes,
        total_bytes / GB,
        full_max_total_num_tokens,
        swa_max_total_num_tokens,
        allocator.available_size(),
    )
    logger.info(
        "[shared-pool] ============================================================"
    )
    return SharedSWAPoolBundle(
        shared_kv_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
    )
