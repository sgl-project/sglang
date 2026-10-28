# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""UnifiedKVPool -- one physical `uint8` byte buffer shared by N sub-pools.

Two END `MultiEndedAllocator`s grow inward from opposite ends; optional
"float" MIDDLE pools live between their frontiers (chain order
`[up end, floats..., down end]`). Eager- or lazy-compacting `free` keeps each
pool's byte range reclaimable. Layout is envelope-major (a slot's data for all
its layers in one contiguous byte envelope) so a freed slot vacates a region a
neighbor can grow into. Everything above the allocator stores virtual slot
IDs; the allocator owns the per-sub-pool virtual<->physical tables and
compaction only mutates those (no reference rewriting).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch.profiler import record_function

from sglang.kernels.ops.kvcache.copy_pages import copy_pages
from sglang.kernels.ops.kvcache.zero_pages import zero_pages
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.layout.base import (
    SubPoolSpec,
    _store_dtype_for,
)
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
from sglang.srt.mem_cache.layout.fused_draft import (
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.layout.token_major import (
    build_dense_views,
    build_mamba_entry_views,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KvBufferDesc,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# UnifiedKVPool — the byte buffer + the per-sub-pool views
# ---------------------------------------------------------------------------


def _assert_physical_id_bound(*, sub_pool_name: str, n_rows: int) -> None:
    """Physical ids must fit the int32 read-index buffers."""
    assert n_rows < 2**31, (
        f"sub-pool {sub_pool_name!r}: {n_rows} physical ids exceed the int32 "
        f"bound (2^31) that read-index buffers assume. "
        "Reduce max_total_num_tokens."
    )


def _reserved_floor_bytes(sub_pool_specs: List[SubPoolSpec], page_size: int) -> int:
    """Bytes at the bottom of the buffer reserved as the slot-0 padding sink.

    Slot-0 dummy writes for every sub-pool land here; each sub-pool's first
    allocatable slot is chosen so real data starts past it. A slot-0 write
    touches only slot 0's entry, but a PAGE-AWARE sub-pool allocates whole
    pages, so its first allocatable page starts past all of page 0
    (page_size * entry_bytes) -- while a mamba sub-pool is page_size=1, so its
    entry is charged ONCE. Charging a mamba entry per page would reserve
    page_size * ~100 MB of buffer that the sink never touches.

    Single source of truth: `UnifiedKVPool` reserves exactly this, and the
    factories' bs=1 feasibility floors charge exactly this.
    """
    return max(
        [max(s.entry_bytes() for s in sub_pool_specs)]
        + [
            page_size * s.entry_bytes()
            for s in sub_pool_specs
            if not isinstance(s, MambaSubPoolSpec)  # mamba is page_size=1
        ]
    )


class UnifiedKVPool:
    """One physical `uint8` byte buffer shared by N sub-pools, each exposing
    per-layer strided views over its own byte range. Two END pools (one
    grow-up, one grow-down) own the buffer's ends; optional "float" MIDDLE
    pools live between their frontiers. Allocators keep byte ranges disjoint;
    no usage tracking here.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpec],
        device: str,
        enable_memory_saver: bool,
        page_size: int = 1,
        post_capture_active: bool = False,
        bs1_floor_terms: Optional[List[Tuple[str, int]]] = None,
        fused_draft: Optional[FusedDraftPlacement] = None,
    ):
        assert page_size >= 1, f"page_size must be >= 1; got {page_size}"
        assert len(sub_pool_specs) >= 2, (
            f"UnifiedKVPool needs >= 2 sub-pools; got {len(sub_pool_specs)}"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == len(names), (
            f"sub-pool names must be unique; got {names}"
        )
        # Per-spec direction validity already ran in each spec's __post_init__.
        up_specs = [s for s in sub_pool_specs if s.grow_direction == "up"]
        down_specs = [s for s in sub_pool_specs if s.grow_direction == "down"]
        float_specs = [s for s in sub_pool_specs if s.grow_direction == "float"]
        assert len(up_specs) == 1 and len(down_specs) == 1, (
            f"UnifiedKVPool needs exactly one grow-up and one grow-down END "
            f"sub-pool; got directions "
            f"{[s.grow_direction for s in sub_pool_specs]}"
        )

        self.device = device
        self.total_bytes = total_bytes
        # Canonical chain order, low byte end -> high: grow-up end, float
        # middles (input order preserved), grow-down end. The allocators'
        # neighbour wiring follows it; all other access is by-name.
        self.sub_pool_specs: List[SubPoolSpec] = [
            up_specs[0],
            *float_specs,
            down_specs[0],
        ]
        self._page_size = page_size
        self._specs_by_name: Dict[str, SubPoolSpec] = {
            s.name: s for s in sub_pool_specs
        }
        self.post_capture_active = post_capture_active
        self._post_capture_owner: Optional[KvVmmBufferOwner] = None
        self._bs1_floor_terms = bs1_floor_terms or []
        self.fused_draft = fused_draft
        for spec in sub_pool_specs:
            expected = None if fused_draft is None else fused_draft.region(spec.name)
            assert spec.draft_region is expected, (
                f"sub-pool {spec.name!r}: draft_region {spec.draft_region} does "
                f"not match the fused draft placement's {expected}"
            )

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        # Slot-0 dummy writes for both pools land in the reserved low-byte sink;
        # each pool's first allocatable slot is chosen so real data starts past it.
        # A page-aware sub-pool allocates whole pages, so the sink is all of page
        # 0 (up to page_size * entry_bytes), not one slot entry -- reserve the
        # max of both.
        reserved_floor = _reserved_floor_bytes(self.sub_pool_specs, page_size)
        self._reserved_floor = reserved_floor

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            if post_capture_active:
                dev = torch.device(device)
                if dev.type != "cuda":
                    raise ValueError(
                        "UnifiedKVPool post-capture backing requires a CUDA device"
                    )
                device_id = (
                    dev.index if dev.index is not None else torch.cuda.current_device()
                )
                self._post_capture_owner = KvVmmBufferOwner(
                    device=device,
                    device_id=device_id,
                    store_dtype=torch.uint8,
                    page_size=1,
                    reserved_num_tokens=total_bytes,
                    buffer_descs=[
                        KvBufferDesc(
                            "unified_raw",
                            (total_bytes,),
                            row_bytes=1,
                            tokens_per_row=1,
                        )
                    ],
                )
                self._raw = self._post_capture_owner.tensors[0]
                # Capture routes every dummy KV write to slot 0, whose page
                # envelope is the whole reserved sink: back it before any graph
                # is captured.
                self._post_capture_owner.ensure_prefix(reserved_floor)
                accessible_raw = self._raw[:reserved_floor]
                self._accessible_bytes = reserved_floor
            else:
                self._raw = torch.empty(total_bytes, dtype=torch.uint8, device=device)
                accessible_raw = self._raw
                self._accessible_bytes = total_bytes
        self._initialize_raw(accessible_raw)

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}
        # MHA: (k_buffer, v_buffer); MLA: [per-layer views];
        # Mamba: (conv_state_list, temporal_state)
        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._mla_views: Dict[str, List[torch.Tensor]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}

        for spec in self.sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            min_slot_index = (reserved_floor + entry_bytes - 1) // entry_bytes  # ceil
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"UnifiedKVPool: sub-pool {spec.name!r} fits only {max_slots} "
                    f"slots in {total_bytes} bytes, but min_slot_index={min_slot_index} "
                    f"leaves no room for real data. Increase total_bytes."
                )
            anchor = 0
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
            elif isinstance(spec, MLASubPoolSpec):
                self._mla_views[spec.name] = self._build_mla_views(
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
            "[unified-memory-pool] UnifiedKVPool %s: total_bytes=%.2f GB (=%d B), "
            "%d sub-pool(s)",
            "reserved VA upper bound" if post_capture_active else "allocated",
            total_bytes / GB,
            total_bytes,
            len(self.sub_pool_specs),
        )
        for s in self.sub_pool_specs:
            logger.info(
                "[unified-memory-pool]   sub-pool %r: kind=%s, layer_num=%d, grow=%s, "
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

    def _initialize_raw(self, raw: torch.Tensor) -> None:
        if envs.SGLANG_DEBUG_POISON_POOL.get():
            # Debug: bf16-NaN-fill so NaN-unsafe reads of never-written bytes
            # fail deterministically.
            raw.view(torch.int16).fill_(0x7FC1)
            raw[: self._reserved_floor].zero_()
            logger.warning(
                "[unified-memory-pool] POISONED: backed pool bytes filled with "
                "bf16-NaN patterns (SGLANG_DEBUG_POISON_POOL)"
            )
        else:
            raw.zero_()  # unset slots must read as zeros (matches non-shared)

    def _max_slots_for_total_bytes(self, total_bytes: int) -> Dict[str, int]:
        max_slots_by_name = {}
        for spec in self.sub_pool_specs:
            max_slots = total_bytes // spec.entry_bytes()
            min_page_index = (
                self._min_slot_index[spec.name] + self._page_size - 1
            ) // self._page_size
            if max_slots // self._page_size <= min_page_index:
                raise ValueError(
                    f"unified pool size {total_bytes} leaves no allocatable page "
                    f"for sub-pool {spec.name!r} after its reserved slot-0 sink"
                )
            max_slots_by_name[spec.name] = max_slots
        return max_slots_by_name

    @property
    def post_capture_backed_bytes(self) -> int:
        if self._post_capture_owner is None:
            return 0
        return self._post_capture_owner.backed_bytes

    @property
    def active_allocation_bytes(self) -> int:
        """Bytes that are safe to access/register in the shared raw tensor."""
        return self._accessible_bytes

    def finalize_backing(self, config) -> None:
        """Back the final shared byte budget without changing any tensor address."""
        if self._post_capture_owner is None:
            raise RuntimeError("UnifiedKVPool post-capture backing is not active")
        final_total_bytes = config.unified_memory_pool_bytes
        if final_total_bytes is None:
            raise ValueError(
                "UnifiedKVPool post-capture sizing requires "
                "config.unified_memory_pool_bytes"
            )
        final_total_bytes = int(final_total_bytes)
        _check_bs1_feasibility_floor(
            total_bytes=final_total_bytes,
            floor_terms=self._bs1_floor_terms,
            factory="UnifiedKVPool.finalize_backing",
        )
        final_max_slots = self._max_slots_for_total_bytes(final_total_bytes)
        self._post_capture_owner.ensure_prefix(final_total_bytes)
        self._initialize_raw(self._raw[:final_total_bytes])
        self.total_bytes = final_total_bytes
        self._accessible_bytes = final_total_bytes
        self._max_slots = final_max_slots

    # -- introspection --

    def spec(self, name: str) -> SubPoolSpec:
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MHASubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        )
        return s

    def mla_spec(self, name: str) -> MLASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MLASubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MLASubPoolSpec"
        )
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MambaSubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        )
        return s

    def require_draft_host_spec(self, name: str) -> SubPoolSpec:
        """The sub-pool spec whose entries carry a fused draft region.

        Kind-agnostic: any spec that resolves a `draft_region` also lays the
        draft parts out in its `layout()`."""
        s = self._specs_by_name[name]
        assert s.draft_region is not None, (
            f"sub-pool {name!r} carries no fused draft region"
        )
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

    def mla_views_for(self, name: str) -> List[torch.Tensor]:
        return self._mla_views[name]

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    def _build_mha_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        num_slots = max_slots // page_size * page_size
        _assert_physical_id_bound(sub_pool_name=spec.name, n_rows=num_slots)
        layout = spec.layout()
        k_views, v_views = (
            build_dense_views(
                self._raw,
                layout=layout,
                part_name=name,
                num_slots=num_slots,
                anchor_bytes=anchor_bytes,
            )
            for name in ("k", "v")
        )
        return k_views, v_views

    def build_dense_draft_views(
        self, sub_pool_name: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K/V views of the DRAFT parts fused into ``sub_pool_name``'s
        entries: same pages, same slot ids, same v2p table as the host."""
        spec = self.require_draft_host_spec(sub_pool_name)
        layout = spec.layout()
        page_size = self._page_size
        num_slots = self.max_slots(sub_pool_name) // page_size * page_size
        k_views, v_views = (
            build_dense_views(
                self._raw,
                layout=layout,
                part_name=name,
                num_slots=num_slots,
                anchor_bytes=self._anchor_bytes[sub_pool_name],
            )
            for name in ("draft_k", "draft_v")
        )
        return k_views, v_views

    def _build_mla_views(
        self,
        spec: MLASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> List[torch.Tensor]:
        num_slots = max_slots // page_size * page_size
        _assert_physical_id_bound(sub_pool_name=spec.name, n_rows=num_slots)
        layout = spec.layout()
        return build_dense_views(
            self._raw,
            layout=layout,
            part_name="kv",
            num_slots=num_slots,
            anchor_bytes=anchor_bytes,
        )

    def _build_mamba_views(
        self, spec: MambaSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return build_mamba_entry_views(
            self._raw,
            layer_num=spec.layer_num,
            conv_state_shapes=spec.conv_state_shapes,
            conv_dtype=spec.conv_dtype,
            temporal_state_shape=spec.temporal_state_shape,
            temporal_dtype=spec.temporal_dtype,
            max_slots=max_slots,
            anchor_bytes=anchor_bytes,
            entry_bytes=spec.entry_bytes(),
        )

    def build_draft_state_views(
        self, sub_pool_name: str
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-layer conv/temporal views of the DRAFT state block fused into
        ``sub_pool_name``'s entries: same slots, same v2p table as the host."""
        spec = self.mamba_spec(sub_pool_name)
        region = spec.draft_region
        assert isinstance(region, DraftStateRegion), (
            f"sub-pool {sub_pool_name!r} carries no fused draft state region"
        )
        return build_mamba_entry_views(
            self._raw,
            layer_num=region.lane_num,
            conv_state_shapes=region.state.conv_state_shapes,
            conv_dtype=region.state.conv_dtype,
            temporal_state_shape=region.state.temporal_state_shape,
            temporal_dtype=region.state.temporal_dtype,
            max_slots=self.max_slots(sub_pool_name),
            anchor_bytes=self._anchor_bytes[sub_pool_name],
            entry_bytes=spec.entry_bytes(),
            offset_bytes=spec.draft_offset_in_entry(),
        )


class UnifiedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool whose per-layer `k_buffer`/`v_buffer` are token-major views
    into a `UnifiedKVPool`: `(num_pages * ps, head_num, head_dim)` with the whole
    entry as slot stride, indexed by the physical token id. `move_physical_pages`
    relocates whole page envelopes.
    """

    requires_physical_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        spec = unified_buffer.mha_spec(sub_pool_name)
        k_views, v_views = unified_buffer.mha_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._k_views = k_views
        self._v_views = v_views
        self._num_pages = max_slots // page_size
        self._page_bytes = page_size * spec.entry_bytes()
        view_rows = self._num_pages * page_size

        super().__init__(
            size=view_rows - page_size,
            page_size=page_size,
            dtype=spec.kv_cache_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=spec.v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,
            kv_cache_layout="page_major",
        )

    def _kv_tokens_per_row(self) -> int:
        return 1

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views
        # This override must initialize the pointers and row strides used by HiCache.
        self._init_data_ptrs_and_strides()

    def _clear_buffers(self):
        # Lifetime owned by UnifiedKVPool; do not delete the views.
        pass

    def get_kv_size_bytes(self):
        return 0, 0  # UnifiedKVPool logs the total; per-sub-pool would double-count

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate slots by whole page envelope (`tgt_loc`/`src_loc` are
        physical token ids, in whole page-major runs)."""
        if tgt_loc.numel() == 0:
            return
        ps = self.page_size
        self.move_physical_pages(
            tgt_loc.view(-1, ps)[:, 0] // ps, src_loc.view(-1, ps)[:, 0] // ps, ps
        )

    def move_physical_pages(
        self, tgt_pages: torch.Tensor, src_pages: torch.Tensor, page_size: int
    ):
        """Copy whole page envelopes `src_pages` -> `tgt_pages`: disjoint sets of
        physical page ids, in pages of `page_size` slots (compaction moves)."""
        assert page_size == self.page_size, (
            f"page ids of {page_size}-slot pages passed to a pool whose envelope "
            f"holds {self.page_size} slots"
        )
        # The envelope view below starts at byte 0, so this sub-pool must be
        # anchored there; a non-zero anchor moves another sub-pool's bytes, and
        # the ids stay in range so nothing downstream notices.
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        with record_function("UnifiedMHA.move_physical_pages"):
            copy_pages(
                self._unified_buffer._raw,
                tgt_pages,
                src_pages,
                self._num_pages,
                self._page_bytes,
            )

    def zero_physical_pages(self, phys_pages: torch.Tensor) -> None:
        """Zero whole page envelopes (PHYSICAL page ids) on allocator
        hand-out."""
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        zero_pages(
            self._unified_buffer._raw,
            phys_pages,
            self._num_pages,
            self._page_bytes,
        )

    def get_contiguous_buf_infos(self):
        """Register the raw buffer as physical page envelopes for PD transfer.

        Full and SWA expose the same allocation with different envelope sizes;
        the transfer backend preserves both logical entries while deduplicating
        the underlying memory registration.
        """
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        return (
            [raw.data_ptr()],
            [self._unified_buffer.active_allocation_bytes],
            [self._page_bytes],
        )

    def get_page_envelope_buffer(self) -> torch.Tensor:
        """Return this sub-pool's page-strided view of the shared allocation."""
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        page_count = (
            self._unified_buffer.max_slots(self._sub_pool_name) // self.page_size
        )
        return raw[: page_count * self._page_bytes].view(page_count, self._page_bytes)

    @property
    def grow_direction(self) -> str:
        """Physical growth direction used by this sub-pool's L1 allocator."""
        return self._unified_buffer.spec(self._sub_pool_name).grow_direction

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported under the unified layout "
            "(_set_kv_buffer_prefix_valid_impl assumes token-id indexing)."
        )


class UnifiedMLATokenToKVPool(MLATokenToKVPool):
    """MLA KV pool whose per-layer `kv_buffer` entries are token-major views
    into a `UnifiedKVPool`: `(num_pages * ps, 1, kv_cache_dim)` with the whole
    entry as slot stride, indexed by the physical token id (`translate_kv_loc`
    of the virtual id), so the stock `MLATokenToKVPool` read/write methods work
    on them unmodified. `move_physical_pages` relocates whole page envelopes.
    """

    requires_physical_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        kv_cache_dtype: torch.dtype,
        page_size: int = 1,
    ):
        spec = unified_buffer.mla_spec(sub_pool_name)
        store_dtype = _store_dtype_for(kv_cache_dtype)
        assert spec.store_dtype == store_dtype, (
            f"sub-pool {sub_pool_name!r} store dtype {spec.store_dtype} does not "
            f"match kv cache dtype {kv_cache_dtype} (store {store_dtype})"
        )

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._kv_views = unified_buffer.mla_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        self._num_pages = max_slots // page_size
        self._page_bytes = page_size * spec.entry_bytes()
        self._view_rows = self._num_pages * page_size

        super().__init__(
            # OOB checks bound locs by `size + page_size`; physical ids run to
            # `_view_rows` (page 0 is the reserved padding sink).
            size=self._view_rows - page_size,
            page_size=page_size,
            dtype=kv_cache_dtype,
            kv_lora_rank=spec.kv_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
        )

    def _create_buffers(self):
        self.kv_buffer = self._kv_views

    def _clear_buffers(self):
        # Lifetime owned by UnifiedKVPool; do not delete the views.
        pass

    # The plan's write translation (`translate_write_loc`) already collapsed
    # the widened id and sent the rows this rank does not own to the padding
    # sink.
    write_loc_is_dcp_resolved = True

    def get_kv_size_bytes(self):
        return 0  # UnifiedKVPool logs the total; per-sub-pool would double-count

    def get_contiguous_buf_infos(self):
        """PD-transfer registration: ONE entry, the raw buffer, addressed as
        ``raw_ptr + physical_page_id * page_envelope_bytes``.

        The transfer item is the whole page envelope (all layers of one page)
        rather than a per-layer region: the layers interleave inside each
        slot's entry. Both sides must therefore build the pool with identical
        specs.
        """
        # The address formula omits the anchor; a nonzero one would mis-address.
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [self._page_bytes]

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate whole page envelopes.

        `tgt_loc`/`src_loc` are physical token ids in whole page-major runs
        (`pages[:, None] * ps + offsets`), relied on here to recover the page
        lists.
        """
        if tgt_loc.numel() == 0:
            return
        ps = self.page_size
        self.move_physical_pages(
            tgt_loc.view(-1, ps)[:, 0] // ps, src_loc.view(-1, ps)[:, 0] // ps, ps
        )

    def move_physical_pages(
        self, tgt_pages: torch.Tensor, src_pages: torch.Tensor, page_size: int
    ):
        """Copy whole page envelopes `src_pages` -> `tgt_pages`: disjoint sets of
        physical page ids, in pages of `page_size` slots (compaction moves). One
        contiguous envelope copy replaces the per-layer strided moves."""
        assert page_size == self.page_size, (
            f"page ids of {page_size}-slot pages passed to a pool whose envelope "
            f"holds {self.page_size} slots"
        )
        with record_function("UnifiedMLA.move_physical_pages"):
            copy_pages(
                self._unified_buffer._raw,
                tgt_pages,
                src_pages,
                self._num_pages,
                self._page_bytes,
            )

    def zero_physical_pages(self, phys_pages: torch.Tensor) -> None:
        """Zero whole page envelopes (PHYSICAL page ids) on allocator
        hand-out."""
        zero_pages(
            self._unified_buffer._raw,
            phys_pages,
            self._num_pages,
            self._page_bytes,
        )


class UnifiedMambaPool(MambaPool):
    """Mamba state pool whose conv/temporal state are strided views into a `UnifiedKVPool`.

    Pure PHYSICAL store: slot lifecycle and the v<->p mapping live in the attached
    `UnifiedMambaSlotAllocator`. Does NOT call `super().__init__()` — replicates the
    minimal `MambaPool` state against the unified buffer so inherited methods work.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        spec = unified_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)
        # PP disagg state transfer maps entries by global layer id.
        self.mamba_layer_ids = list(mamba_layer_ids)
        conv_views, temporal_view = unified_buffer.mamba_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name

        # Replicate the state MambaPool.__init__ would have set.
        self._max_size = max_slots - 1  # -1 for reserved slot 0
        self.size = self._max_size
        self.device = unified_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num
        # GDN/KDA ReplaySSM / spec unsupported; replicate parent's disabled-state
        # attrs so unconditional reads (e.g. `replayssm_cache_base is not None` in
        # the req-slot alloc path) and `... is not None` guards don't AttributeError.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False
        self.enable_linear_replayssm_spec = False
        self.replayssm_spec_fold = False
        self.replayssm_spec_write_pos = None
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None
        self.debug_memory_pool = False
        self.conv_shard_groups = None
        self.conv_slice_axis = spec.conv_slice_axis

        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == self._max_size + 1, (
            f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"
        )

        # Per-draft-token intermediate buffers have a different outer size
        # (spec_state_size+1), so they're NOT in the shared buffer; allocate locally.
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
                    device=unified_buffer.device,
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
                        device=unified_buffer.device,
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

        self.mem_usage = unified_buffer.total_bytes / GB
        logger.info(
            "[unified-memory-pool] UnifiedMambaPool(%s) wrapped unified buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )

    # State ops take PHYSICAL slot ids and act on whole entries, so a fused
    # draft state block is cleared, copied and moved with the host streams.

    def _entry_view(self) -> torch.Tensor:
        spec = self._unified_buffer.mamba_spec(self._sub_pool_name)
        entry_bytes = spec.entry_bytes()
        anchor = self._unified_buffer.anchor_bytes(self._sub_pool_name)
        num_slots = self._max_size + 1
        raw = self._unified_buffer._raw
        entries = raw[anchor : anchor + num_slots * entry_bytes]
        return entries.view(num_slots, entry_bytes)

    def clear_slots(self, indices: torch.Tensor):
        # index_fill_, not `view[indices] = 0`: assigning a Python scalar
        # through an index tensor synchronizes the host.
        self._entry_view().index_fill_(0, indices.to(torch.int64), 0)
        for sibling in self._slot_siblings:
            sibling.reset_slots(indices)

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        assert self.replayssm_write_pos is None
        entries = self._entry_view()
        entries[dst_indices] = entries[src_indices]
        for sibling in self._slot_siblings:
            sibling.copy_slots(src_indices, dst_indices)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Compaction moves disjoint physical slots. Copy their whole envelopes
        # without the conv/temporal gather temporary used by copy_from (COW).
        # Sibling callbacks retain their own copy and scratch-allocation contracts.
        if tgt_loc.numel() == 0:
            return
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        copy_pages(
            self._unified_buffer._raw,
            tgt_loc,
            src_loc,
            self._unified_buffer.max_slots(self._sub_pool_name),
            self._unified_buffer.mamba_spec(self._sub_pool_name).entry_bytes(),
        )
        if self.replayssm_write_pos is not None:
            self.replayssm_write_pos[tgt_loc] = 0
        for sibling in self._slot_siblings:
            sibling.copy_slots(src_loc, tgt_loc)

    def move_physical_pages(
        self, tgt_pages: torch.Tensor, src_pages: torch.Tensor, page_size: int
    ):
        # Cross-pool physical-move contract, implemented by every pool the
        # MultiEndedAllocator wraps. A state slot is one page.
        assert page_size == 1, f"mamba state pages are single slots, got {page_size}"
        self.move_kv_cache(tgt_pages, src_pages)

    # -- PD state transfer (StateType.MAMBA) --
    # The transfer item is the whole per-slot envelope, addressed as
    # `raw_ptr + physical_slot * entry_bytes`. An envelope cannot be TP-resliced
    # or PP-subset, so the per-tensor metadata below stays empty and both sides
    # must build identical mamba specs (equal attn TP, pp=1).

    def get_contiguous_buf_infos(self):
        # The address formula omits the anchor; a nonzero one would mis-address.
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        spec = self._unified_buffer.mamba_spec(self._sub_pool_name)
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [spec.entry_bytes()]

    def get_state_dim_per_tensor(self):
        return []

    def get_state_layer_ids(self):
        return []

    def get_state_slice_outer_counts(self):
        return []

    def get_state_conv_shard_groups(self):
        return []


class UnifiedMambaSlotAllocator:
    """Mamba slot allocator (PHYSICAL view) for the unified memory pool.

    Owns slot alloc/free, sizing, and the v<->p mapping (``translate``), presenting the
    upstream ``MambaSlotAllocator`` interface. ``alloc()`` returns VIRTUAL ids and does
    NOT clear state — clearing is deferred to ``UnifiedMambaPool.clear_slots``.
    """

    def __init__(
        self,
        mea,
        max_size: int,
        device: str,
        take: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    ):
        self._multi_ended_allocator = mea
        # Slots are taken through the chain allocator when one owns the band.
        self._take = mea.alloc if take is None else take
        self._max_size = max_size  # excludes reserved slot 0
        self._device = device
        self._alloc_iter = None  # active alloc_group batch iterator

    # -- translation (owns the v<->p mapping) --

    def translate(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        # VIRTUAL -> PHYSICAL slot ids; page_size==1, so a direct v2p gather.
        return self._multi_ended_allocator.virtual_to_physical[virtual_ids]

    @property
    def virtual_to_physical(self) -> torch.Tensor:
        return self._multi_ended_allocator.virtual_to_physical

    # -- sizing / free-list --

    @property
    def size(self) -> int:
        return self._max_size

    def available_size(self) -> int:
        # Slot-conservation count (max - allocated): the leak-check view, NOT the
        # planner value (use schedulable_available_size for that).
        return self._max_size - self._multi_ended_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        # Byte-coordinated count (>= N => alloc(N) succeeds); credits the peer's
        # drainable holes since alloc flushes the peer before extending.
        return self._multi_ended_allocator.schedulable_available_size()

    @property
    def free_slots(self) -> torch.Tensor:
        # Watermark-derived physical free-list for the invariant checker.
        a = self._multi_ended_allocator
        assert a.page_size == 1, (
            "UnifiedMambaSlotAllocator.free_slots assumes page_size==1; got "
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
        return self._take(need_size)  # VIRTUAL ids

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch that ``alloc(1)`` then draws from."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._take(num_reqs)
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


class UnifiedHybridReqToTokenPool(HybridReqToTokenPool):
    """The request table of a unified pool with a state sub-pool: request rows
    and their slot-id columns only. The slot ids are VIRTUAL, which the
    allocator's id space for `state_sub_pool` maps to physical slots; the
    state itself (pool, slot allocator, layer map) is the allocator's
    `state`, which the unified pool builder constructs."""

    def __init__(
        self,
        *,
        mamba_sub_pool_name: str,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        pre_alloc_size: int = 0,
    ):
        self.state_sub_pool = mamba_sub_pool_name
        super().__init__(
            # `DecodeReqToTokenPool` semantics: rows cover the preallocated
            # requests too, while `self.size` (rebound below) stays the
            # running-request cap the scheduler and leak invariant expect.
            size=size + pre_alloc_size,
            # The state storage is not built here; `_init_mamba_pool` takes
            # only what the id columns need.
            mamba_size=0,
            mamba_spec_state_size=0,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            cache_params=None,
            mamba_layer_ids=[],
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            enable_mamba_extra_buffer_lazy=enable_mamba_extra_buffer_lazy,
            enable_overlap_schedule=enable_overlap_schedule,
            start_layer=start_layer,
        )
        self.size = size
        self.pre_alloc_size = pre_alloc_size

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
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape=None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        # Only the slot-id columns; the storage arguments match the parent
        # signature and are not read.
        if short_conv_layer_ids or ngram_context_len:
            raise ValueError(
                "Qwen4-Exp PLE side states are not supported with "
                "--enable-unified-memory"
            )
        self.device = device
        # Sized by req_to_token's first dim (size + 1; row 0 is padding); self.size
        # would under-size by one row.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    # int64 to match the parent's uncast index_put source (int32 dest
                    # would dtype-mismatch on the first radix prefill).
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def clone_with_new_mamba(self, **kwargs) -> HybridReqToTokenPool:
        raise ValueError(
            "--enable-unified-memory: a draft head with its own recurrent state "
            "binds it through the target's fused draft placement"
        )

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        # The state slots are the allocator's: `ChainAllocator.clear` resets them.
        ReqToTokenPool.clear(self)
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class UnifiedPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object  # RoutedKVCache
    token_to_kv_pool_allocator: object  # ChainAllocator
    # A plan with state has its own request table; None for the caller to build.
    req_to_token_pool: object  # Optional[UnifiedHybridReqToTokenPool]


def _check_bs1_feasibility_floor(
    *,
    total_bytes: int,
    floor_terms: List[Tuple[str, int]],
    factory: str,
) -> None:
    """bs=1 feasibility FLOOR — the retract loop's terminal guarantee.

    The scheduler retracts requests until the LAST one fits; if one worst-case
    request running ALONE does not fit in the buffer, under-sizing is a retract
    LIVELOCK at runtime, not a perf bug. Fail loud at boot, before any pool
    construction, with the itemized requirement.
    """
    floor = sum(b for _, b in floor_terms)
    if total_bytes >= floor:
        return
    detail = " + ".join(f"{name}={b}" for name, b in floor_terms)
    raise RuntimeError(
        f"[unified-memory-pool] {factory}: byte budget {total_bytes} cannot fit "
        f"ONE worst-case request (bs=1 floor {floor} = {detail}). A pool this "
        f"size retract-livelocks at runtime. Raise --mem-fraction-static, lower "
        f"the model context length, or reduce reserved memory."
    )


def _wire_mamba_slot_allocator(
    *,
    mamba_end,
    take: Callable[[int], Optional[torch.Tensor]],
    state: StateStorage,
    device,
) -> UnifiedMambaSlotAllocator:
    """Install Mamba slot allocation and host capacities."""
    state_pool = state.pool
    slot_allocator = UnifiedMambaSlotAllocator(
        mamba_end,
        max_size=state_pool.size,
        device=device,
        take=take,
    )
    state.slots = slot_allocator
    state_pool.host_capacity_tokens = state_pool.size
    state_pool.host_capacity_bytes = (
        state_pool.host_capacity_tokens * mamba_end.entry_bytes
    )
    return slot_allocator
