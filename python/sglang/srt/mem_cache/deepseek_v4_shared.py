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

"""Page-sharded persistent cache storage for DeepSeek V4 prefill CP."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup

from sglang.jit_kernel.dsv4 import (
    fused_k_norm_rope_flashmla,
    fused_store_cache_shared,
)
from sglang.srt.layers.attention.dsv4.shared_cache_access import (
    DSV4SharedCacheAccess,
    DSV4SharedPageLayout,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
    get_compress_state_layout,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    ONLINE_C128,
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.shared_kv.family import (
    OwnerShardedFamily,
    OwnerShardedFamilySpec,
)
from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher
from sglang.srt.mem_cache.shared_kv.transfer import OwnerShardedTransferBuffer
from sglang.srt.utils.common import ceil_div

logger = logging.getLogger(__name__)


def build_dsv4_shared_page_layout(
    *, logical_size: int, page_size: int, cp_size: int
) -> DSV4SharedPageLayout:
    """Build the owner-page layout for one DeepSeek V4 cache family."""
    if logical_size < 0:
        raise ValueError(f"logical_size must be non-negative, got {logical_size}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if cp_size <= 1:
        raise ValueError(f"shared cache requires cp_size > 1, got {cp_size}")

    # Preserve a dummy slot/page and one spare owner page, matching the paged
    # allocators' valid logical range.
    logical_pages = ceil_div(logical_size + 1, page_size)
    requested_pages = ceil_div(logical_pages, cp_size) + 1
    return DSV4SharedPageLayout(
        OwnerShardedLayout(
            cp_size=cp_size,
            ownership_granule=page_size,
            logical_rows=requested_pages * cp_size * page_size,
        )
    )


@triton.jit
def _translate_shared_slots_kernel(
    logical_slots,
    physical_slots,
    numel,
    PAGE_SIZE: tl.constexpr,
    CP_SIZE: tl.constexpr,
    PAGES_PER_RANK: tl.constexpr,
    PADDING_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    logical_slot = tl.load(logical_slots + offsets, mask=mask)
    valid = logical_slot != PADDING_VALUE
    safe_slot = tl.where(valid, logical_slot, 0)
    logical_page = safe_slot // PAGE_SIZE
    page_offset = safe_slot % PAGE_SIZE
    owner = logical_page % CP_SIZE
    local_page = logical_page // CP_SIZE
    physical_page = owner * PAGES_PER_RANK + local_page
    physical_slot = physical_page * PAGE_SIZE + page_offset
    tl.store(
        physical_slots + offsets,
        tl.where(valid, physical_slot, logical_slot),
        mask=mask,
    )


def _translate_shared_slots_fused(
    layout: DSV4SharedPageLayout, logical_slots: torch.Tensor
) -> torch.Tensor:
    """Translate owner-sharded slots with one CUDA launch."""
    if not logical_slots.is_cuda:
        return layout.translate_slots(logical_slots)
    if logical_slots.numel() == 0:
        return logical_slots.clone()
    physical_slots = torch.empty_like(logical_slots)
    block_size = 256
    _translate_shared_slots_kernel[(triton.cdiv(logical_slots.numel(), block_size),)](
        logical_slots,
        physical_slots,
        logical_slots.numel(),
        PAGE_SIZE=layout.page_size,
        CP_SIZE=layout.cp_size,
        PAGES_PER_RANK=layout.pages_per_rank,
        PADDING_VALUE=layout.padding_value,
        BLOCK_SIZE=block_size,
    )
    return physical_slots


def _build_shared_page_stage_plan(
    layout: DSV4SharedPageLayout,
    logical_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a page gather/remap plan that can be reused across layers."""
    remapped = logical_pages.clone()
    valid = logical_pages >= 0
    if logical_pages.numel() == 0:
        return logical_pages.reshape(-1).long(), remapped

    physical_pages = layout.translate_pages(logical_pages[valid]).long()
    unique_pages, inverse = torch.unique(
        physical_pages, sorted=True, return_inverse=True
    )
    remapped[valid] = inverse.to(remapped.dtype)
    return unique_pages, remapped


def _build_single_request_page_stage_plan(
    layout: DSV4SharedPageLayout,
    logical_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a compact plan from one repeated per-query request row."""
    if logical_pages.ndim != 2 or logical_pages.shape[0] == 0:
        return _build_shared_page_stage_plan(layout, logical_pages)

    physical_pages = layout.translate_pages(logical_pages[0]).long()
    compact_row = torch.arange(
        logical_pages.shape[1],
        dtype=logical_pages.dtype,
        device=logical_pages.device,
    )
    remapped = compact_row.unsqueeze(0).expand_as(logical_pages).contiguous()
    return physical_pages, remapped


def _stage_shared_cache_from_plan(
    cache: torch.Tensor, physical_pages: torch.Tensor
) -> torch.Tensor:
    """Gather one layer using a precomputed shared-page read plan."""
    return cache.index_select(0, physical_pages)


def _build_shared_slot_stage_plan(
    layout: DSV4SharedPageLayout,
    logical_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a reusable page gather plan for flattened slot ids."""
    remapped = logical_slots.clone()
    valid = logical_slots >= 0
    if logical_slots.numel() == 0:
        return logical_slots.reshape(-1).long(), remapped

    valid_slots = logical_slots[valid]
    if valid_slots.numel() == 0:
        # FlashMLA clamps compressed top-k lengths to at least one during
        # empty/startup batches. Keep a readable dummy page even though every
        # index is the padding sentinel; a zero-page cache would make that
        # guarded access point outside the allocation.
        return logical_slots.new_zeros(1, dtype=torch.long), remapped
    logical_pages = torch.div(valid_slots, layout.page_size, rounding_mode="floor")
    offsets = torch.remainder(valid_slots, layout.page_size)
    physical_pages = layout.translate_pages(logical_pages).long()
    unique_pages, inverse = torch.unique(
        physical_pages, sorted=True, return_inverse=True
    )
    remapped[valid] = inverse.to(remapped.dtype) * layout.page_size + offsets.to(
        remapped.dtype
    )
    return unique_pages, remapped


@triton.jit
def _remap_shared_slots_from_sorted_pages_kernel(
    logical_slots,
    sorted_logical_pages,
    remapped_slots,
    numel,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    PADDING_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    logical_slot = tl.load(logical_slots + offsets, mask=mask, other=PADDING_VALUE)
    valid = logical_slot != PADDING_VALUE
    safe_slot = tl.where(valid, logical_slot, 0)
    logical_page = safe_slot // PAGE_SIZE
    page_offset = safe_slot % PAGE_SIZE

    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    hi = tl.full((BLOCK_SIZE,), num_pages, dtype=tl.int32)
    for _ in range(SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        candidate = tl.load(
            sorted_logical_pages + mid,
            mask=mask & active & (mid < num_pages),
            other=0x7FFFFFFF,
        )
        advance = active & (candidate <= logical_page)
        lo = tl.where(advance, mid + 1, lo)
        hi = tl.where(active & ~advance, mid, hi)

    remapped = (lo - 1) * PAGE_SIZE + page_offset
    tl.store(
        remapped_slots + offsets,
        tl.where(valid, remapped, logical_slot),
        mask=mask,
    )


def _remap_shared_slots_from_sorted_pages(
    layout: DSV4SharedPageLayout,
    logical_slots: torch.Tensor,
    sorted_logical_pages: torch.Tensor,
) -> torch.Tensor:
    if not logical_slots.is_cuda:
        valid = logical_slots >= 0
        safe_slots = torch.where(valid, logical_slots, 0)
        logical_pages = torch.div(safe_slots, layout.page_size, rounding_mode="floor")
        page_offsets = torch.remainder(safe_slots, layout.page_size)
        compact_pages = (
            torch.searchsorted(sorted_logical_pages, logical_pages, right=True) - 1
        )
        return torch.where(
            valid,
            compact_pages.to(logical_slots.dtype) * layout.page_size + page_offsets,
            logical_slots,
        )

    remapped = torch.empty_like(logical_slots)
    block_size = 256
    max_logical_pages = layout.pages_per_rank * layout.cp_size
    _remap_shared_slots_from_sorted_pages_kernel[
        (triton.cdiv(logical_slots.numel(), block_size),)
    ](
        logical_slots,
        sorted_logical_pages,
        remapped,
        logical_slots.numel(),
        sorted_logical_pages.numel(),
        PAGE_SIZE=layout.page_size,
        SEARCH_STEPS=max_logical_pages.bit_length(),
        PADDING_VALUE=layout.padding_value,
        BLOCK_SIZE=block_size,
    )
    return remapped


def _build_single_request_slot_stage_plan(
    layout: DSV4SharedPageLayout,
    logical_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an SWA plan without uniquing every overlapping query window.

    The first full causal window covers the prefix before this chunk and
    column zero contains every CP-local query's current slot. Since a cache
    page is wider than the CP stride, those candidates cover every page used
    by the overlapping single-request windows.
    """
    if (
        logical_slots.ndim != 2
        or logical_slots.shape[0] == 0
        or logical_slots.shape[1] == 0
    ):
        return _build_shared_slot_stage_plan(layout, logical_slots)

    candidates = torch.cat((logical_slots[0], logical_slots[:, 0]))
    fallback = torch.clamp_min(candidates[0], 0)
    safe_candidates = torch.where(candidates >= 0, candidates, fallback)
    candidate_pages = torch.div(
        safe_candidates, layout.page_size, rounding_mode="floor"
    )
    # Row zero walks the current query's window and column zero walks the
    # CP-local queries at ``cp_size`` slot increments, so their actual
    # allocator page ids cover every overlapping window. Compact those ids
    # into a fixed maximum page span; allocator ids need not be contiguous
    # after radix-cache reuse, and torch.unique's dynamic output size would
    # synchronize the host once per prefill chunk.
    max_slot_span = (
        logical_slots.shape[1] - 1 + (logical_slots.shape[0] - 1) * layout.cp_size
    )
    # The continuous real-token span needs at most ``ceil(span / page) + 1``
    # pages. CP padding can append a token-zero row after the real rows, whose
    # allocator page may be disjoint after radix reuse, so reserve one more.
    max_num_pages = (max_slot_span + layout.page_size - 1) // layout.page_size + 2
    sorted_pages = torch.sort(candidate_pages).values
    starts = torch.ones_like(sorted_pages, dtype=torch.bool)
    starts[1:] = sorted_pages[1:] != sorted_pages[:-1]
    unique_ranks = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    # Keep a fixed output extent without torch.unique's dynamic-size host
    # synchronization. Duplicate scatters write the same page id, and the
    # unused tail repeats the final valid page for a readable padded entry.
    logical_pages = sorted_pages[-1:].expand(max_num_pages).clone()
    logical_pages.scatter_(0, unique_ranks, sorted_pages)
    physical_pages = layout.translate_pages(logical_pages).long()
    remapped = _remap_shared_slots_from_sorted_pages(
        layout,
        logical_slots,
        logical_pages,
    )
    return physical_pages, remapped


@dataclass
class SharedDeepSeekV4Family:
    """One page-sharded DSV4 tensor family with per-layer VMM aliases."""

    name: str
    layout: DSV4SharedPageLayout
    storage: OwnerShardedFamily
    logical_size: int

    @classmethod
    def create(
        cls,
        *,
        name: str,
        logical_size: int,
        page_size: int,
        layer_num: int,
        dtype: torch.dtype,
        row_shape: tuple[int, ...],
        rows_per_page: int,
        cp_size: int,
        cpu_group: ProcessGroup,
        map_rank_local: bool = True,
        zero_initialize: bool = True,
    ) -> SharedDeepSeekV4Family:
        if layer_num <= 0:
            raise ValueError(f"layer_num must be positive, got {layer_num}")
        if rows_per_page <= 0:
            raise ValueError(f"rows_per_page must be positive, got {rows_per_page}")
        requested_layout = build_dsv4_shared_page_layout(
            logical_size=logical_size,
            page_size=page_size,
            cp_size=cp_size,
        )
        storage = OwnerShardedFamily.create(
            spec=OwnerShardedFamilySpec(
                name=name,
                num_layers=layer_num,
                logical_rows_per_layer=(requested_layout.owner_layout.logical_rows),
                ownership_granule=page_size,
                storage_rows_per_granule=rows_per_page,
                row_shape=row_shape,
                dtype=dtype,
                map_rank_local=map_rank_local,
            ),
            cp_size=cp_size,
            cpu_group=cpu_group,
            zero_initialize=zero_initialize,
        )
        return cls(
            name=name,
            layout=DSV4SharedPageLayout(storage.layout),
            storage=storage,
            logical_size=logical_size,
        )

    @property
    def slab(self):
        return self.storage.slab

    @property
    def global_views(self) -> list[torch.Tensor]:
        return self.slab.global_views

    @property
    def local_views(self) -> list[torch.Tensor]:
        return self.slab.local_views

    @property
    def rank_local_views(self) -> list[torch.Tensor]:
        return self.slab.rank_local_views

    @property
    def physical_bytes(self) -> int:
        return self.storage.accounting().mapped_bytes_per_rank

    def close(self) -> None:
        self.storage.close()


class SharedDeepSeekV4SingleKVPool(DeepSeekV4SingleKVPool):
    """Packed DSV4 FlashMLA cache with one physical page across CP ranks."""

    def __init__(
        self,
        *args,
        shared_rank: int,
        shared_size: int,
        shared_cpu_group: ProcessGroup,
        shared_family_name: str,
        **kwargs,
    ):
        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.shared_cpu_group = shared_cpu_group
        self.shared_family_name = shared_family_name
        self.shared_family: SharedDeepSeekV4Family | None = None
        self.local_kv_buffer: list[torch.Tensor] = []
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._clear_shared_family()
            raise

    def _clear_shared_family(self) -> None:
        family = self.shared_family
        self.kv_buffer = []
        self.local_kv_buffer = []
        self.shared_family = None
        if family is not None:
            family.close()

    def close(self) -> None:
        self._clear_shared_family()

    def _create_buffers(self) -> None:
        # create_buffer establishes the packed DSV4 page stride and validates
        # the model's 584-byte token layout before the VMM allocation.
        prototype = self.create_buffer(num_pages=1)
        del prototype
        self.shared_family = SharedDeepSeekV4Family.create(
            name=self.shared_family_name,
            logical_size=self.size,
            page_size=self.page_size,
            layer_num=self.layer_num,
            dtype=self.store_dtype,
            row_shape=(self.bytes_per_page_padded,),
            rows_per_page=1,
            cp_size=self.shared_size,
            cpu_group=self.shared_cpu_group,
            map_rank_local=False,
        )
        self.kv_buffer = self.shared_family.global_views
        self.local_kv_buffer = self.shared_family.local_views

    def translate_slots_for_read(self, loc: torch.Tensor) -> torch.Tensor:
        assert self.shared_family is not None
        return _translate_shared_slots_fused(self.shared_family.layout, loc)

    def prepare_slots_for_read(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.shared_family is not None
        return _build_shared_slot_stage_plan(self.shared_family.layout, slots)

    def stage_slots_with_plan(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        return _stage_shared_cache_from_plan(
            self.get_key_buffer(layer_id), physical_pages
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache_shared(
            input=cache_k,
            cache=self.local_kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
        )


class SharedDeepSeekV4IndexerPool(DeepSeekV4IndexerPool):
    """Packed DSV4 C4 indexer with page-owner shared VMM storage."""

    def __init__(
        self,
        *args,
        shared_rank: int,
        shared_size: int,
        shared_cpu_group: ProcessGroup,
        **kwargs,
    ):
        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.shared_cpu_group = shared_cpu_group
        self.shared_family: SharedDeepSeekV4Family | None = None
        self.local_index_k_with_scale_buffer: list[torch.Tensor] = []
        self.rank_local_index_k_with_scale_buffer: list[torch.Tensor] = []
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._clear_shared_family()
            raise

    def _clear_shared_family(self) -> None:
        family = self.shared_family
        self.index_k_with_scale_buffer = []
        self.local_index_k_with_scale_buffer = []
        self.rank_local_index_k_with_scale_buffer = []
        self.shared_family = None
        if family is not None:
            family.close()

    def close(self) -> None:
        self._clear_shared_family()

    def _create_buffer(self) -> None:
        page_bytes = self.page_size * self.get_bytes_per_token()
        self.shared_family = SharedDeepSeekV4Family.create(
            name="c4_indexer",
            logical_size=self.size,
            page_size=self.page_size,
            layer_num=self.layer_num,
            dtype=self.index_k_with_scale_buffer_dtype,
            row_shape=(page_bytes,),
            rows_per_page=1,
            cp_size=self.shared_size,
            cpu_group=self.shared_cpu_group,
        )
        self.local_index_k_with_scale_buffer = self.shared_family.local_views
        self.rank_local_index_k_with_scale_buffer = self.shared_family.rank_local_views
        if not self.rank_local_index_k_with_scale_buffer:
            raise RuntimeError("DSV4 shared indexer requires rank-local VMM aliases")
        # DeepGEMM derives the CUDA device from the cache base pointer.  Put
        # this rank's physical segment first, then translate owner ids relative
        # to this rank so every process presents a device-local base pointer.
        self.index_k_with_scale_buffer = self.rank_local_index_k_with_scale_buffer

    def translate_slots_for_read(self, loc: torch.Tensor) -> torch.Tensor:
        assert self.shared_family is not None
        return self.shared_family.layout.translate_slots_for_rank(
            loc, rank=self.shared_rank
        )

    def prepare_pages_for_read(
        self, pages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.shared_family is not None
        return _build_shared_page_stage_plan(self.shared_family.layout, pages)

    def stage_pages_with_plan(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        assert self.shared_family is not None
        # The staged tensor is ordinary local CUDA memory, so DeepGEMM no
        # longer issues repeated TMA reads to peer HBM for every query tile.
        return _stage_shared_cache_from_plan(
            self.shared_family.global_views[layer_id], physical_pages
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache_shared(
            input=cache_k,
            cache=self.local_index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
        )


class SharedCompressStatePool(CompressStatePool):
    """One logical compressor-state layer backed by a shared family slab."""

    def __init__(
        self,
        *args,
        shared_family: SharedDeepSeekV4Family,
        shared_layer_id: int,
        shared_rank: int,
        **kwargs,
    ):
        self.shared_family = shared_family
        self.shared_layer_id = shared_layer_id
        self.shared_rank = shared_rank
        self._shared_buffer = shared_family.rank_local_views[shared_layer_id]
        super().__init__(*args, **kwargs)

    def _alloc_kv_score_buffer(
        self, *, dtype: torch.dtype, device: str, enable_memory_saver: bool
    ) -> None:
        del device, enable_memory_saver
        if self._shared_buffer.dtype != dtype:
            raise TypeError(
                f"shared state dtype mismatch: {self._shared_buffer.dtype} != {dtype}"
            )
        if self._shared_buffer.shape[-1] != self.last_dim:
            raise ValueError(
                "shared state row mismatch: "
                f"{self._shared_buffer.shape[-1]} != {self.last_dim}"
            )
        self.kv_score_buffer = KVAndScore(self._shared_buffer)

    def _initialize_kv_score_buffer(self, *, online: bool, ratio: int) -> None:
        if online:
            raise ValueError("DSV4 shared state does not support online C128")
        dummy_slot = self._size - 1
        dummy_page, page_offset = divmod(dummy_slot, ratio)
        layout = self.shared_family.layout
        if dummy_page % layout.cp_size != self.shared_rank:
            return
        local_slot = dummy_page // layout.cp_size * ratio + page_offset
        self.kv_score_buffer[local_slot].clear()

    def get_shared_state_layout(self) -> tuple[int, int, int]:
        layout = self.shared_family.layout
        return self.shared_rank, layout.cp_size, layout.pages_per_rank

    def translate_state_slots_for_read(self, slots: torch.Tensor) -> torch.Tensor:
        return self.shared_family.layout.translate_slots_for_rank(
            slots, rank=self.shared_rank
        )

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> KVAndScore:
        return self.kv_score_buffer[self.translate_state_slots_for_read(state_loc)]

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: KVAndScore):
        physical = self.translate_state_slots_for_read(state_loc)
        self.kv_score_buffer[physical] = value


class SharedDeepSeekV4TokenToKVPool(DeepSeekV4TokenToKVPool):
    """DeepSeek V4 cache whose persistent page families are shared by CP.

    Logical slot/page ids remain unchanged at the allocator boundary.  Stores
    select the unique page owner and target its compact local allocation;
    readers translate logical ids into the rank-major VMM alias.
    """

    def __init__(
        self,
        *args,
        shared_rank: int,
        shared_size: int,
        **kwargs,
    ):
        if shared_size <= 1:
            raise ValueError(f"shared_size must be greater than one, got {shared_size}")
        if kwargs.get("enable_hisparse", False):
            raise ValueError("DSV4 shared KV cache is incompatible with HiSparse")
        if ONLINE_C128:
            raise ValueError("DSV4 shared state does not support online C128")

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError("DSV4 shared KV cache does not support unified KV")

        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.shared_state_families: dict[str, SharedDeepSeekV4Family] = {}
        self.shared_cp_group = None
        self.shared_write_publisher: SharedWritePublisher | None = None
        self.shared_cache_access: DSV4SharedCacheAccess | None = None
        try:
            super().__init__(*args, **kwargs)
            self._log_shared_family_accounting()
            self.shared_write_publisher = SharedWritePublisher(self._get_cp_group())
            self.shared_cache_access = DSV4SharedCacheAccess(self)
        except BaseException:
            self._clear_buffers()
            raise

    def _clear_buffers(self) -> None:
        families: list[SharedDeepSeekV4Family] = []
        for pool_name in (
            "swa_kv_pool",
            "c4_kv_pool",
            "c128_kv_pool",
            "c4_indexer_kv_pool",
        ):
            pool = getattr(self, pool_name, None)
            if pool is None:
                continue
            family = getattr(pool, "shared_family", None)
            if family is not None:
                families.append(family)
            for buffer_name in (
                "kv_buffer",
                "local_kv_buffer",
                "index_k_with_scale_buffer",
                "local_index_k_with_scale_buffer",
                "rank_local_index_k_with_scale_buffer",
            ):
                buffer = getattr(pool, buffer_name, None)
                if isinstance(buffer, list):
                    buffer.clear()
            if hasattr(pool, "shared_family"):
                pool.shared_family = None

        for pool_list_name in (
            "compress_state_pools",
            "indexer_compress_state_pools",
        ):
            for pool in getattr(self, pool_list_name, None) or []:
                if pool is None:
                    continue
                if hasattr(pool, "kv_score_buffer"):
                    pool.kv_score_buffer = None
                if hasattr(pool, "_shared_buffer"):
                    pool._shared_buffer = None
                if hasattr(pool, "shared_family"):
                    pool.shared_family = None

        state_families = getattr(self, "shared_state_families", None)
        if state_families:
            families.extend(state_families.values())
            state_families.clear()
        self.shared_write_publisher = None
        self.shared_cache_access = None

        closed_ids: set[int] = set()
        for family in families:
            family_id = id(family)
            if family_id in closed_ids:
                continue
            closed_ids.add(family_id)
            family.close()

    def close(self) -> None:
        self._clear_buffers()

    @staticmethod
    def _page_transfer_buffer(
        tensor: torch.Tensor,
        *,
        rank_stride_owner_pages: int | None = None,
    ) -> OwnerShardedTransferBuffer:
        page_bytes = tensor[0].nbytes
        return OwnerShardedTransferBuffer(
            tensor=tensor,
            item_bytes=page_bytes,
            owner_page_bytes=page_bytes,
            rank_stride_owner_pages=rank_stride_owner_pages,
        )

    def get_owner_sharded_kv_transfer_buffers(
        self,
    ) -> list[OwnerShardedTransferBuffer]:
        """Return local-owner main-KV tensors in ``kv_data_ptrs`` order."""
        buffers = [
            *(self._page_transfer_buffer(t) for t in self.c4_kv_pool.local_kv_buffer),
            *(
                self._page_transfer_buffer(t)
                for t in self.c4_indexer_kv_pool.local_index_k_with_scale_buffer
            ),
            *(self._page_transfer_buffer(t) for t in self.c128_kv_pool.local_kv_buffer),
        ]
        if len(buffers) != len(self.get_contiguous_buf_infos()[0]):
            raise RuntimeError("shared PD main-KV transfer layout is inconsistent")
        return buffers

    def get_rank_aggregated_kv_transfer_buffers(
        self,
    ) -> list[OwnerShardedTransferBuffer]:
        """Return rank-major VMM views in main-KV wire-pointer order."""

        def family_buffers(family: SharedDeepSeekV4Family):
            stride = family.slab.rank_stride_rows
            return [
                self._page_transfer_buffer(
                    tensor,
                    rank_stride_owner_pages=stride,
                )
                for tensor in family.global_views
            ]

        buffers = [
            *family_buffers(self.c4_kv_pool.shared_family),
            *family_buffers(self.c4_indexer_kv_pool.shared_family),
            *family_buffers(self.c128_kv_pool.shared_family),
        ]
        if len(buffers) != len(self.get_contiguous_buf_infos()[0]):
            raise RuntimeError("shared PD aggregated KV layout is inconsistent")
        return buffers

    @staticmethod
    def _state_transfer_buffer(
        pool: SharedCompressStatePool,
    ) -> OwnerShardedTransferBuffer:
        local = pool.shared_family.local_views[pool.shared_layer_id]
        row_bytes = local[0].nbytes
        owner_page_bytes = pool.ratio * row_bytes
        item_bytes = pool.ring_size * row_bytes
        return OwnerShardedTransferBuffer(
            tensor=local,
            item_bytes=item_bytes,
            owner_page_bytes=owner_page_bytes,
            owner_pages_per_item=pool.ring_size // pool.ratio,
        )

    def get_owner_sharded_state_transfer_buffers(
        self,
    ) -> list[OwnerShardedTransferBuffer]:
        """Return local-owner SWA/C4-state tensors in ``state_data_ptrs`` order."""
        buffers = [
            *(self._page_transfer_buffer(t) for t in self.swa_kv_pool.local_kv_buffer),
            *(
                self._state_transfer_buffer(pool)
                for pool in self.compress_state_pools
                if pool is not None and pool.ratio == 4
            ),
            *(
                self._state_transfer_buffer(pool)
                for pool in self.indexer_compress_state_pools
                if pool is not None and pool.ratio == 4
            ),
        ]
        if len(buffers) != len(self.get_state_buf_infos()[0]):
            raise RuntimeError("shared PD SWA/C4-state transfer layout is inconsistent")
        return buffers

    def get_owner_sharded_c128_state_transfer_buffers(
        self,
    ) -> list[OwnerShardedTransferBuffer]:
        """Return local-owner C128 state tensors in component pointer order."""
        buffers = [
            self._state_transfer_buffer(pool)
            for pool in self.compress_state_pools
            if pool is not None and pool.ratio == 128
        ]
        if len(buffers) != len(self.get_c128_state_buf_infos()[0]):
            raise RuntimeError("shared PD C128-state transfer layout is inconsistent")
        return buffers

    def _log_shared_family_accounting(self) -> None:
        if self.shared_rank != 0:
            return
        families = [
            self.swa_kv_pool.shared_family,
            self.c4_kv_pool.shared_family,
            self.c128_kv_pool.shared_family,
            self.c4_indexer_kv_pool.shared_family,
            *self.shared_state_families.values(),
        ]
        total_mapped_bytes = 0
        for family in families:
            if family is None:
                continue
            spec = family.storage.spec
            accounting = family.storage.accounting()
            total_mapped_bytes += accounting.mapped_bytes_per_rank
            logger.info(
                "DSV4 Shared family=%s logical_rows_per_layer=%d "
                "logical_blocks_per_layer=%d ownership_granule=%d "
                "minimum_blocks_per_rank=%d physical_blocks_per_rank=%d "
                "mapped_bytes_per_rank=%d alignment_overhead_bytes_per_rank=%d",
                accounting.name,
                spec.logical_rows_per_layer,
                accounting.logical_blocks_per_layer,
                spec.ownership_granule,
                accounting.minimum_blocks_per_rank,
                accounting.physical_blocks_per_rank,
                accounting.mapped_bytes_per_rank,
                accounting.alignment_overhead_bytes_per_rank,
            )
        logger.info(
            "DSV4 Shared aggregate mapped_bytes_per_rank=%d; CUDA VMM virtual "
            "address and handle metadata are driver-managed and excluded",
            total_mapped_bytes,
        )

    def _get_cp_group(self):
        if self.shared_cp_group is None:
            from sglang.srt.runtime_context import get_parallel

            self.shared_cp_group = get_parallel().attn_cp_group
        return self.shared_cp_group

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> DeepSeekV4SingleKVPool:
        if cls is not DeepSeekV4SingleKVPool:
            raise ValueError("DSV4 shared KV cache only supports the CUDA packed pool")
        family_name = (
            "swa"
            if page_size == global_page_size
            else (
                "c4"
                if page_size * 4 == global_page_size
                else (
                    "c128"
                    if page_size * 128 == global_page_size
                    else f"compressed_p{page_size}"
                )
            )
        )
        return SharedDeepSeekV4SingleKVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            shared_rank=self.shared_rank,
            shared_size=self.shared_size,
            shared_cpu_group=self._get_cp_group().cpu_group,
            shared_family_name=family_name,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> DeepSeekV4IndexerPool:
        return SharedDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            shared_rank=self.shared_rank,
            shared_size=self.shared_size,
            shared_cpu_group=self._get_cp_group().cpu_group,
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        del enable_memory_saver
        total_layers = len(self.compression_ratios)
        self.compress_state_pools = [None] * total_layers
        self.indexer_compress_state_pools = [None] * total_layers
        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        c4_layers = sum(ratio == 4 for ratio in stage_ratios)
        c128_layers = sum(ratio == 128 for ratio in stage_ratios)
        cpu_group = self._get_cp_group().cpu_group

        def make_family(
            *,
            name: str,
            ratio: int,
            head_dim: int,
            layer_num: int,
            dtype: torch.dtype,
        ) -> SharedDeepSeekV4Family:
            allocated_rows, _, last_dim = get_compress_state_layout(
                size=self._state_pool_size(ratio),
                ring_size=self.get_ring_size(ratio),
                overlap=ratio == 4,
                head_dim=head_dim,
                ratio=ratio,
                swa_page_size=self.swa_page_size,
            )
            family = SharedDeepSeekV4Family.create(
                name=name,
                logical_size=allocated_rows - 1,
                page_size=ratio,
                layer_num=layer_num,
                dtype=dtype,
                row_shape=(last_dim,),
                rows_per_page=ratio,
                cp_size=self.shared_size,
                cpu_group=cpu_group,
                zero_initialize=False,
            )
            self.shared_state_families[name] = family
            return family

        c4_attn = make_family(
            name="c4_attn_state",
            ratio=4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            layer_num=c4_layers,
            dtype=self.c4_state_dtype,
        )
        c4_indexer = make_family(
            name="c4_indexer_state",
            ratio=4,
            head_dim=self.indexer_head_dim,
            layer_num=c4_layers,
            dtype=self.c4_state_dtype,
        )
        c128_attn = make_family(
            name="c128_attn_state",
            ratio=128,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            layer_num=c128_layers,
            dtype=self.c128_state_dtype,
        )

        c4_id = c128_id = 0
        for layer_id in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[layer_id]
            if ratio == 0:
                continue
            family = c4_attn if ratio == 4 else c128_attn
            family_layer_id = c4_id if ratio == 4 else c128_id
            self.compress_state_pools[layer_id] = SharedCompressStatePool(
                size=self._state_pool_size(ratio),
                ring_size=self.get_ring_size(ratio),
                overlap=ratio == 4,
                head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
                device=self.device,
                enable_memory_saver=False,
                ratio=ratio,
                swa_page_size=self.swa_page_size,
                shared_family=family,
                shared_layer_id=family_layer_id,
                shared_rank=self.shared_rank,
            )
            if ratio == 4:
                self.indexer_compress_state_pools[layer_id] = SharedCompressStatePool(
                    size=self._state_pool_size(4),
                    ring_size=self.get_ring_size(4),
                    overlap=True,
                    head_dim=self.indexer_head_dim,
                    dtype=self.c4_state_dtype,
                    device=self.device,
                    enable_memory_saver=False,
                    ratio=4,
                    swa_page_size=self.swa_page_size,
                    shared_family=c4_indexer,
                    shared_layer_id=c4_id,
                    shared_rank=self.shared_rank,
                )
                c4_id += 1
            else:
                c128_id += 1

        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    def translate_swa_slots_for_read(self, slots: torch.Tensor) -> torch.Tensor:
        return self.swa_kv_pool.translate_slots_for_read(slots)

    def translate_extra_slots_for_read(
        self, layer_id: int, slots: torch.Tensor
    ) -> torch.Tensor:
        pool = self.layer_mapping[layer_id].compress_kv_pool
        assert pool is not None
        return pool.translate_slots_for_read(slots)

    @staticmethod
    def _shared_dequant_params(
        pool: SharedDeepSeekV4SingleKVPool,
    ) -> tuple[int, int]:
        family = pool.shared_family
        assert family is not None
        return family.layout.cp_size, family.layout.pages_per_rank

    def get_swa_shared_dequant_params(self, layer_id: int) -> tuple[int, int]:
        del layer_id
        return self._shared_dequant_params(self.swa_kv_pool)

    def get_extra_shared_dequant_params(self, layer_id: int) -> tuple[int, int]:
        item = self.layer_mapping[layer_id]
        pool = item.compress_kv_pool
        assert isinstance(pool, SharedDeepSeekV4SingleKVPool)
        return self._shared_dequant_params(pool)

    def prepare_indexer_pages_for_read(
        self, pages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.c4_indexer_kv_pool.prepare_pages_for_read(pages)

    def prepare_compressed_pages_for_read(
        self,
        pages: torch.Tensor,
        *,
        single_request: bool = False,
    ) -> tuple[dict[str | int, torch.Tensor], torch.Tensor]:
        assert self.c4_indexer_kv_pool.shared_family is not None
        indexer_layout = self.c4_indexer_kv_pool.shared_family.layout
        if single_request:
            physical_indexer, compact_page_table = (
                _build_single_request_page_stage_plan(indexer_layout, pages)
            )
        else:
            physical_indexer, compact_page_table = (
                self.c4_indexer_kv_pool.prepare_pages_for_read(pages)
            )
        owner = torch.div(
            physical_indexer,
            indexer_layout.pages_per_rank,
            rounding_mode="floor",
        )
        local_page = torch.remainder(physical_indexer, indexer_layout.pages_per_rank)
        logical_pages = local_page * indexer_layout.cp_size + owner

        physical_by_family: dict[str | int, torch.Tensor] = {
            "indexer": physical_indexer
        }
        for ratio, pool in ((4, self.c4_kv_pool), (128, self.c128_kv_pool)):
            assert isinstance(pool, SharedDeepSeekV4SingleKVPool)
            assert pool.shared_family is not None
            layout = pool.shared_family.layout
            assert layout.cp_size == indexer_layout.cp_size
            physical_by_family[ratio] = layout.translate_pages(logical_pages).long()
        return physical_by_family, compact_page_table

    def stage_indexer_pages_with_plan(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        assert item.compress_ratio == 4
        return self.c4_indexer_kv_pool.stage_pages_with_plan(
            item.compress_layer_id, physical_pages
        )

    def prepare_swa_slots_for_read(
        self,
        slots: torch.Tensor,
        *,
        single_request: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if single_request or (slots.ndim == 2 and slots.shape[0] == 1):
            assert self.swa_kv_pool.shared_family is not None
            return _build_single_request_slot_stage_plan(
                self.swa_kv_pool.shared_family.layout,
                slots,
            )
        return self.swa_kv_pool.prepare_slots_for_read(slots)

    def stage_swa_slots_with_plan(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        return self.swa_kv_pool.stage_slots_with_plan(
            self._swa_local_layer_id(layer_id), physical_pages
        )

    def prepare_extra_slots_for_read(
        self,
        layer_id: int,
        slots: torch.Tensor,
        *,
        single_request: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.layer_mapping[layer_id]
        pool = item.compress_kv_pool
        assert pool is not None
        if single_request:
            assert isinstance(pool, SharedDeepSeekV4SingleKVPool)
            assert pool.shared_family is not None
            return _build_single_request_slot_stage_plan(
                pool.shared_family.layout,
                slots,
            )
        return pool.prepare_slots_for_read(slots)

    def stage_extra_slots_with_plan(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        pool = item.compress_kv_pool
        assert pool is not None
        return pool.stage_slots_with_plan(item.compress_layer_id, physical_pages)

    def stage_compressed_pages_with_indexer_plan(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        """Gather C4/C128 KV with their common compact full-page plan."""
        item = self.layer_mapping[layer_id]
        pool = item.compress_kv_pool
        assert item.compress_ratio in (4, 128)
        assert isinstance(pool, SharedDeepSeekV4SingleKVPool)
        return pool.stage_slots_with_plan(item.compress_layer_id, physical_pages)

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        local_layer_id = self._swa_local_layer_id(layer_id)
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.local_kv_buffer[local_layer_id],
            page_size=self.swa_kv_pool.page_size,
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
        )

    def get_compressor_write_info(
        self, layer_id: int, *, is_indexer: bool
    ) -> tuple[torch.Tensor, int, int]:
        """Return this rank's owner-local V2 compressor write target.

        Compressor inputs are all-gathered across the attention CP group before
        compression, so every rank computes the complete output.  Keep exactly
        one writer per logical page; direct peer writes here would create eight
        redundant, numerically non-identical writers for the same FP8/state row.
        """
        item = self.layer_mapping[layer_id]
        compress_layer_id = item.compress_layer_id
        assert compress_layer_id is not None
        if is_indexer:
            pool = self.c4_indexer_kv_pool
            local_layer_id = compress_layer_id - pool.start_layer
            cache = pool.local_index_k_with_scale_buffer[local_layer_id]
        else:
            pool = item.compress_kv_pool
            assert pool is not None
            cache = pool.local_kv_buffer[compress_layer_id]
        return cache, self.shared_rank, self.shared_size

    def synchronize_shared_writes(self) -> None:
        assert self.shared_write_publisher is not None
        self.shared_write_publisher.publish()

    def get_kv_size_bytes(self) -> int:
        families = [
            self.swa_kv_pool.shared_family,
            self.c4_kv_pool.shared_family,
            self.c128_kv_pool.shared_family,
            self.c4_indexer_kv_pool.shared_family,
        ]
        families.extend(self.shared_state_families.values())
        return sum(family.physical_bytes for family in families if family is not None)
