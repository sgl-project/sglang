"""Framework policy and storage for FlashMLA Shared-KV demand caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.shared_kv.demand_cache import (
    DEMAND_CACHE_EPOCH_BYTES,
    DEMAND_CACHE_TAG_BYTES,
)

FLASHMLA_SHARED_ROW_BYTES = 656
FLASHMLA_SHARED_PREFILL_ROWS = 1 << 18
FLASHMLA_SHARED_DECODE_ROWS_PER_REQUEST = 1 << 12
_FLASHMLA_SHARED_DECODE_CALLS_PER_GENERATION = 64
_FLASHMLA_SHARED_DEMAND_CACHE_MAX_EPOCH = (1 << 31) - 1

logger = logging.getLogger(__name__)


@dataclass
class SharedMLACurrentRows:
    """Graph-stable current-row shadow owned by the DSA cache adapter."""

    encoded_rows: torch.Tensor
    physical_rows: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        max_query_rows: int,
        max_current_rows: int,
        device: torch.device | str,
    ) -> SharedMLACurrentRows:
        if max_query_rows <= 0 or max_current_rows <= 0:
            raise ValueError("Shared MLA current-row capacities must be positive")
        return cls(
            encoded_rows=torch.empty(
                (max_query_rows, max_current_rows, FLASHMLA_SHARED_ROW_BYTES),
                dtype=torch.uint8,
                device=device,
            ),
            physical_rows=torch.full(
                (max_query_rows, max_current_rows),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            counts=torch.zeros((max_query_rows,), dtype=torch.int32, device=device),
        )


def get_indexer_pool_cache_layer_ids(
    config: object, start_layer: int, end_layer: int
) -> tuple[int, ...]:
    """Return layers that produce fresh Indexer TopK and need local pages."""
    if start_layer < 0 or end_layer < start_layer:
        raise ValueError("Indexer Pool Demand Cache layer range is invalid")
    cli_factor = getattr(config, "cli_factor", 1) or 1
    if cli_factor > 1:
        return tuple(
            layer_id
            for layer_id in range(start_layer, end_layer)
            if layer_id % cli_factor == 0
        )

    from sglang.srt.configs.model_config import dsa_layer_skips_topk

    return tuple(
        layer_id
        for layer_id in range(start_layer, end_layer)
        if not dsa_layer_skips_topk(config, layer_id)
    )


def get_indexer_pool_cache_bytes(
    *,
    num_tokens: int,
    page_size: int,
    num_layers: int,
    indexer_bytes_per_token: int,
) -> int:
    """Exact allocation for token-pool page data, tags, and epochs."""
    if min(num_tokens, num_layers) < 0 or page_size <= 0:
        raise ValueError("Indexer Pool Demand Cache dimensions must be non-negative")
    if indexer_bytes_per_token <= 0:
        raise ValueError("Indexer Pool Demand Cache row size must be positive")
    if num_layers == 0:
        return 0
    logical_pages = (num_tokens + page_size + 1) // page_size
    page_bytes = page_size * indexer_bytes_per_token
    return (
        num_layers * logical_pages * (page_bytes + DEMAND_CACHE_TAG_BYTES)
        + num_layers * DEMAND_CACHE_EPOCH_BYTES
    )


@dataclass
class FlashMLASharedDecodeGeneration:
    """One graph-stable generation shared by all Decode layer caches."""

    tensor: torch.Tensor
    tags: torch.Tensor
    max_generation: int = _FLASHMLA_SHARED_DEMAND_CACHE_MAX_EPOCH
    calls_per_generation: int = 1
    generation: int = 1
    calls_in_generation: int = 0

    @classmethod
    def create(
        cls,
        *,
        device: torch.device | str,
        tags: torch.Tensor,
        max_generation: int = _FLASHMLA_SHARED_DEMAND_CACHE_MAX_EPOCH,
        calls_per_generation: int = 1,
    ) -> FlashMLASharedDecodeGeneration:
        if max_generation < 2:
            raise ValueError("Decode cache generation limit must be at least 2")
        if calls_per_generation <= 0:
            raise ValueError("Decode cache generation interval must be positive")
        return cls(
            tensor=torch.ones((), dtype=torch.int32, device=device),
            tags=tags,
            max_generation=max_generation,
            calls_per_generation=calls_per_generation,
        )

    def advance(self) -> torch.Tensor:
        self.calls_in_generation += 1
        if self.calls_in_generation < self.calls_per_generation:
            return self.tensor
        self.calls_in_generation = 0
        if self.generation >= self.max_generation:
            self.tags.zero_()
            self.generation = 1
        else:
            self.generation += 1
        self.tensor.fill_(self.generation)
        return self.tensor


@dataclass
class FlashMLASharedDecodeSlotLifecycle:
    """Invalidate a fixed request-local cache slice when its request changes."""

    tags: torch.Tensor
    installed_generations: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        tags: torch.Tensor,
        num_request_slots: int,
    ) -> FlashMLASharedDecodeSlotLifecycle:
        if tags.dim() != 3 or tags.shape[1] != num_request_slots:
            raise ValueError("Decode cache tags must be [layers, request_slots, rows]")
        if num_request_slots <= 0 or tags.shape[2] <= 0:
            raise ValueError("Decode request-slot count must be positive")
        return cls(
            tags=tags,
            installed_generations=torch.zeros(
                num_request_slots + 1, dtype=torch.int64, device="cpu"
            ),
        )

    def refresh(
        self,
        *,
        active_request_slots: torch.Tensor,
        request_generations: torch.Tensor,
    ) -> list[int]:
        slots = active_request_slots.to(device="cpu", dtype=torch.int64).view(-1)
        generations = request_generations.to(device="cpu", dtype=torch.int64).view(-1)
        if slots.shape != generations.shape:
            raise ValueError("Request slots and generations must have equal shape")
        if slots.numel() and (
            int(slots.min()) <= 0
            or int(slots.max()) >= self.installed_generations.numel()
        ):
            raise ValueError("Decode cache request slot is out of range")

        active_generations: dict[int, int] = {}
        for slot, generation in zip(slots.tolist(), generations.tolist()):
            previous = active_generations.setdefault(slot, generation)
            if previous != generation:
                raise ValueError("One request slot has conflicting generations")

        changed_slots = []
        for slot in range(1, self.installed_generations.numel()):
            if slot not in active_generations:
                self.installed_generations[slot] = 0
                continue
            if self.installed_generations[slot] != active_generations[slot]:
                changed_slots.append(slot)
            self.installed_generations[slot] = active_generations[slot]

        if changed_slots:
            self.tags.index_fill_(
                1,
                torch.tensor(
                    [slot - 1 for slot in changed_slots],
                    dtype=torch.int64,
                    device=self.tags.device,
                ),
                0,
            )
        return changed_slots


@dataclass
class SlotDemandCache:
    """One-layer scratch that turns repeated peer-VMM rows into local hits."""

    row_cache: torch.Tensor
    tags: torch.Tensor
    local_row_begin: int
    local_row_end: int
    rows_per_request: int
    num_request_slots: int
    generation_tensor: Optional[torch.Tensor] = None
    epoch: int = 0

    def next_call_kwargs(
        self,
        *,
        persistent: bool = False,
        request_slots: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        if not persistent:
            self.epoch += 1
        elif self.epoch == 0:
            # Persistent Decode tags retain their generation until the owning
            # request slot is allocated to a new request generation.
            self.epoch = 1
        if self.epoch > _FLASHMLA_SHARED_DEMAND_CACHE_MAX_EPOCH:
            # The clear and the following kernel are enqueued on the same stream.
            # Epoch wrap is practically unreachable, but retaining an explicit
            # reset makes stale READY tags impossible by construction.
            self.tags.zero_()
            self.epoch = 1
        if self.num_request_slots == 1:
            request_slots = None
        kwargs = {
            "shared_kv_row_cache": self.row_cache,
            "shared_kv_cache_tags": self.tags,
            "shared_kv_request_slots": request_slots,
            "shared_kv_cache_rows_per_request": self.rows_per_request,
            "shared_kv_num_request_slots": self.num_request_slots,
            "shared_kv_cache_epoch": self.epoch,
            "shared_kv_cache_generation_tensor": self.generation_tensor,
            "shared_kv_local_row_begin": self.local_row_begin,
            "shared_kv_local_row_end": self.local_row_end,
        }
        return kwargs


def _build_slot_demand_cache(
    pool,
    *,
    device: torch.device | str,
    enabled: bool,
    rows_per_request: int = FLASHMLA_SHARED_PREFILL_ROWS,
    num_request_slots: int = 1,
    tags: Optional[torch.Tensor] = None,
    generation_tensor: Optional[torch.Tensor] = None,
) -> Optional[SlotDemandCache]:
    layout = getattr(pool, "main_layout", None)
    if not enabled or layout is None or layout.cp_size <= 1:
        return None
    if layout.page_size != 64:
        raise ValueError(
            "FlashMLA Shared-KV demand cache requires page size 64, got "
            f"{layout.page_size}"
        )
    if rows_per_request <= 0 or rows_per_request & (rows_per_request - 1):
        raise ValueError(
            "FlashMLA Shared-KV rows per request must be a power of two, got "
            f"{rows_per_request}"
        )
    if num_request_slots <= 0:
        raise ValueError(
            "FlashMLA Shared-KV request-slot count must be positive, got "
            f"{num_request_slots}"
        )
    shared_rank = getattr(pool, "shared_rank", None)
    if shared_rank is None or not 0 <= shared_rank < layout.cp_size:
        raise ValueError(f"invalid Shared-KV rank: {shared_rank}")

    if layout.local_pages_per_layer is None:
        raise ValueError("Shared-KV layout is missing local_pages_per_layer")
    rank_stride_rows = layout.pages_per_rank * layout.page_size
    local_rows_per_layer = layout.local_pages_per_layer * layout.page_size
    local_row_begin = shared_rank * rank_stride_rows
    if tags is None:
        tags = torch.zeros(
            (num_request_slots, rows_per_request),
            dtype=torch.int64,
            device=device,
        )
    elif tags.shape != (num_request_slots, rows_per_request):
        raise ValueError(
            "FlashMLA Shared-KV tag storage has shape "
            f"{tuple(tags.shape)}, expected "
            f"{(num_request_slots, rows_per_request)}"
        )
    return SlotDemandCache(
        row_cache=torch.empty(
            (num_request_slots * rows_per_request, FLASHMLA_SHARED_ROW_BYTES),
            dtype=torch.uint8,
            device=device,
        ),
        tags=tags,
        local_row_begin=local_row_begin,
        local_row_end=local_row_begin + local_rows_per_layer,
        rows_per_request=rows_per_request,
        num_request_slots=num_request_slots,
        generation_tensor=generation_tensor,
    )


def _build_decode_slot_demand_caches(
    pool,
    *,
    device: torch.device | str,
    num_layers: int,
    num_request_slots: int,
    rows_per_request: int = FLASHMLA_SHARED_DECODE_ROWS_PER_REQUEST,
) -> tuple[
    list[SlotDemandCache],
    FlashMLASharedDecodeGeneration,
    FlashMLASharedDecodeSlotLifecycle,
]:
    tags = torch.zeros(
        (num_layers, num_request_slots, rows_per_request),
        dtype=torch.int64,
        device=device,
    )
    generation = FlashMLASharedDecodeGeneration.create(
        device=device,
        tags=tags,
        calls_per_generation=_FLASHMLA_SHARED_DECODE_CALLS_PER_GENERATION,
    )
    lifecycle = FlashMLASharedDecodeSlotLifecycle.create(
        tags=tags,
        num_request_slots=num_request_slots,
    )
    caches: list[SlotDemandCache] = []
    for layer_id in range(num_layers):
        cache = _build_slot_demand_cache(
            pool,
            device=device,
            enabled=True,
            rows_per_request=rows_per_request,
            num_request_slots=num_request_slots,
            tags=tags[layer_id],
            generation_tensor=generation.tensor,
        )
        assert cache is not None
        caches.append(cache)
    return caches, generation, lifecycle


def _expand_flashmla_shared_cache_request_slots(
    req_pool_indices: torch.Tensor,
    *,
    num_query_rows: int,
) -> torch.Tensor:
    if req_pool_indices.dim() != 1 or req_pool_indices.numel() == 0:
        raise ValueError("FlashMLA Shared-KV request slots must be non-empty 1D")
    request_count = req_pool_indices.numel()
    repeats, remainder = divmod(num_query_rows, request_count)
    if repeats <= 0 or remainder:
        raise ValueError(
            "FlashMLA Shared-KV query rows must be a positive multiple of "
            f"request count, got rows={num_query_rows} requests={request_count}"
        )
    if repeats == 1:
        return req_pool_indices
    return req_pool_indices.repeat_interleave(repeats)


def resolve_flashmla_shared_max_current_rows(
    speculative_num_draft_tokens: int | None,
) -> int:
    """Resolve the graph-stable current-row width for Decode/verify."""
    return max(1, speculative_num_draft_tokens or 0)


def get_flashmla_shared_demand_workspace_bytes(
    *,
    num_layers: int,
    num_request_slots: int,
    max_current_rows: int,
) -> int:
    """Return graph-stable HBM allocated outside the token KV pool."""
    if num_layers <= 0 or num_request_slots <= 0 or max_current_rows <= 0:
        raise ValueError("Shared demand-cache dimensions must be positive")
    if max_current_rows > 4:
        raise ValueError(
            "FlashMLA Shared-KV supports at most four current rows per request"
        )

    row_and_tag_bytes = FLASHMLA_SHARED_ROW_BYTES + DEMAND_CACHE_TAG_BYTES
    prefill_bytes = FLASHMLA_SHARED_PREFILL_ROWS * row_and_tag_bytes
    decode_bytes = (
        num_layers
        * num_request_slots
        * FLASHMLA_SHARED_DECODE_ROWS_PER_REQUEST
        * row_and_tag_bytes
    )

    max_query_rows = num_request_slots * max_current_rows
    current_rows_per_layer = (
        max_query_rows * max_current_rows * FLASHMLA_SHARED_ROW_BYTES
        + max_query_rows * max_current_rows * 4
        + max_query_rows * 4
    )
    current_row_bytes = num_layers * current_rows_per_layer

    generation_bytes = DEMAND_CACHE_EPOCH_BYTES
    return prefill_bytes + decode_bytes + current_row_bytes + generation_bytes


@dataclass
class DSAFlashMLADemandCacheManager:
    """Own all framework-side Main-KV Demand Cache policy state."""

    prefill_cache: Optional[SlotDemandCache]
    decode_caches: list[SlotDemandCache]
    decode_generation: Optional[FlashMLASharedDecodeGeneration]
    decode_lifecycle: Optional[FlashMLASharedDecodeSlotLifecycle]
    current_rows_by_layer: list[SharedMLACurrentRows]
    max_current_rows: int

    @property
    def allocated_bytes(self) -> int:
        """HBM bytes owned by the graph-stable Main-KV Demand workspace."""
        total = 0
        if self.prefill_cache is not None:
            total += self.prefill_cache.row_cache.nbytes
            total += self.prefill_cache.tags.nbytes
        for cache in self.decode_caches:
            total += cache.row_cache.nbytes
            total += cache.tags.nbytes
        if self.decode_generation is not None:
            total += self.decode_generation.tensor.nbytes
        for current_rows in self.current_rows_by_layer:
            total += current_rows.encoded_rows.nbytes
            total += current_rows.physical_rows.nbytes
            total += current_rows.counts.nbytes
        return total

    @classmethod
    def create(
        cls,
        *,
        pool,
        device: torch.device | str,
        enable_prefill: bool,
        enable_decode: bool,
        enable_current_rows: bool,
        num_layers: int,
        num_request_slots: int,
        max_current_rows: int,
    ) -> DSAFlashMLADemandCacheManager:
        prefill_cache = _build_slot_demand_cache(
            pool,
            device=device,
            enabled=enable_prefill,
        )
        decode_caches: list[SlotDemandCache] = []
        decode_generation = None
        decode_lifecycle = None
        if prefill_cache is not None and enable_decode:
            (
                decode_caches,
                decode_generation,
                decode_lifecycle,
            ) = _build_decode_slot_demand_caches(
                pool,
                device=device,
                num_layers=num_layers,
                num_request_slots=num_request_slots,
            )
            logger.info(
                "FlashMLA Shared-KV Slot Decode Demand Cache: "
                "layers=%s request_slots=%s rows_per_request=%s bytes=%s",
                len(decode_caches),
                num_request_slots,
                FLASHMLA_SHARED_DECODE_ROWS_PER_REQUEST,
                sum(
                    cache.row_cache.nbytes + cache.tags.nbytes
                    for cache in decode_caches
                ),
            )

        current_rows_by_layer: list[SharedMLACurrentRows] = []
        if decode_caches and enable_current_rows:
            max_query_rows = num_request_slots * max_current_rows
            current_rows_by_layer = [
                SharedMLACurrentRows.create(
                    max_query_rows=max_query_rows,
                    max_current_rows=max_current_rows,
                    device=device,
                )
                for _ in range(num_layers)
            ]

        if prefill_cache is not None:
            logger.info(
                "FlashMLA Shared-KV Slot Prefill Demand Cache: "
                "rows=%s bytes=%s local_rows=[%s,%s)",
                prefill_cache.row_cache.shape[0],
                prefill_cache.row_cache.nbytes + prefill_cache.tags.nbytes,
                prefill_cache.local_row_begin,
                prefill_cache.local_row_end,
            )
        if current_rows_by_layer:
            shadow = current_rows_by_layer[0]
            bytes_per_layer = (
                shadow.encoded_rows.nbytes
                + shadow.physical_rows.nbytes
                + shadow.counts.nbytes
            )
            logger.info(
                "FlashMLA Shared-KV current-row shadow: max_query_rows=%s "
                "max_current_rows=%s layers=%s bytes_per_layer=%s total_bytes=%s",
                num_request_slots * max_current_rows,
                max_current_rows,
                num_layers,
                bytes_per_layer,
                bytes_per_layer * num_layers,
            )
        return cls(
            prefill_cache=prefill_cache,
            decode_caches=decode_caches,
            decode_generation=decode_generation,
            decode_lifecycle=decode_lifecycle,
            current_rows_by_layer=current_rows_by_layer,
            max_current_rows=max_current_rows,
        )

    @property
    def has_decode_cache(self) -> bool:
        return bool(self.decode_caches)

    def current_rows_per_request(self, *, target_verify: bool, decode: bool) -> int:
        if not self.current_rows_by_layer or not (target_verify or decode):
            return 0
        return self.max_current_rows if target_verify else 1

    def advance_decode_generation(self) -> None:
        if self.decode_generation is not None:
            self.decode_generation.advance()

    def refresh_decode_requests(
        self,
        *,
        active_request_slots: torch.Tensor,
        request_generations: torch.Tensor,
    ) -> None:
        if self.decode_lifecycle is not None:
            self.decode_lifecycle.refresh(
                active_request_slots=active_request_slots,
                request_generations=request_generations,
            )

    def cache_for_layer(
        self,
        *,
        local_layer_id: int,
        persistent: bool,
    ) -> tuple[Optional[SlotDemandCache], bool]:
        if persistent and self.decode_caches:
            return self.decode_caches[local_layer_id], True
        return self.prefill_cache, False
