"""Lower Mooncake plans against SGLang request-local page mappings."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mooncake.reshard.kv_cache import (
    KVCachePreparedTransferEdge,
    KVCachePreparedTransferPlan,
)


@dataclass(frozen=True)
class KVCacheTransferBatch:
    """One native transfer-engine batch targeting a single endpoint."""

    endpoint: str
    source_addresses: tuple[int, ...]
    target_addresses: tuple[int, ...]
    sizes: tuple[int, ...]


def _page_runs(
    source_page_ids: Iterable[int],
    target_page_ids: Iterable[int],
    *,
    page_size: int,
    first_page_offset: int,
    token_count: int,
) -> tuple[tuple[tuple[int, int, int], ...], int, int]:
    required_pages = (first_page_offset + token_count + page_size - 1) // page_size
    source_pages = tuple(int(page_id) for page_id in source_page_ids)
    target_pages = tuple(int(page_id) for page_id in target_page_ids)
    if len(source_pages) != required_pages or len(target_pages) != required_pages:
        raise ValueError("page_ids length does not cover the logical token span")
    if any(page_id < 0 for page_id in (*source_pages, *target_pages)):
        raise ValueError("page_ids must be non-negative")

    remaining = token_count
    page_offset = first_page_offset
    runs: list[tuple[int, int, int]] = []
    max_source_slot = -1
    max_target_slot = -1
    for source_page, target_page in zip(source_pages, target_pages, strict=True):
        run_tokens = min(page_size - page_offset, remaining)
        source_slot = source_page * page_size + page_offset
        target_slot = target_page * page_size + page_offset
        max_source_slot = max(max_source_slot, source_slot + run_tokens - 1)
        max_target_slot = max(max_target_slot, target_slot + run_tokens - 1)
        if (
            runs
            and runs[-1][0] + runs[-1][2] == source_slot
            and runs[-1][1] + runs[-1][2] == target_slot
        ):
            previous = runs[-1]
            runs[-1] = (previous[0], previous[1], previous[2] + run_tokens)
        else:
            runs.append((source_slot, target_slot, run_tokens))
        remaining -= run_tokens
        page_offset = 0
    return tuple(runs), max_source_slot, max_target_slot


def _validate_capacity(
    edge: KVCachePreparedTransferEdge,
    max_source_slot: int,
    max_target_slot: int,
) -> None:
    source_end = (
        max_source_slot * edge.source_row_stride
        + edge.source_head_offset_bytes
        + edge.nbytes
    )
    target_end = (
        max_target_slot * edge.target_row_stride
        + edge.target_head_offset_bytes
        + edge.nbytes
    )
    if source_end > edge.source_capacity:
        raise ValueError("source page map exceeds KV buffer capacity")
    if target_end > edge.target_capacity:
        raise ValueError("target page map exceeds KV buffer capacity")


def _lower_full_rows(
    plan: KVCachePreparedTransferPlan,
    runs: tuple[tuple[int, int, int], ...],
    max_source_slot: int,
    max_target_slot: int,
) -> tuple[KVCacheTransferBatch, ...]:
    pending: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for edge in plan.edges:
        _validate_capacity(edge, max_source_slot, max_target_slot)
        sources, targets, sizes = pending.setdefault(edge.endpoint, ([], [], []))
        for source_slot, target_slot, run_tokens in runs:
            sources.append(
                edge.source_base_address + source_slot * edge.source_row_stride
            )
            targets.append(
                edge.target_base_address + target_slot * edge.target_row_stride
            )
            sizes.append(run_tokens * edge.nbytes)
    return tuple(
        KVCacheTransferBatch(endpoint, tuple(sources), tuple(targets), tuple(sizes))
        for endpoint, (sources, targets, sizes) in pending.items()
    )


def _lower_general(
    plan: KVCachePreparedTransferPlan,
    runs: tuple[tuple[int, int, int], ...],
    max_source_slot: int,
    max_target_slot: int,
    max_batch_operations: int,
) -> tuple[KVCacheTransferBatch, ...]:
    source_slots: list[int] = []
    target_slots: list[int] = []
    for source_start, target_start, run_tokens in runs:
        source_slots.extend(range(source_start, source_start + run_tokens))
        target_slots.extend(range(target_start, target_start + run_tokens))

    pending: dict[tuple[str, int | None], tuple[list[int], list[int], list[int]]] = {}
    batches: list[KVCacheTransferBatch] = []

    def flush(key: tuple[str, int | None]) -> None:
        sources, targets, sizes = pending[key]
        if not sizes:
            return
        batches.append(
            KVCacheTransferBatch(key[0], tuple(sources), tuple(targets), tuple(sizes))
        )
        sources.clear()
        targets.clear()
        sizes.clear()

    for edge in plan.edges:
        _validate_capacity(edge, max_source_slot, max_target_slot)
        batch_limit = None if edge.is_full_row else max_batch_operations
        key = (edge.endpoint, batch_limit)
        sources, targets, sizes = pending.setdefault(key, ([], [], []))
        if edge.is_full_row:
            for source_slot, target_slot, run_tokens in runs:
                sources.append(
                    edge.source_base_address + source_slot * edge.source_row_stride
                )
                targets.append(
                    edge.target_base_address + target_slot * edge.target_row_stride
                )
                sizes.append(run_tokens * edge.nbytes)
            continue
        for source_slot, target_slot in zip(source_slots, target_slots, strict=True):
            sources.append(
                edge.source_base_address
                + source_slot * edge.source_row_stride
                + edge.source_head_offset_bytes
            )
            targets.append(
                edge.target_base_address
                + target_slot * edge.target_row_stride
                + edge.target_head_offset_bytes
            )
            sizes.append(edge.nbytes)
            if len(sizes) == max_batch_operations:
                flush(key)
    for key in pending:
        flush(key)
    return tuple(batches)


def lower_kv_cache_transfer(
    prepared_plan: KVCachePreparedTransferPlan,
    source_page_ids: Iterable[int],
    target_page_ids: Iterable[int],
    *,
    token_start: int,
    token_count: int,
    max_batch_operations: int = 1024,
) -> tuple[KVCacheTransferBatch, ...]:
    """Bind one logical token span without expanding full-row pages per token."""

    if not isinstance(prepared_plan, KVCachePreparedTransferPlan):
        raise TypeError("prepared_plan must be a KVCachePreparedTransferPlan")
    if max_batch_operations <= 0:
        raise ValueError("max_batch_operations must be positive")
    if token_count <= 0:
        raise ValueError("token_count must be positive")

    runs, max_source_slot, max_target_slot = _page_runs(
        source_page_ids,
        target_page_ids,
        page_size=prepared_plan.page_size,
        first_page_offset=token_start % prepared_plan.page_size,
        token_count=token_count,
    )
    if all(edge.is_full_row for edge in prepared_plan.edges):
        return _lower_full_rows(prepared_plan, runs, max_source_slot, max_target_slot)
    return _lower_general(
        prepared_plan,
        runs,
        max_source_slot,
        max_target_slot,
        max_batch_operations,
    )


__all__ = ["KVCacheTransferBatch", "lower_kv_cache_transfer"]
