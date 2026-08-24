"""Batch-level attention-parallel mode selection.

The module intentionally has no model, scheduler, or CP-package imports so it
can be used by both ``ScheduleBatch`` and ``ForwardBatch`` without cycles.
Process groups and weight residency remain static; only the execution axis is
selected per finalized batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class AttnParallelMode(IntEnum):
    """Attention execution mode stamped once on a scheduler batch."""

    TP = 0
    CP = 1
    DCP = 2

    @property
    def is_cp(self) -> bool:
        return self is AttnParallelMode.CP

    @property
    def is_dcp(self) -> bool:
        return self is AttnParallelMode.DCP


class KvResidency(IntEnum):
    """Physical MLA KV placement for a request or server-level PoC."""

    REPLICATED = 0
    STRIPED = 1
    TRANSITIONING = 2


@dataclass(frozen=True)
class AttnParallelDecision:
    mode: AttnParallelMode
    veto_reason: str | None = None


def resolve_kv_residency(
    parallel: Any, forward_batch: Any | None = None
) -> KvResidency:
    if forward_batch is not None:
        residency = getattr(forward_batch, "kv_residency", None)
        if residency is not None:
            return KvResidency(residency)
    if bool(getattr(parallel, "dynamic_attn_parallel_enable_dcp", False)):
        # Functional dynamic-DCP PoC: retain a complete local KV view so TP can
        # be selected on the next batch without an online all-gather.
        return KvResidency.REPLICATED
    if bool(getattr(parallel, "dcp_enabled", False)):
        return KvResidency.STRIPED
    return KvResidency.REPLICATED


def kv_storage_dcp_size(parallel: Any, forward_batch: Any | None = None) -> int:
    return (
        int(getattr(parallel, "attn_dcp_size", 1))
        if resolve_kv_residency(parallel, forward_batch) is KvResidency.STRIPED
        else 1
    )


def strategy_min_tokens(strategy: str | None, cp_size: int) -> tuple[int, int]:
    """Return ``(batch_floor, per_request_floor)`` for a CP layout."""

    if strategy == "zigzag":
        return 2 * cp_size, 2 * cp_size
    if strategy == "interleave":
        return cp_size, 1
    return 0, 0


def select_attn_parallel_mode(
    *,
    forward_mode: Any,
    extend_seq_lens: Sequence[int] | None,
    num_tokens: int,
    strategy: str | None,
    cp_size: int,
    min_prefill_tokens: int | None = None,
    allow_mixed: bool = False,
    enable_decode_dcp: bool = False,
    dcp_size: int = 1,
    decode_seq_lens: Sequence[int] | None = None,
    min_decode_context: int = 8192,
    kv_residency: KvResidency | None = None,
) -> AttnParallelDecision:
    """Select TP or prefill CP from immutable, rank-consistent batch facts."""

    mode_name = getattr(forward_mode, "name", str(forward_mode))
    if mode_name == "DECODE":
        if not enable_decode_dcp or dcp_size <= 1:
            if kv_residency is KvResidency.STRIPED:
                return AttnParallelDecision(AttnParallelMode.DCP)
            return AttnParallelDecision(AttnParallelMode.TP, "dcp_disabled")
        if kv_residency is KvResidency.STRIPED:
            return AttnParallelDecision(AttnParallelMode.DCP)
        max_context = max(
            (int(length) for length in (decode_seq_lens or ())), default=0
        )
        if max_context < min_decode_context:
            return AttnParallelDecision(AttnParallelMode.TP, "short_context")
        return AttnParallelDecision(AttnParallelMode.DCP)

    if kv_residency is KvResidency.STRIPED:
        # Compact KV is not a complete local prefix, so the current CP-v2
        # paged-attention path cannot consume it directly. The Aiter DCP
        # prefill path assembles the full prefix while keeping TP heads.
        return AttnParallelDecision(AttnParallelMode.TP, "striped_prefill")

    if cp_size <= 1 or strategy not in ("zigzag", "interleave"):
        return AttnParallelDecision(AttnParallelMode.TP, "disabled")

    is_cp_extend = bool(
        getattr(forward_mode, "is_context_parallel_extend", lambda: False)()
    )
    if not is_cp_extend:
        return AttnParallelDecision(AttnParallelMode.TP, "not_extend")
    if mode_name == "MIXED" and not allow_mixed:
        return AttnParallelDecision(AttnParallelMode.TP, "mixed")
    if bool(getattr(forward_mode, "is_target_verify", lambda: False)()) or bool(
        getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
    ):
        return AttnParallelDecision(AttnParallelMode.TP, "speculative")

    lengths = (
        [int(length) for length in extend_seq_lens]
        if extend_seq_lens is not None
        else None
    )
    logical_tokens = sum(lengths) if lengths is not None else int(num_tokens)
    batch_floor, request_floor = strategy_min_tokens(strategy, cp_size)
    if logical_tokens < batch_floor:
        return AttnParallelDecision(AttnParallelMode.TP, "too_few_tokens")

    threshold = max(batch_floor, int(min_prefill_tokens or batch_floor))
    if logical_tokens < threshold:
        return AttnParallelDecision(AttnParallelMode.TP, "below_threshold")

    if (
        strategy == "zigzag"
        and lengths is not None
        and any(length < request_floor for length in lengths)
    ):
        return AttnParallelDecision(AttnParallelMode.TP, "short_request")

    return AttnParallelDecision(AttnParallelMode.CP)
