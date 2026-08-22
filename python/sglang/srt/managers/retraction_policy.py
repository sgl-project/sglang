from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Optional, Sequence


@dataclass(frozen=True)
class BackupCostCandidate:
    """CPU-only inputs used to choose a decode-retraction victim."""

    index: int
    backup_tokens: int
    estimated_relief: int
    priority: Optional[int] = None


def compute_decode_shortfall(required_tokens: int, available_tokens: int) -> int:
    return max(0, required_tokens - available_tokens)


def make_backup_cost_candidate(
    *,
    index: int,
    sequence_length: int,
    kv_allocated_len: int,
    next_decode_tokens: int,
    page_size: int,
    priority: Optional[int] = None,
) -> BackupCostCandidate:
    """Build a logical target-KV cost proxy for deterministic selection.

    Page-aligned tokens are monotonic with target-only backup bytes. Specialized
    sidecars and cache layouts can change the absolute physical byte count.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    unaligned_backup_tokens = max(sequence_length - 1, 0)
    backup_tokens = (unaligned_backup_tokens + page_size - 1) // page_size * page_size
    return BackupCostCandidate(
        index=index,
        backup_tokens=backup_tokens,
        estimated_relief=kv_allocated_len + next_decode_tokens,
        priority=priority,
    )


def select_single_sufficient_victim(
    candidates: Sequence[BackupCostCandidate], shortfall: int
) -> Optional[BackupCostCandidate]:
    sufficient = [
        candidate for candidate in candidates if candidate.estimated_relief >= shortfall
    ]
    if not sufficient:
        return None
    return min(
        sufficient,
        key=lambda candidate: (
            candidate.backup_tokens,
            -candidate.estimated_relief,
            candidate.index,
        ),
    )


def _compare_transfer_efficiency(
    lhs: BackupCostCandidate, rhs: BackupCostCandidate
) -> int:
    """Compare cost/relief ratios using integers for cross-rank stability."""
    lhs_relief = max(lhs.estimated_relief, 0)
    rhs_relief = max(rhs.estimated_relief, 0)
    if lhs_relief == 0 or rhs_relief == 0:
        if lhs_relief != rhs_relief:
            return 1 if lhs_relief == 0 else -1
    else:
        cross_lhs = lhs.backup_tokens * rhs_relief
        cross_rhs = rhs.backup_tokens * lhs_relief
        if cross_lhs != cross_rhs:
            return -1 if cross_lhs < cross_rhs else 1

    lhs_key = (lhs.backup_tokens, -lhs.estimated_relief, lhs.index)
    rhs_key = (rhs.backup_tokens, -rhs.estimated_relief, rhs.index)
    return (lhs_key > rhs_key) - (lhs_key < rhs_key)


def select_greedy_victims(
    candidates: Sequence[BackupCostCandidate], shortfall: int
) -> list[BackupCostCandidate]:
    selected = []
    covered = 0
    for candidate in sorted(candidates, key=cmp_to_key(_compare_transfer_efficiency)):
        selected.append(candidate)
        covered += max(candidate.estimated_relief, 0)
        if covered >= shortfall:
            break
    return selected


def _priority_tier_key(
    priority: Optional[int], schedule_low_priority_values_first: bool
) -> tuple[int, int]:
    # A missing priority is least preferred in the existing priority policy.
    if priority is None:
        return (0, 0)
    if schedule_low_priority_values_first:
        return (1, -priority)
    return (1, priority)


def select_backup_cost_victims(
    candidates: Sequence[BackupCostCandidate],
    shortfall: int,
    *,
    respect_priority: bool = False,
    schedule_low_priority_values_first: bool = False,
) -> list[BackupCostCandidate]:
    """Choose the estimated minimum-transfer victim prefix for a shortfall.

    Priority-aware selection exhausts less-preferred tiers before considering a
    more-preferred tier. Within a tier, a single sufficient request is the fast
    path; otherwise deterministic cost/relief greedy selection is used.
    """
    if shortfall <= 0 or not candidates:
        return []

    if respect_priority:
        tier_keys = sorted(
            {
                _priority_tier_key(
                    candidate.priority, schedule_low_priority_values_first
                )
                for candidate in candidates
            }
        )
    else:
        tier_keys = [(0, 0)]

    selected = []
    remaining_shortfall = shortfall
    for tier_key in tier_keys:
        tier = [
            candidate
            for candidate in candidates
            if not respect_priority
            or _priority_tier_key(
                candidate.priority, schedule_low_priority_values_first
            )
            == tier_key
        ]
        single = select_single_sufficient_victim(tier, remaining_shortfall)
        tier_selected = (
            [single]
            if single is not None
            else select_greedy_victims(tier, remaining_shortfall)
        )
        selected.extend(tier_selected)
        remaining_shortfall -= sum(
            max(candidate.estimated_relief, 0) for candidate in tier_selected
        )
        if remaining_shortfall <= 0:
            break
    return selected


def build_backup_cost_retraction_order(
    candidates: Sequence[BackupCostCandidate],
    shortfall: int,
    *,
    respect_priority: bool = False,
    schedule_low_priority_values_first: bool = False,
) -> list[int]:
    """Return a complete victim-first order, including estimation fallbacks."""
    selected = select_backup_cost_victims(
        candidates,
        shortfall,
        respect_priority=respect_priority,
        schedule_low_priority_values_first=schedule_low_priority_values_first,
    )
    selected_indices = {candidate.index for candidate in selected}

    def fallback_key(candidate: BackupCostCandidate):
        priority_key = (
            _priority_tier_key(candidate.priority, schedule_low_priority_values_first)
            if respect_priority
            else (0, 0)
        )
        return (
            priority_key,
            candidate.backup_tokens,
            -candidate.estimated_relief,
            candidate.index,
        )

    fallback = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.index not in selected_indices
        ),
        key=fallback_key,
    )
    return [candidate.index for candidate in (*selected, *fallback)]
