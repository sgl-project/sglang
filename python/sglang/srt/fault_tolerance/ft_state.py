from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List


class RankState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


class FaultToleranceState:
    def __init__(self, *, dp_size: int, strategy: str, global_rank_count: int):
        self.dp_size = dp_size
        self.strategy = strategy
        self.global_rank_count = global_rank_count
        self.global_ranks_per_dp = global_rank_count // dp_size
        self.expected_dp_mask = [True] * dp_size
        self.runtime_active_dp_mask = [True] * dp_size
        self.process_alive_global_rank_mask = [True] * global_rank_count
        self.pending_recovery_global_ranks: set[int] = set()
        self.unhealthy_dp_ranks: set[int] = set()
        self.ft_operation_in_progress = False
        self.cluster_paused = False

    def global_ranks_for_dp(self, dp_rank: int) -> range:
        start = dp_rank * self.global_ranks_per_dp
        return range(start, start + self.global_ranks_per_dp)

    def global_ranks_for_dps(self, dp_ranks: Iterable[int]) -> List[int]:
        return [
            rank
            for dp in sorted(set(dp_ranks))
            for rank in self.global_ranks_for_dp(dp)
        ]

    def expand_dp_mask_to_global_rank_mask(self, dp_mask: List[bool]) -> List[bool]:
        return [
            dp_mask[rank // self.global_ranks_per_dp]
            for rank in range(self.global_rank_count)
        ]

    def project_global_rank_mask_to_dp_mask(
        self, global_rank_mask: List[bool]
    ) -> List[bool]:
        return [
            all(global_rank_mask[rank] for rank in self.global_ranks_for_dp(dp))
            for dp in range(self.dp_size)
        ]

    def process_alive_dp_mask(self) -> List[bool]:
        return self.project_global_rank_mask_to_dp_mask(
            self.process_alive_global_rank_mask
        )

    def expected_dp_ranks(self) -> List[int]:
        return [rank for rank, expected in enumerate(self.expected_dp_mask) if expected]

    def _rank_state(self, rank: int) -> RankState:
        if not self.expected_dp_mask[rank] or not self.process_alive_dp_mask()[rank]:
            return RankState.DEAD
        if rank in self.unhealthy_dp_ranks:
            return RankState.UNHEALTHY
        return RankState.HEALTHY

    def status_response(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "total_engines": self.dp_size,
            "engines": [
                {"id": rank, "status": self._rank_state(rank).value}
                for rank in range(self.dp_size)
            ],
        }

    def has_unresolved_expected_dp_fault(self) -> bool:
        """Return whether an expected DP is unhealthy or has lost a process."""
        process_alive_dp_mask = self.process_alive_dp_mask()
        return bool(self.unhealthy_dp_ranks) or any(
            expected and not process_alive_dp_mask[rank]
            for rank, expected in enumerate(self.expected_dp_mask)
        )

    def is_global_admission_blocked(self, route_dp_mask: List[bool]) -> bool:
        return (
            not any(route_dp_mask)
            or self.ft_operation_in_progress
            or (self.strategy == "pause" and self.cluster_paused)
        )

    def observe_process_active_ranks(self, ranks: List[int], *, active: bool) -> None:
        for rank in ranks:
            self.process_alive_global_rank_mask[rank] = active
        if not active:
            self.pending_recovery_global_ranks.update(ranks)
            self.cluster_paused = True

    def observe_runtime_active_dp_mask(self, active_dp_mask: List[bool]) -> None:
        self.runtime_active_dp_mask = active_dp_mask
        self.pending_recovery_global_ranks = {
            rank
            for rank in self.pending_recovery_global_ranks
            if not active_dp_mask[rank // self.global_ranks_per_dp]
        }

    def observe_rank_fault(self, rank: int) -> None:
        if self.strategy == "pause":
            self.unhealthy_dp_ranks.add(rank)
            self.cluster_paused = True
