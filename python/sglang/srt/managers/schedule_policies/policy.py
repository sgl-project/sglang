"""The --schedule-policy choices."""

from enum import Enum


class SchedulePolicy(Enum):
    """A scheduling policy. The cache-aware ones rank with prefix matches."""

    # Cache aware.
    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
    HRRN = "hrrn"  # highest response ratio next, token-based aging
    SHORTEST_PREFILL_FIRST = "shortest-prefill-first"
    UNIBOOST = "uniboost"  # tail-aware admission boosting (UniBoost)
    # Cache agnostic.
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch

    @property
    def cache_aware(self) -> bool:
        return self in _CACHE_AWARE


_CACHE_AWARE = frozenset(
    {
        SchedulePolicy.LPM,
        SchedulePolicy.DFS_WEIGHT,
        SchedulePolicy.HRRN,
        SchedulePolicy.SHORTEST_PREFILL_FIRST,
        SchedulePolicy.UNIBOOST,
    }
)
