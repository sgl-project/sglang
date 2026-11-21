"""The --schedule-policy choices."""

from enum import Enum


class SchedulePolicy(Enum):
    """A scheduling policy. Its ranker's match plan says whether it ranks with
    prefix matches."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
    HRRN = "hrrn"  # highest response ratio next, token-based aging
    SHORTEST_PREFILL_FIRST = "shortest-prefill-first"
    UNIBOOST = "uniboost"  # tail-aware admission boosting (UniBoost)
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch
