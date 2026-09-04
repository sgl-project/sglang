"""Pure decision logic for the Prefill queue/utilization PD Flip policy."""

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class QueueUtilDirectionDecision:
    direction: Optional[str]
    reason: str
    queue_sample_count: int
    queue_window_size: int
    queue_over_threshold_count: int
    queue_over_threshold_ratio: Optional[float]
    queue_threshold_seconds: float
    queue_overload_ratio_threshold: float
    queue_scale_in_ratio_threshold: float
    prefill_scale_in_headroom_workers: float
    prefill_worker_count: int
    prefill_average_utilization: Optional[float]
    prefill_scale_in_utilization_threshold: Optional[float]
    prefill_min_role_seconds: float
    prefill_utilizations: Mapping[str, Optional[float]]
    prefill_role_seconds: Mapping[str, Optional[float]]

    def to_dict(self):
        return asdict(self)


def decide_queue_util_flip_direction(
    queue_seconds: Sequence[float],
    *,
    prefill_utilizations: Mapping[str, Optional[float]],
    prefill_role_seconds: Mapping[str, Optional[float]],
    queue_window_size: int,
    queue_threshold_seconds: float,
    queue_overload_ratio_threshold: float,
    queue_scale_in_ratio_threshold: float,
    prefill_scale_in_headroom_workers: float,
    prefill_min_role_seconds: float,
) -> QueueUtilDirectionDecision:
    """Choose one PD Flip direction from request queueing and Prefill busy time.

    D->P has priority when both directions could otherwise appear eligible.  A
    full request-count window is required for either direction so a newly
    started controller cannot immediately scale in before traffic arrives.
    """

    if queue_window_size <= 0:
        raise ValueError("queue_window_size must be positive")
    if queue_threshold_seconds < 0:
        raise ValueError("queue_threshold_seconds must be non-negative")
    if not 0 <= queue_overload_ratio_threshold <= 1:
        raise ValueError("queue_overload_ratio_threshold must be between 0 and 1")
    if not 0 <= queue_scale_in_ratio_threshold <= 1:
        raise ValueError("queue_scale_in_ratio_threshold must be between 0 and 1")
    if queue_scale_in_ratio_threshold > queue_overload_ratio_threshold:
        raise ValueError(
            "queue_scale_in_ratio_threshold must not exceed "
            "queue_overload_ratio_threshold"
        )
    if prefill_scale_in_headroom_workers <= 0:
        raise ValueError("prefill_scale_in_headroom_workers must be positive")
    if prefill_min_role_seconds < 0:
        raise ValueError("prefill_min_role_seconds must be non-negative")

    window = [float(value) for value in queue_seconds[-queue_window_size:]]
    over_count = sum(value >= queue_threshold_seconds for value in window)
    over_ratio = over_count / len(window) if window else None
    prefill_worker_count = len(prefill_utilizations)
    complete_utilizations = [
        float(value)
        for value in prefill_utilizations.values()
        if value is not None
    ]
    average_utilization = (
        sum(complete_utilizations) / prefill_worker_count
        if prefill_worker_count
        and len(complete_utilizations) == prefill_worker_count
        else None
    )
    scale_in_utilization_threshold = (
        (prefill_worker_count - prefill_scale_in_headroom_workers)
        / prefill_worker_count
        if prefill_worker_count
        else None
    )
    common = {
        "queue_sample_count": len(window),
        "queue_window_size": queue_window_size,
        "queue_over_threshold_count": over_count,
        "queue_over_threshold_ratio": over_ratio,
        "queue_threshold_seconds": queue_threshold_seconds,
        "queue_overload_ratio_threshold": queue_overload_ratio_threshold,
        "queue_scale_in_ratio_threshold": queue_scale_in_ratio_threshold,
        "prefill_scale_in_headroom_workers": (
            prefill_scale_in_headroom_workers
        ),
        "prefill_worker_count": prefill_worker_count,
        "prefill_average_utilization": average_utilization,
        "prefill_scale_in_utilization_threshold": (
            scale_in_utilization_threshold
        ),
        "prefill_min_role_seconds": prefill_min_role_seconds,
        "prefill_utilizations": dict(prefill_utilizations),
        "prefill_role_seconds": dict(prefill_role_seconds),
    }

    if len(window) < queue_window_size:
        return QueueUtilDirectionDecision(
            direction=None, reason="insufficient_queue_samples", **common
        )
    if over_ratio is not None and over_ratio >= queue_overload_ratio_threshold:
        return QueueUtilDirectionDecision(
            direction="d_to_p", reason="prefill_queue_overloaded", **common
        )
    if not prefill_utilizations:
        return QueueUtilDirectionDecision(
            direction=None, reason="missing_prefill_workers", **common
        )
    if set(prefill_utilizations) != set(prefill_role_seconds):
        return QueueUtilDirectionDecision(
            direction=None, reason="incomplete_prefill_utilization_evidence", **common
        )
    if any(value is None for value in prefill_utilizations.values()) or any(
        value is None for value in prefill_role_seconds.values()
    ):
        return QueueUtilDirectionDecision(
            direction=None, reason="incomplete_prefill_utilization_evidence", **common
        )
    if any(
        float(value) < prefill_min_role_seconds
        for value in prefill_role_seconds.values()
    ):
        return QueueUtilDirectionDecision(
            direction=None, reason="prefill_role_observation_too_short", **common
        )
    if (
        average_utilization is not None
        and scale_in_utilization_threshold is not None
        and average_utilization <= scale_in_utilization_threshold
        and over_ratio is not None
        and over_ratio < queue_scale_in_ratio_threshold
    ):
        return QueueUtilDirectionDecision(
            direction="p_to_d",
            reason="prefill_average_under_capacity_and_queue_quiet",
            **common,
        )
    return QueueUtilDirectionDecision(
        direction=None,
        reason="prefill_scale_in_joint_condition_not_met",
        **common,
    )
