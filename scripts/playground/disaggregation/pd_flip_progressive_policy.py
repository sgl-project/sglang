from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Optional, Sequence


class ProgressiveDecision(Enum):
    START = "start"
    COMMIT = "commit"
    RECOVER = "recover"
    INSUFFICIENT_SAMPLES = "insufficient_samples"


def evaluate_slo_decision(
    prefill_good: int,
    prefill_total: int,
    decode_good: int,
    decode_total: int,
    threshold: float,
    min_prefill_samples: int,
    min_decode_samples: int,
    *,
    observing: bool = False,
) -> ProgressiveDecision:
    if prefill_total < min_prefill_samples or decode_total < min_decode_samples:
        return ProgressiveDecision.INSUFFICIENT_SAMPLES

    prefill_ratio = prefill_good / prefill_total
    decode_ratio = decode_good / decode_total
    if prefill_ratio < threshold and decode_ratio >= threshold:
        return ProgressiveDecision.COMMIT if observing else ProgressiveDecision.START
    return ProgressiveDecision.RECOVER


@dataclass(frozen=True)
class RequestCapacity:
    rid: str
    committed_tokens: int


@dataclass(frozen=True)
class RatioSelection:
    configured_ratio: float
    effective_ratio: float
    selected_rids: tuple[str, ...]
    required_kv_tokens: int
    fallback_count: int


def select_first_batch(
    requests: Sequence[RequestCapacity],
    configured_ratio: float,
    target_req_slots: int,
    target_kv_tokens: int,
    reserve_tokens_per_req: int,
) -> Optional[RatioSelection]:
    if not 0 < configured_ratio < 1:
        raise ValueError("configured_ratio must be between 0 and 1")

    ratio = configured_ratio
    fallback_count = 0
    while requests:
        count = max(1, ceil(len(requests) * ratio))
        selected = tuple(requests[:count])
        required = sum(
            request.committed_tokens + reserve_tokens_per_req
            for request in selected
        )
        if count <= target_req_slots and required <= target_kv_tokens:
            return RatioSelection(
                configured_ratio,
                ratio,
                tuple(request.rid for request in selected),
                required,
                fallback_count,
            )
        if count == 1:
            return None
        ratio /= 2
        fallback_count += 1
    return None
