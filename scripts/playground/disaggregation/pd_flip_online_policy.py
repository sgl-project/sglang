#!/usr/bin/env python3
"""Pure online direction policies for PD Flip experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


JsonDict = Dict[str, Any]

DECODE_FIRST_TPOT_INTERCEPT_MS = 6.8165
DECODE_FIRST_TPOT_BATCH_SLOPE_MS = 0.40830
DECODE_FIRST_BATCH_SIZE_FORMULA = "(tpot_slo_ms-6.8165)/0.40830"


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _format_model_coefficient(value: float) -> str:
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text if "." in text else text + ".0"


@dataclass(frozen=True)
class PolicyViolationRates:
    """Policy-specific violation rates over one controller window."""

    prefill_violation_rate: Optional[float]
    decode_violation_rate: Optional[float]
    prefill_queue_attributable_violations: int
    prefill_ttft_violations: int
    prefill_total_requests: int
    prefill_requests_with_queue_evidence: int
    prefill_missing_queue_evidence: int
    prefill_ttft_violations_missing_queue_evidence: int
    decode_bad_tpot_intervals: int
    decode_total_tpot_intervals: int

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class DecodeSufficiencyEstimate:
    """Decode capacity estimate derived from inflight requests and TPOT SLO."""

    current_decode_sufficient: Optional[bool]
    decode_sufficient_after_scale_in: Optional[bool]
    inflight_decode_requests: int
    current_decode_instances: int
    decode_instances_after_scale_in: int
    tpot_slo_seconds: Optional[float]
    estimated_batch_size: Optional[float]
    required_decode_instances: Optional[int]
    batch_size_formula: str
    rounding_rule: str
    reason: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


def estimate_decode_sufficiency(
    *,
    inflight_decode_requests: int,
    current_decode_instances: int,
    tpot_slo_seconds: Optional[float],
    tpot_intercept_ms: float = DECODE_FIRST_TPOT_INTERCEPT_MS,
    tpot_per_batch_ms: float = DECODE_FIRST_TPOT_BATCH_SLOPE_MS,
    round_required_instances_up: bool = False,
    estimated_batch_size_override: Optional[float] = None,
    batch_size_formula_override: Optional[str] = None,
    estimation_reason: Optional[str] = None,
) -> DecodeSufficiencyEstimate:
    """Estimate whether N and N-1 Decode instances satisfy the current load.

    The model supplied by the experiment policy is::

        batch_size = (TPOT_SLO_ms - intercept_ms) / slope_ms_per_batch
        required_decode_instances = round_at_fraction_0_2(
            inflight_requests / batch_size
        )

    A positive inflight load is clamped to at least one Decode instance.  By
    default the historical custom 0.2 fractional boundary is retained.
    Policies that require a strict capacity guarantee set
    ``round_required_instances_up`` and use ``ceil(capacity_ratio)``.
    """

    intercept = float(tpot_intercept_ms)
    slope = float(tpot_per_batch_ms)
    if (
        intercept == DECODE_FIRST_TPOT_INTERCEPT_MS
        and slope == DECODE_FIRST_TPOT_BATCH_SLOPE_MS
    ):
        formula = DECODE_FIRST_BATCH_SIZE_FORMULA
    else:
        formula = "(tpot_slo_ms-{})/{}".format(
            _format_model_coefficient(intercept),
            _format_model_coefficient(slope),
        )
    if batch_size_formula_override:
        formula = str(batch_size_formula_override)
    rounding_rule = (
        "ceil_exact_capacity"
        if round_required_instances_up
        else "fractional_boundary_0.2"
    )
    inflight = int(inflight_decode_requests)
    current = int(current_decode_instances)
    after = max(0, current - 1)
    slo = _positive_float(tpot_slo_seconds)
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        return DecodeSufficiencyEstimate(
            current_decode_sufficient=None,
            decode_sufficient_after_scale_in=None,
            inflight_decode_requests=inflight,
            current_decode_instances=current,
            decode_instances_after_scale_in=after,
            tpot_slo_seconds=slo,
            estimated_batch_size=None,
            required_decode_instances=None,
            batch_size_formula=formula,
            rounding_rule=rounding_rule,
            reason="invalid_tpot_capacity_model",
        )
    if inflight < 0 or current < 0:
        return DecodeSufficiencyEstimate(
            current_decode_sufficient=None,
            decode_sufficient_after_scale_in=None,
            inflight_decode_requests=inflight,
            current_decode_instances=current,
            decode_instances_after_scale_in=after,
            tpot_slo_seconds=slo,
            estimated_batch_size=None,
            required_decode_instances=None,
            batch_size_formula=formula,
            rounding_rule=rounding_rule,
            reason="invalid_negative_decode_capacity_input",
        )
    override = (
        _positive_float(estimated_batch_size_override)
        if estimated_batch_size_override is not None
        else None
    )
    if estimated_batch_size_override is not None and override is None:
        return DecodeSufficiencyEstimate(
            current_decode_sufficient=None,
            decode_sufficient_after_scale_in=None,
            inflight_decode_requests=inflight,
            current_decode_instances=current,
            decode_instances_after_scale_in=after,
            tpot_slo_seconds=slo,
            estimated_batch_size=None,
            required_decode_instances=None,
            batch_size_formula=formula,
            rounding_rule=rounding_rule,
            reason="invalid_estimated_batch_size_override",
        )
    if slo is None and override is None:
        return DecodeSufficiencyEstimate(
            current_decode_sufficient=None,
            decode_sufficient_after_scale_in=None,
            inflight_decode_requests=inflight,
            current_decode_instances=current,
            decode_instances_after_scale_in=after,
            tpot_slo_seconds=None,
            estimated_batch_size=None,
            required_decode_instances=None,
            batch_size_formula=formula,
            rounding_rule=rounding_rule,
            reason="missing_tpot_slo",
        )

    batch_size = (
        override
        if override is not None
        else (slo * 1000.0 - intercept) / slope
    )
    if not math.isfinite(batch_size) or batch_size <= 0:
        return DecodeSufficiencyEstimate(
            current_decode_sufficient=None,
            decode_sufficient_after_scale_in=None,
            inflight_decode_requests=inflight,
            current_decode_instances=current,
            decode_instances_after_scale_in=after,
            tpot_slo_seconds=slo,
            estimated_batch_size=batch_size,
            required_decode_instances=None,
            batch_size_formula=formula,
            rounding_rule=rounding_rule,
            reason="tpot_slo_does_not_produce_positive_batch_size",
        )

    capacity_ratio = inflight / batch_size
    required = (
        int(math.ceil(capacity_ratio))
        if round_required_instances_up
        else int(math.floor(capacity_ratio + 0.8))
    )
    if inflight > 0:
        required = max(1, required)
    return DecodeSufficiencyEstimate(
        current_decode_sufficient=required <= current,
        decode_sufficient_after_scale_in=required <= after,
        inflight_decode_requests=inflight,
        current_decode_instances=current,
        decode_instances_after_scale_in=after,
        tpot_slo_seconds=slo,
        estimated_batch_size=batch_size,
        required_decode_instances=required,
        batch_size_formula=formula,
        rounding_rule=rounding_rule,
        reason=(
            str(estimation_reason)
            if override is not None and estimation_reason
            else "estimated_from_window_request_nonattainment"
            if override is not None
            else "estimated_from_inflight_requests_and_tpot_slo"
        ),
    )


def estimate_window_batch_size_at_nonattainment(
    records: Iterable[JsonDict],
    *,
    target_violation_rate: float = 0.20,
    min_samples: int = 20,
) -> JsonDict:
    """Select the observed integer BS bucket nearest a target violation rate.

    Each eligible request must carry ``decode_batch_size_mean`` plus the
    request-level ``tpot_met`` result.  BS means are rounded half-up into
    integer buckets.  The bucket whose request-level TPOT violation rate is
    nearest ``target_violation_rate`` is selected; ties choose the smaller BS
    as the conservative capacity estimate.  The function is fail-closed until
    the complete window contains ``min_samples`` eligible requests.
    """

    target = float(target_violation_rate)
    minimum = int(min_samples)
    evidence: JsonDict = {
        "estimator": "window_p20_nonattainment",
        "target_request_violation_rate": target,
        "minimum_request_samples": minimum,
        "eligible_request_samples": 0,
        "missing_batch_size_samples": 0,
        "missing_request_tpot_outcomes": 0,
        "batch_size_bucket_rounding": "round_half_up",
        "bucket_selection": (
            "minimum_absolute_violation_rate_distance_then_smaller_batch_size"
        ),
        "buckets": [],
        "estimated_batch_size": None,
        "reason": "insufficient_window_batch_size_samples",
    }
    if not math.isfinite(target) or not 0 <= target <= 1 or minimum <= 0:
        evidence["reason"] = "invalid_window_batch_size_estimator_config"
        return evidence

    buckets: Dict[int, Dict[str, int]] = {}
    for record in records:
        if str(record.get("status") or "") != "completed" or record.get("error"):
            continue
        batch_size = _positive_float(record.get("decode_batch_size_mean"))
        if batch_size is None:
            evidence["missing_batch_size_samples"] += 1
            continue
        tpot_met = record.get("tpot_met")
        if not isinstance(tpot_met, bool):
            evidence["missing_request_tpot_outcomes"] += 1
            continue
        bucket = max(1, int(math.floor(batch_size + 0.5)))
        stats = buckets.setdefault(bucket, {"requests": 0, "violations": 0})
        stats["requests"] += 1
        stats["violations"] += int(not tpot_met)

    bucket_rows = []
    for batch_size, stats in sorted(buckets.items()):
        rate = stats["violations"] / stats["requests"]
        bucket_rows.append(
            {
                "batch_size": batch_size,
                "requests": stats["requests"],
                "violations": stats["violations"],
                "violation_rate": rate,
                "distance_to_target": abs(rate - target),
            }
        )
    evidence["buckets"] = bucket_rows
    evidence["eligible_request_samples"] = sum(
        row["requests"] for row in bucket_rows
    )
    if evidence["eligible_request_samples"] < minimum or not bucket_rows:
        return evidence

    selected = min(
        bucket_rows,
        key=lambda row: (row["distance_to_target"], row["batch_size"]),
    )
    evidence["estimated_batch_size"] = float(selected["batch_size"])
    evidence["selected_bucket"] = dict(selected)
    evidence["reason"] = "estimated_from_window_request_nonattainment"
    return evidence


def compute_policy_violation_rates(
    records: Iterable[JsonDict],
    queue_seconds_by_request: Mapping[str, float],
) -> PolicyViolationRates:
    """Compute the Prefill queue-attributable and Decode TPOT violation rates.

    Prefill counts a request as violated only when its client-observed TTFT is
    above its SLO while ``TTFT - scheduler_queue_time`` would meet that SLO.
    The denominator remains every request with valid TTFT/SLO evidence.

    Decode preserves the current controller boundary: bad inter-token intervals
    divided by all observed inter-token intervals.
    """

    prefill_violations = 0
    prefill_ttft_violations = 0
    prefill_ttft_violations_missing_queue = 0
    prefill_total = 0
    prefill_with_queue = 0
    decode_bad = 0
    decode_total = 0

    for record in records:
        request_id = record.get("request_id")
        ttft = _positive_float(record.get("ttft_seconds"))
        ttft_slo = _positive_float(record.get("ttft_slo_seconds"))
        if request_id is not None and ttft is not None and ttft_slo is not None:
            prefill_total += 1
            ttft_violated = ttft > ttft_slo
            prefill_ttft_violations += int(ttft_violated)
            queue_seconds = queue_seconds_by_request.get(str(request_id))
            if queue_seconds is not None:
                try:
                    queue_seconds = max(0.0, float(queue_seconds))
                except (TypeError, ValueError):
                    queue_seconds = None
            if queue_seconds is not None and math.isfinite(queue_seconds):
                prefill_with_queue += 1
                if ttft_violated and max(0.0, ttft - queue_seconds) <= ttft_slo:
                    prefill_violations += 1
            elif ttft_violated:
                prefill_ttft_violations_missing_queue += 1

        try:
            interval_total = int(record.get("total_tpot_intervals") or 0)
            interval_good = int(record.get("good_tpot_intervals") or 0)
        except (TypeError, ValueError):
            continue
        if interval_total <= 0:
            continue
        interval_good = max(0, min(interval_good, interval_total))
        decode_total += interval_total
        decode_bad += interval_total - interval_good

    return PolicyViolationRates(
        prefill_violation_rate=(
            prefill_violations / prefill_total if prefill_total > 0 else None
        ),
        decode_violation_rate=(decode_bad / decode_total if decode_total > 0 else None),
        prefill_queue_attributable_violations=prefill_violations,
        prefill_ttft_violations=prefill_ttft_violations,
        prefill_total_requests=prefill_total,
        prefill_requests_with_queue_evidence=prefill_with_queue,
        prefill_missing_queue_evidence=max(0, prefill_total - prefill_with_queue),
        prefill_ttft_violations_missing_queue_evidence=(
            prefill_ttft_violations_missing_queue
        ),
        decode_bad_tpot_intervals=decode_bad,
        decode_total_tpot_intervals=decode_total,
    )


@dataclass(frozen=True)
class OnlinePolicyDecision:
    policy: str
    direction: Optional[str]
    candidate_direction: Optional[str]
    reason: str
    prefill_violation_rate: Optional[float]
    decode_violation_rate: Optional[float]
    violation_gap: Optional[float]
    gap_definition: str
    gap_threshold: float
    decode_first_prefill_protect: Optional[bool] = None
    decode_first_d_to_p_require_prefill_gap: Optional[bool] = None
    current_decode_sufficient: Optional[bool] = None
    decode_sufficient_after_scale_in: Optional[bool] = None
    decode_sufficiency_evidence: Any = "not_used"

    def to_dict(self) -> JsonDict:
        return asdict(self)


def decide_slo_target(
    rates: PolicyViolationRates, *, gap_threshold: float = 0.20
) -> OnlinePolicyDecision:
    """Move one worker toward the role with the higher violation rate."""

    prefill = rates.prefill_violation_rate
    decode = rates.decode_violation_rate
    gap = decode - prefill if prefill is not None and decode is not None else None
    common = {
        "policy": "slo_target",
        "prefill_violation_rate": prefill,
        "decode_violation_rate": decode,
        "violation_gap": gap,
        "gap_definition": "decode_violation_rate_minus_prefill_violation_rate",
        "gap_threshold": gap_threshold,
    }
    if gap is None:
        return OnlinePolicyDecision(
            direction=None,
            candidate_direction=None,
            reason="missing_policy_violation_rates",
            **common,
        )
    if gap > gap_threshold:
        return OnlinePolicyDecision(
            direction="p_to_d",
            candidate_direction="p_to_d",
            reason="decode_violation_gap_exceeded",
            **common,
        )
    if gap < -gap_threshold:
        return OnlinePolicyDecision(
            direction="d_to_p",
            candidate_direction="d_to_p",
            reason="prefill_violation_gap_exceeded",
            **common,
        )
    return OnlinePolicyDecision(
        direction=None,
        candidate_direction=None,
        reason="within_violation_gap_deadband",
        **common,
    )


def decide_decode_first(
    rates: PolicyViolationRates,
    *,
    current_decode_sufficient: Optional[bool],
    decode_sufficient_after_scale_in: Optional[bool],
    sufficiency_evidence: Any,
    gap_threshold: float = 0.10,
    protect_prefill_when_decode_insufficient: bool = True,
    require_prefill_gap_for_d_to_p: bool = True,
) -> OnlinePolicyDecision:
    """Protect Decode capacity and optionally gate D-to-P on Prefill pressure."""

    prefill = rates.prefill_violation_rate
    decode = rates.decode_violation_rate
    gap = prefill - decode if prefill is not None and decode is not None else None
    common = {
        "policy": "decode_first",
        "prefill_violation_rate": prefill,
        "decode_violation_rate": decode,
        "violation_gap": gap,
        "gap_definition": "prefill_violation_rate_minus_decode_violation_rate",
        "gap_threshold": gap_threshold,
        "decode_first_prefill_protect": (
            protect_prefill_when_decode_insufficient
        ),
        "decode_first_d_to_p_require_prefill_gap": (
            require_prefill_gap_for_d_to_p
        ),
        "current_decode_sufficient": current_decode_sufficient,
        "decode_sufficient_after_scale_in": decode_sufficient_after_scale_in,
        "decode_sufficiency_evidence": sufficiency_evidence,
    }
    if current_decode_sufficient is None:
        return OnlinePolicyDecision(
            direction=None,
            candidate_direction="p_to_d",
            reason="decode_sufficiency_placeholder_unresolved",
            **common,
        )
    if not current_decode_sufficient:
        if not protect_prefill_when_decode_insufficient:
            return OnlinePolicyDecision(
                direction="p_to_d",
                candidate_direction="p_to_d",
                reason="current_decode_insufficient_prefill_protect_disabled",
                **common,
            )
        if gap is None:
            return OnlinePolicyDecision(
                direction=None,
                candidate_direction="p_to_d",
                reason="missing_policy_violation_rates",
                **common,
            )
        if gap > gap_threshold:
            return OnlinePolicyDecision(
                direction=None,
                candidate_direction="p_to_d",
                reason="current_decode_insufficient_but_prefill_gap_protected",
                **common,
            )
        return OnlinePolicyDecision(
            direction="p_to_d",
            candidate_direction="p_to_d",
            reason="current_decode_insufficient_and_prefill_gap_not_exceeded",
            **common,
        )
    if require_prefill_gap_for_d_to_p:
        if gap is None:
            return OnlinePolicyDecision(
                direction=None,
                candidate_direction=None,
                reason="missing_policy_violation_rates",
                **common,
            )
        if gap <= gap_threshold:
            return OnlinePolicyDecision(
                direction=None,
                candidate_direction=None,
                reason="prefill_gap_not_above_decode_first_threshold",
                **common,
            )
    if decode_sufficient_after_scale_in is None:
        return OnlinePolicyDecision(
            direction=None,
            candidate_direction="d_to_p",
            reason="post_scale_in_decode_sufficiency_placeholder_unresolved",
            **common,
        )
    if not decode_sufficient_after_scale_in:
        return OnlinePolicyDecision(
            direction=None,
            candidate_direction="d_to_p",
            reason="decode_would_be_insufficient_after_scale_in",
            **common,
        )
    return OnlinePolicyDecision(
        direction="d_to_p",
        candidate_direction="d_to_p",
        reason=(
            "prefill_gap_exceeded_and_decode_remains_sufficient"
            if require_prefill_gap_for_d_to_p
            else "decode_minus_one_sufficient_prefill_gap_disabled"
        ),
        **common,
    )


def decide_tpot_capacity(
    *,
    current_decode_sufficient: Optional[bool],
    decode_sufficient_after_scale_in: Optional[bool],
    sufficiency_evidence: Any,
) -> OnlinePolicyDecision:
    """Scale Decode by the TPOT-derived capacity model only.

    Donate one Decode worker when ``D-1`` workers still satisfy the current
    inflight load, add one Decode worker when the current ``D`` workers are
    insufficient, and otherwise keep the topology unchanged.  TTFT and
    Prefill violation evidence are intentionally outside this policy.
    """

    common = {
        "policy": "tpot_capacity",
        "prefill_violation_rate": None,
        "decode_violation_rate": None,
        "violation_gap": None,
        "gap_definition": "not_used_tpot_capacity_only",
        "gap_threshold": 0.0,
        "current_decode_sufficient": current_decode_sufficient,
        "decode_sufficient_after_scale_in": decode_sufficient_after_scale_in,
        "decode_sufficiency_evidence": sufficiency_evidence,
    }
    if current_decode_sufficient is None or decode_sufficient_after_scale_in is None:
        return OnlinePolicyDecision(
            direction=None,
            candidate_direction=None,
            reason="decode_capacity_evidence_unavailable",
            **common,
        )
    if decode_sufficient_after_scale_in:
        return OnlinePolicyDecision(
            direction="d_to_p",
            candidate_direction="d_to_p",
            reason="decode_minus_one_sufficient",
            **common,
        )
    if not current_decode_sufficient:
        return OnlinePolicyDecision(
            direction="p_to_d",
            candidate_direction="p_to_d",
            reason="current_decode_insufficient",
            **common,
        )
    return OnlinePolicyDecision(
        direction=None,
        candidate_direction=None,
        reason="current_decode_exact_capacity_band",
        **common,
    )
