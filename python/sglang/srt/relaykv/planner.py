from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import RelayKVConfig


@dataclass(frozen=True)
class RelayKVPlan:
    relaykv_enabled: bool
    mode: str
    runtime_policy_state: str
    request_id: Optional[str]
    seq_len: int
    page_size: int
    full_kv_fits: Optional[bool]
    available_kv_budget_tokens: int
    estimated_full_kv_tokens: int
    resident_budget_tokens: int
    planned_resident_tokens: int
    planned_cold_tokens: int
    available_kv_budget_mib: float
    kv_working_budget_tokens: int
    kv_working_budget_source: str
    recent_window: int
    recent_window_tokens: int
    budget_block_size: int
    anchor_blocks: int
    anchor_budget_tokens: int
    retrieval_budget_tokens: int
    retrieval_block_budget: int
    retrieval_top_k_requested: int
    retrieval_top_k_effective: int
    budget_pressure: bool
    coverage_ratio: float
    risk_level: str
    policy_reason: str
    layer_idx: Optional[int]
    prompt_risk: str
    budget_overflow: bool
    budget_policy_reason: str
    anchor_pages: List[int]
    recent_page_range: Tuple[int, int]
    resident_anchor_ranges: List[List[int]]
    resident_recent_ranges: List[List[int]]
    cold_candidate_ranges: List[List[int]]
    estimated_resident_ratio: float

    def to_log_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recent_page_range"] = list(self.recent_page_range)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return self.to_log_dict()


def _ceil_div(x: int, y: int) -> int:
    if y <= 0:
        raise ValueError("page_size must be > 0")
    return (x + y - 1) // y


def _make_range(start: int, end: int) -> List[List[int]]:
    if end <= start:
        return []
    return [[start, end]]


def _estimate_working_budget_tokens(
    config: RelayKVConfig,
    kv_bytes_per_token: Optional[int],
) -> tuple[int, str]:
    if config.kv_working_budget_tokens > 0:
        return config.kv_working_budget_tokens, "explicit_working_budget_tokens"
    if config.available_kv_budget_mib > 0:
        if kv_bytes_per_token and kv_bytes_per_token > 0:
            budget_bytes = int(config.available_kv_budget_mib * 1024 * 1024)
            return (
                budget_bytes // kv_bytes_per_token,
                "estimated_from_available_kv_budget_mib",
            )
        if config.resident_budget_tokens > 0:
            return (
                config.resident_budget_tokens,
                "fallback_resident_budget_tokens_missing_kv_bytes_per_token",
            )
        return 0, "missing_kv_bytes_per_token_for_available_kv_budget_mib"
    return config.resident_budget_tokens, "legacy_resident_budget_tokens"


def _budget_metadata(
    *,
    config: RelayKVConfig,
    seq_len: int,
    page_size: int,
    kv_bytes_per_token: Optional[int],
) -> dict[str, Any]:
    working_budget, reason = _estimate_working_budget_tokens(config, kv_bytes_per_token)
    working_budget = max(int(working_budget), 0)
    budget_block_size = max(int(config.budget_block_size), 1)
    anchor_block_count = (
        config.anchor_blocks if config.anchor_blocks > 0 else config.anchor_pages
    )
    requested_anchor_tokens = max(anchor_block_count, 0) * budget_block_size

    recent_window_tokens = min(config.recent_window, working_budget)
    remaining_after_recent = max(working_budget - recent_window_tokens, 0)
    anchor_budget_tokens = min(requested_anchor_tokens, remaining_after_recent)

    remaining_after_anchor = max(
        working_budget - recent_window_tokens - anchor_budget_tokens, 0
    )
    retrieval_top_k_requested = config.retrieval_top_k
    retrieval_budget_tokens = remaining_after_anchor
    retrieval_block_budget = retrieval_budget_tokens // budget_block_size
    retrieval_top_k_effective = min(
        retrieval_top_k_requested,
        retrieval_block_budget,
    )

    overflow = False
    policy_reason = reason
    if working_budget <= 0:
        overflow = bool(seq_len > 0)
        policy_reason = reason
    elif config.recent_window > recent_window_tokens:
        overflow = True
        policy_reason = "recent_window_clipped_to_working_budget"
    elif requested_anchor_tokens > anchor_budget_tokens:
        overflow = True
        policy_reason = "anchor_budget_clipped_after_recent_window"
    elif retrieval_top_k_requested > 0 and retrieval_block_budget <= 0:
        overflow = True
        policy_reason = "no_retrieval_room_after_recent_and_anchor"
    elif retrieval_top_k_requested > retrieval_top_k_effective:
        overflow = True
        policy_reason = "retrieval_top_k_clipped_to_remaining_budget"

    return {
        "available_kv_budget_mib": config.available_kv_budget_mib,
        "kv_working_budget_tokens": working_budget,
        "kv_working_budget_source": reason,
        "recent_window": config.recent_window,
        "recent_window_tokens": recent_window_tokens,
        "budget_block_size": budget_block_size,
        "anchor_blocks": anchor_block_count,
        "anchor_budget_tokens": anchor_budget_tokens,
        "retrieval_budget_tokens": retrieval_budget_tokens,
        "retrieval_block_budget": retrieval_block_budget,
        "retrieval_top_k_requested": retrieval_top_k_requested,
        "retrieval_top_k_effective": retrieval_top_k_effective,
        "budget_overflow": overflow,
        "budget_policy_reason": policy_reason,
    }


def _policy_metadata(
    *,
    relaykv_enabled: bool,
    mode: str,
    seq_len: int,
    working_budget_tokens: int,
    retrieval_top_k_requested: int,
    retrieval_top_k_effective: int,
    coverage_ratio: float,
    budget_policy_reason: str,
    layer_idx: Optional[int],
    total_layers: Optional[int],
) -> dict[str, Any]:
    full_kv_fits = None
    if working_budget_tokens > 0:
        full_kv_fits = seq_len <= working_budget_tokens

    budget_pressure = bool(full_kv_fits is False)
    top_k_clipped = retrieval_top_k_effective < retrieval_top_k_requested
    low_coverage = coverage_ratio < 0.75 if seq_len > 0 else False
    late_layer = False
    if layer_idx is not None and total_layers and total_layers > 0:
        late_layer = layer_idx >= int(total_layers * 0.75)

    risk_reasons = []
    if low_coverage:
        risk_reasons.append("low_coverage_ratio")
    if top_k_clipped:
        risk_reasons.append("retrieval_top_k_effective_below_requested")
    if late_layer:
        risk_reasons.append("late_layer")

    risk_level = "high" if risk_reasons else "normal"
    prompt_risk = "unknown"

    if not relaykv_enabled or mode == "off":
        runtime_policy_state = "off"
        policy_reason = "relaykv_disabled"
    elif full_kv_fits is True:
        runtime_policy_state = "off"
        policy_reason = "full_kv_fits"
    elif full_kv_fits is None:
        runtime_policy_state = "shadow"
        policy_reason = "insufficient_budget_metadata"
    elif risk_level == "high":
        runtime_policy_state = "fallback_candidate"
        policy_reason = "budget_pressure_high_risk_" + "_and_".join(risk_reasons)
    else:
        runtime_policy_state = "applied_candidate"
        policy_reason = "budget_pressure_normal_risk"

    if runtime_policy_state in ("shadow", "applied_candidate", "fallback_candidate"):
        policy_reason = f"{policy_reason}; budget_policy={budget_policy_reason}"

    return {
        "runtime_policy_state": runtime_policy_state,
        "full_kv_fits": full_kv_fits,
        "available_kv_budget_tokens": working_budget_tokens,
        "estimated_full_kv_tokens": seq_len,
        "budget_pressure": budget_pressure,
        "coverage_ratio": coverage_ratio,
        "risk_level": risk_level,
        "policy_reason": policy_reason,
        "layer_idx": layer_idx,
        "prompt_risk": prompt_risk,
    }


def build_shadow_plan(
    *,
    config: RelayKVConfig,
    seq_len: int,
    page_size: int = 1,
    request_id: Optional[str] = None,
    kv_bytes_per_token: Optional[int] = None,
    layer_idx: Optional[int] = None,
    total_layers: Optional[int] = None,
) -> RelayKVPlan:
    """Build a deterministic shadow resident/cold plan.

    This is deliberately simple for MVP-0:
    - reserve the first N pages as anchors
    - reserve a trailing recent window
    - cap total resident tokens by the effective working KV budget
    - never mutate or inspect actual KV tensors
    """

    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    if page_size <= 0:
        raise ValueError("page_size must be > 0")

    if not config.enabled or config.mode == "off":
        return RelayKVPlan(
            relaykv_enabled=False,
            mode="off",
            runtime_policy_state="off",
            request_id=request_id,
            seq_len=seq_len,
            page_size=page_size,
            full_kv_fits=True,
            available_kv_budget_tokens=0,
            estimated_full_kv_tokens=seq_len,
            resident_budget_tokens=0,
            planned_resident_tokens=seq_len,
            planned_cold_tokens=0,
            available_kv_budget_mib=config.available_kv_budget_mib,
            kv_working_budget_tokens=0,
            kv_working_budget_source="relaykv_disabled",
            recent_window=0,
            recent_window_tokens=0,
            budget_block_size=config.budget_block_size,
            anchor_blocks=config.anchor_blocks,
            anchor_budget_tokens=0,
            retrieval_budget_tokens=0,
            retrieval_block_budget=0,
            retrieval_top_k_requested=config.retrieval_top_k,
            retrieval_top_k_effective=0,
            budget_pressure=False,
            coverage_ratio=1.0 if seq_len > 0 else 0.0,
            risk_level="normal",
            policy_reason="relaykv_disabled",
            layer_idx=layer_idx,
            prompt_risk="unknown",
            budget_overflow=False,
            budget_policy_reason="relaykv_disabled",
            anchor_pages=[],
            recent_page_range=(0, seq_len),
            resident_anchor_ranges=[],
            resident_recent_ranges=_make_range(0, seq_len),
            cold_candidate_ranges=[],
            estimated_resident_ratio=1.0 if seq_len > 0 else 0.0,
        )

    config.validate()

    budget_metadata = _budget_metadata(
        config=config,
        seq_len=seq_len,
        page_size=page_size,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    resident_budget = min(budget_metadata["kv_working_budget_tokens"], seq_len)
    planned_resident = resident_budget
    planned_cold = max(seq_len - planned_resident, 0)

    total_pages = _ceil_div(seq_len, page_size) if seq_len else 0
    anchor_block_count = (
        config.anchor_blocks if config.anchor_blocks > 0 else config.anchor_pages
    )
    anchor_page_count = min(anchor_block_count, total_pages)
    anchors = list(range(anchor_page_count))
    anchor_token_end = min(anchor_page_count * page_size, seq_len)
    anchor_ranges = _make_range(0, anchor_token_end)

    recent_tokens = min(config.recent_window, planned_resident, seq_len)
    recent_start = max(seq_len - recent_tokens, 0)
    recent_end = seq_len
    recent_ranges = _make_range(recent_start, recent_end)

    cold_start = anchor_token_end
    cold_end = recent_start
    if cold_end < cold_start:
        cold_end = cold_start
    cold_candidate_ranges = (
        [] if planned_cold <= 0 else _make_range(cold_start, cold_end)
    )

    ratio = (planned_resident / seq_len) if seq_len > 0 else 0.0
    policy_metadata = _policy_metadata(
        relaykv_enabled=True,
        mode=config.mode,
        seq_len=seq_len,
        working_budget_tokens=budget_metadata["kv_working_budget_tokens"],
        retrieval_top_k_requested=budget_metadata["retrieval_top_k_requested"],
        retrieval_top_k_effective=budget_metadata["retrieval_top_k_effective"],
        coverage_ratio=ratio,
        budget_policy_reason=budget_metadata["budget_policy_reason"],
        layer_idx=layer_idx,
        total_layers=total_layers,
    )

    return RelayKVPlan(
        relaykv_enabled=True,
        mode=config.mode,
        runtime_policy_state=policy_metadata["runtime_policy_state"],
        request_id=request_id,
        seq_len=seq_len,
        page_size=page_size,
        full_kv_fits=policy_metadata["full_kv_fits"],
        available_kv_budget_tokens=policy_metadata["available_kv_budget_tokens"],
        estimated_full_kv_tokens=policy_metadata["estimated_full_kv_tokens"],
        resident_budget_tokens=budget_metadata["kv_working_budget_tokens"],
        planned_resident_tokens=planned_resident,
        planned_cold_tokens=planned_cold,
        available_kv_budget_mib=budget_metadata["available_kv_budget_mib"],
        kv_working_budget_tokens=budget_metadata["kv_working_budget_tokens"],
        kv_working_budget_source=budget_metadata["kv_working_budget_source"],
        recent_window=budget_metadata["recent_window"],
        recent_window_tokens=budget_metadata["recent_window_tokens"],
        budget_block_size=budget_metadata["budget_block_size"],
        anchor_blocks=budget_metadata["anchor_blocks"],
        anchor_budget_tokens=budget_metadata["anchor_budget_tokens"],
        retrieval_budget_tokens=budget_metadata["retrieval_budget_tokens"],
        retrieval_block_budget=budget_metadata["retrieval_block_budget"],
        retrieval_top_k_requested=budget_metadata["retrieval_top_k_requested"],
        retrieval_top_k_effective=budget_metadata["retrieval_top_k_effective"],
        budget_pressure=policy_metadata["budget_pressure"],
        coverage_ratio=policy_metadata["coverage_ratio"],
        risk_level=policy_metadata["risk_level"],
        policy_reason=policy_metadata["policy_reason"],
        layer_idx=policy_metadata["layer_idx"],
        prompt_risk=policy_metadata["prompt_risk"],
        budget_overflow=budget_metadata["budget_overflow"],
        budget_policy_reason=budget_metadata["budget_policy_reason"],
        anchor_pages=anchors,
        recent_page_range=(recent_start, recent_end),
        resident_anchor_ranges=anchor_ranges,
        resident_recent_ranges=recent_ranges,
        cold_candidate_ranges=cold_candidate_ranges,
        estimated_resident_ratio=ratio,
    )


def make_shadow_plan(
    seq_len: int,
    config: RelayKVConfig,
    page_size: int = 1,
    request_id: Optional[str] = None,
    kv_bytes_per_token: Optional[int] = None,
    layer_idx: Optional[int] = None,
    total_layers: Optional[int] = None,
) -> RelayKVPlan:
    return build_shadow_plan(
        config=config,
        seq_len=seq_len,
        page_size=page_size,
        request_id=request_id,
        kv_bytes_per_token=kv_bytes_per_token,
        layer_idx=layer_idx,
        total_layers=total_layers,
    )
