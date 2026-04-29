from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import RelayKVConfig


@dataclass(frozen=True)
class RelayKVPlan:
    relaykv_enabled: bool
    mode: str
    request_id: Optional[str]
    seq_len: int
    page_size: int
    resident_budget_tokens: int
    planned_resident_tokens: int
    planned_cold_tokens: int
    anchor_pages: List[int]
    recent_page_range: Tuple[int, int]
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


def build_shadow_plan(
    *,
    config: RelayKVConfig,
    seq_len: int,
    page_size: int = 1,
    request_id: Optional[str] = None,
) -> RelayKVPlan:
    """Build a deterministic shadow resident/cold plan.

    This is deliberately simple for MVP-0:
    - reserve the first N pages as anchors
    - reserve a trailing recent window
    - cap total resident tokens by resident_budget_tokens
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
            request_id=request_id,
            seq_len=seq_len,
            page_size=page_size,
            resident_budget_tokens=0,
            planned_resident_tokens=seq_len,
            planned_cold_tokens=0,
            anchor_pages=[],
            recent_page_range=(0, seq_len),
            estimated_resident_ratio=1.0 if seq_len > 0 else 0.0,
        )

    config.validate()

    resident_budget = min(config.resident_budget_tokens, seq_len)
    planned_resident = resident_budget
    planned_cold = max(seq_len - planned_resident, 0)

    total_pages = _ceil_div(seq_len, page_size) if seq_len else 0
    anchor_page_count = min(config.anchor_pages, total_pages)
    anchors = list(range(anchor_page_count))

    recent_tokens = min(config.recent_window, planned_resident, seq_len)
    recent_start = max(seq_len - recent_tokens, 0)
    recent_end = seq_len

    ratio = (planned_resident / seq_len) if seq_len > 0 else 0.0

    return RelayKVPlan(
        relaykv_enabled=True,
        mode=config.mode,
        request_id=request_id,
        seq_len=seq_len,
        page_size=page_size,
        resident_budget_tokens=config.resident_budget_tokens,
        planned_resident_tokens=planned_resident,
        planned_cold_tokens=planned_cold,
        anchor_pages=anchors,
        recent_page_range=(recent_start, recent_end),
        estimated_resident_ratio=ratio,
    )


def make_shadow_plan(
    seq_len: int,
    config: RelayKVConfig,
    page_size: int = 1,
    request_id: Optional[str] = None,
) -> RelayKVPlan:
    return build_shadow_plan(
        config=config,
        seq_len=seq_len,
        page_size=page_size,
        request_id=request_id,
    )
