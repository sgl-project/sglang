from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .planner import RelayKVPlan

logger = logging.getLogger(__name__)


def log_shadow_plan(
    plan: RelayKVPlan,
    *,
    prefix: str = "relaykv_shadow_plan",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a compact JSON log line for MVP-0 shadow planning."""

    payload = plan.to_log_dict()
    if extra:
        payload.update(extra)
    logger.info("%s=%s", prefix, json.dumps(payload, sort_keys=True))


def should_log(step_idx: Optional[int], interval: int) -> bool:
    if interval <= 0:
        return True
    if step_idx is None:
        return True
    return step_idx % interval == 0
