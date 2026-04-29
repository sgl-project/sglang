from __future__ import annotations

import json
import logging
from typing import Optional

from .planner import RelayKVPlan

logger = logging.getLogger(__name__)


def log_shadow_plan(plan: RelayKVPlan, *, prefix: str = "relaykv_shadow_plan") -> None:
    """Emit a compact JSON log line for MVP-0 shadow planning."""

    logger.info("%s=%s", prefix, json.dumps(plan.to_log_dict(), sort_keys=True))


def should_log(step_idx: Optional[int], interval: int) -> bool:
    if interval <= 0:
        return True
    if step_idx is None:
        return True
    return step_idx % interval == 0
