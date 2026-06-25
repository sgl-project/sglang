from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Mutable ``_model_update_group`` dict prevents ``frozen=True``; explicit
# Rule-5 exception per the dataclass-defaults sprint-wide rule.
@dataclass(slots=True, kw_only=True)
class WeightUpdater:
    tp_rank: int
    _mr: Any  # ModelRunner — kept untyped to avoid TYPE_CHECKING import here
    _model_update_group: dict = field(default_factory=dict)
