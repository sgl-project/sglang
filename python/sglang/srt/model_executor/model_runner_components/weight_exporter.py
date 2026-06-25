from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Mutable ``_weights_send_group`` dict prevents ``frozen=True``; explicit
# Rule-5 exception per the dataclass-defaults sprint-wide rule.
# tp_rank / tp_size / gpu_id read via ``self._mr`` (consistent with
# WeightUpdater) — no redundant storage.
@dataclass(slots=True, kw_only=True)
class WeightExporter:
    _mr: Any  # ModelRunner — kept untyped to avoid TYPE_CHECKING import here
    _weights_send_group: dict = field(default_factory=dict)
