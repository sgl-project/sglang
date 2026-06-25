from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class WeightExporter:
    _model_runner: Any
    _weights_send_group: dict = field(default_factory=dict)
