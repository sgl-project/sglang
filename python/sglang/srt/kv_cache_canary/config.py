from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


class CanaryMode(str, enum.Enum):
    OFF = "off"
    LOG = "log"
    RAISE = "raise"

    @classmethod
    def parse(cls, value: str | "CanaryMode" | None) -> "CanaryMode":
        if value is None:
            return cls.OFF
        if isinstance(value, cls):
            return value
        return cls(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    mode: CanaryMode
    seed: int = 0xC0FFEE1234567890
    violation_ring_capacity: int = 256
    health_print_every_n_forwards: int = 1024
    counter_zero_warmup_forwards: int = 64
    perturb_req_to_token_prob: float = 0.0
    perturb_req_to_token_seed: int = 0

    @classmethod
    def from_server_args(cls, mode: str | CanaryMode | None) -> "CanaryConfig":
        parsed = CanaryMode.parse(mode)
        perturb_prob, perturb_seed = _parse_perturb_env(
            os.environ.get("SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN", "")
        )
        return cls(
            mode=parsed,
            perturb_req_to_token_prob=perturb_prob,
            perturb_req_to_token_seed=perturb_seed,
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not CanaryMode.OFF


def _parse_perturb_env(raw: str) -> Tuple[float, int]:
    """Parse ``<probability>[:<seed>]`` (e.g. ``0.01:42`` or ``0.01``)."""
    if not raw:
        return 0.0, 0
    parts = raw.split(":", 1)
    try:
        prob = float(parts[0])
    except ValueError:
        logger.warning(
            "kv-canary: malformed SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN value %r; disabling perturb",
            raw,
        )
        return 0.0, 0
    if not (0.0 <= prob <= 1.0):
        clamped = max(0.0, min(1.0, prob))
        logger.warning(
            "kv-canary: perturb probability %f out of [0,1]; clamped to %f",
            prob,
            clamped,
        )
        prob = clamped
    seed = 0
    if len(parts) == 2 and parts[1]:
        try:
            seed = int(parts[1])
        except ValueError:
            logger.warning(
                "kv-canary: malformed perturb seed %r; defaulting to 0", parts[1]
            )
            seed = 0
    return prob, seed
