from __future__ import annotations

import enum
import os
from dataclasses import dataclass


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
    seed: int = 0x9E3779B97F4A7C15
    violation_ring_capacity: int = 256
    health_print_every_n_forwards: int = 1024
    counter_zero_warmup_forwards: int = 64
    perturb_req_to_token_prob: float = 0.0

    @classmethod
    def from_server_args(cls, mode: str | CanaryMode | None) -> "CanaryConfig":
        parsed = CanaryMode.parse(mode)
        perturb_env = os.environ.get("SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN", "")
        try:
            perturb = float(perturb_env) if perturb_env else 0.0
        except ValueError:
            perturb = 0.0
        return cls(mode=parsed, perturb_req_to_token_prob=perturb)

    @property
    def enabled(self) -> bool:
        return self.mode is not CanaryMode.OFF
