from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class CanaryMode(str, Enum):
    NONE = "none"
    LOG = "log"
    RAISE = "raise"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    """Top-level canary configuration. All knobs live here; nothing reads env vars deeper in the stack.

    Constructed once inside install_canary(server_args, model_runner) via
    CanaryConfig.from_env(server_args), then frozen and threaded through the canary stack.
    Subsequent runtime never mutates it.

    Fields:
        mode: CanaryMode value. none = no canary installed; log = canary runs, violations are logged
            but do NOT raise (used for production observability + canary self-test perturb); raise =
            violations propagate to host as RuntimeError after the next D2H pump.
        ring_capacity: Violation ring capacity (rows in ViolationLog.violation_ring). Sized generously;
            overflow only drops detail beyond row N, the monotonic counter still grows.
    """

    mode: CanaryMode
    ring_capacity: int

    @classmethod
    def from_env(cls, server_args: "ServerArgs") -> "CanaryConfig":
        mode_raw = server_args.kv_canary.strip().lower()
        if mode_raw not in ("none", "log", "raise"):
            raise ValueError(
                f"kv-canary: kv_canary must be one of none/log/raise, got {mode_raw!r}"
            )

        return cls(
            mode=CanaryMode(mode_raw),
            ring_capacity=envs.SGLANG_KV_CANARY_RING_CAPACITY.get(),
        )
