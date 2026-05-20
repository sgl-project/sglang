"""One-shot perturbation injection for SteppableEngine.

One-shot fires exactly once on the next body iteration and auto-resets on
consumption. Stored as a Scheduler-instance attribute (not env-var) so arming
and consumption are race-free under stepping mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


_VALID_CHANNELS = frozenset(
    {
        "default",
        "pd_transfer",
        "kv_write",
        "sampler_output",
        "spec_draft",
    }
)

_VALID_KINDS = frozenset(
    {
        "byte_flip",
        "single_byte_corruption",
        "timing_jitter",
    }
)

_FORBIDDEN_PAIRS = frozenset(
    {
        ("sampler_output", "timing_jitter"),
        ("spec_draft", "timing_jitter"),
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArmedPerturbation:
    channel: str
    kind: str
    rank: Optional[int]


def validate_channel_kind(*, channel: str, kind: str) -> None:
    if channel not in _VALID_CHANNELS:
        raise ValueError(f"channel {channel!r} not in {sorted(_VALID_CHANNELS)}")
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind {kind!r} not in {sorted(_VALID_KINDS)}")
    if (channel, kind) in _FORBIDDEN_PAIRS:
        raise ValueError(f"channel x kind combination forbidden: {channel} x {kind}")


def arm_one_shot(
    scheduler: "Scheduler",
    *,
    channel: str,
    kind: str,
    rank: Optional[int],
) -> None:
    validate_channel_kind(channel=channel, kind=kind)
    scheduler._armed_perturbation = _ArmedPerturbation(
        channel=channel,
        kind=kind,
        rank=rank,
    )


def consume_one_shot(
    scheduler: "Scheduler",
    *,
    channel: str,
) -> Optional[_ArmedPerturbation]:
    armed: Optional[_ArmedPerturbation] = getattr(
        scheduler, "_armed_perturbation", None
    )
    if armed is None or armed.channel != channel:
        return None
    scheduler._armed_perturbation = None
    return armed
