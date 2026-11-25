"""Ascend expert parallel load balancing."""

from typing import TYPE_CHECKING

from sglang.srt.hardware_backend.npu.eplb.config import validate_async_eplb_args

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.eplb.weights import (
        migrate_expert_weights,
        prewarm_eplb_communication,
    )

__all__ = [
    "migrate_expert_weights",
    "prewarm_eplb_communication",
    "validate_async_eplb_args",
]


def __getattr__(name):
    # Keep the existing migration API without loading torch for argument checks.
    if name in {"migrate_expert_weights", "prewarm_eplb_communication"}:
        from sglang.srt.hardware_backend.npu.eplb import weights

        return getattr(weights, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
