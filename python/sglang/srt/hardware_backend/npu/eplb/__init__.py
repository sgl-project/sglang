"""Ascend expert parallel load balancing."""

from typing import TYPE_CHECKING

from sglang.srt.hardware_backend.npu.eplb.config import validate_async_eplb_args

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.eplb.transport import (
        prewarm_eplb_communication,
    )
    from sglang.srt.hardware_backend.npu.eplb.weights import migrate_expert_weights

__all__ = [
    "migrate_expert_weights",
    "prewarm_eplb_communication",
    "validate_async_eplb_args",
]


def __getattr__(name):
    # Keep the existing migration API without loading torch for argument checks.
    if name == "migrate_expert_weights":
        from sglang.srt.hardware_backend.npu.eplb.weights import migrate_expert_weights

        return migrate_expert_weights
    if name == "prewarm_eplb_communication":
        from sglang.srt.hardware_backend.npu.eplb.transport import (
            prewarm_eplb_communication,
        )

        return prewarm_eplb_communication
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
