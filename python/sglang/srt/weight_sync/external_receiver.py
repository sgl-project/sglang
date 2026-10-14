"""Allowlisted external weight-update receivers.

Each path in the ``--weight-update-receivers`` allowlist names a factory
taking one ``WeightUpdateReceiverContext`` and returning an object with
``receive(payload)`` and ``destroy()``. See the integrator contract in
docs/docs/advanced_features/sglang_for_rl.mdx ("External Weight-Update
Receivers").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

from sglang.srt.utils import dynamic_import


@dataclass(frozen=True, kw_only=True)
class WeightUpdateReceiverContext:
    """Per-TP-worker context handed to a receiver factory."""

    model: Any
    device: Any
    tp_rank: int
    tp_size: int
    group_name: str
    master_address: str
    master_port: int
    world_size: int
    rank_offset: int
    # Opaque, request-supplied; SGLang never interprets it.
    init_payload: Optional[Dict[str, Any]] = None


class WeightUpdateReceiver(Protocol):
    def receive(self, payload: Optional[Dict[str, Any]]) -> None:
        """Write one round of weights into the live model."""

    def destroy(self) -> None:
        """Release everything the receiver holds."""


WeightUpdateReceiverFactory = Callable[
    [WeightUpdateReceiverContext], WeightUpdateReceiver
]


def build_weight_update_receiver(
    path: str,
    allowed: Optional[Sequence[str]],
    context: WeightUpdateReceiverContext,
) -> WeightUpdateReceiver:
    """Check the allowlist before importing the path."""
    if path not in (allowed or ()):
        raise ValueError(
            f"receiver {path!r} is not listed in --weight-update-receivers"
        )
    receiver = dynamic_import(path)(context)
    if not callable(getattr(receiver, "receive", None)) or not callable(
        getattr(receiver, "destroy", None)
    ):
        error = TypeError(
            f"receiver factory {path!r} must return an object with "
            "receive(payload) and destroy()"
        )
        destroy = getattr(receiver, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except Exception as destroy_error:
                # The rejection explains what is wrong; a failed cleanup rides
                # along as its cause instead of masking it.
                raise error from destroy_error
        raise error
    return receiver
