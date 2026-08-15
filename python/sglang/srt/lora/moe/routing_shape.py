"""Pure shape rules shared by routing and CPU-only sweep admission."""

from __future__ import annotations

FUSED_ALIGN_MIN_VIRTUAL_EXPERTS = 8192
FUSED_ALIGN_MIN_PAIRS = 16384


def uses_fused_align_shape(*, num_virtual_experts: int, num_pairs: int) -> bool:
    """Whether an aligned route uses the fused builder for this shape."""
    if num_virtual_experts < 1:
        raise ValueError("num_virtual_experts must be positive")
    if num_pairs < 0:
        raise ValueError("num_pairs must be non-negative")
    return (
        num_virtual_experts >= FUSED_ALIGN_MIN_VIRTUAL_EXPERTS
        or num_pairs >= FUSED_ALIGN_MIN_PAIRS
    )
