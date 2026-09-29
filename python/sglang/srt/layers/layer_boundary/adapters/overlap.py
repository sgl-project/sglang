"""Row requirements at the entrance to two-batch overlap."""

from sglang.srt.layers.layer_boundary.contracts import HandoffRows
from sglang.srt.layers.layer_boundary.layout import enable_moe_dense_fully_dp
from sglang.srt.runtime_context import get_exec


def tbo_handoff(sparse, next_sparse):
    """Record a potential split between a dense layer and a sparse layer."""
    return HandoffRows.TBO_SPLIT if not sparse and next_sparse else None


def resolve_handoff_rows(requirement):
    """Bind a handoff requirement with the construction-time parallel config.

    Dense-local output must reach attention rows before the TBO split.
    Moving that gather after the split changes padding and collective sizes.
    """
    if requirement is HandoffRows.TBO_SPLIT:
        return (
            HandoffRows.ATTENTION
            if enable_moe_dense_fully_dp()
            and get_exec().overlap.enable_two_batch_overlap
            else None
        )
    return requirement
