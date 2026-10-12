"""Row requirements at the entrance to two-batch overlap."""

from sglang.srt.layers.layer_boundary.contracts import ExitRows
from sglang.srt.layers.layer_boundary.layout import is_dense_ffn_fully_dp
from sglang.srt.runtime_context import get_exec


def tbo_exit_rows(sparse, next_layer_sparse):
    """Record a potential split between a dense layer and a sparse layer."""
    return ExitRows.TBO_SPLIT if not sparse and next_layer_sparse else None


def resolve_exit_rows(requirement):
    """Bind an exit-row requirement to the construction-time parallel config.

    Dense-local output must reach attention rows before the TBO split.
    Moving that gather after the split changes padding and collective sizes.
    """
    if requirement is ExitRows.TBO_SPLIT:
        return (
            ExitRows.ATTENTION
            if is_dense_ffn_fully_dp() and get_exec().overlap.enable_two_batch_overlap
            else None
        )
    return requirement
