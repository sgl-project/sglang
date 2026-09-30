"""TP4/EP1 large-prefill adapter for the shared full-expert kernel family."""

from .fused_moe_tp8_m4193_16768 import fused_moe as _fused_moe


def fused_moe(*args, expert_start=0, fuse_shared_expert=False, **kwargs):
    """Run the bucketed 32K kernel; TP4/EP1 always owns experts from rank zero."""
    assert expert_start == 0
    assert fuse_shared_expert
    return _fused_moe(*args, **kwargs)
