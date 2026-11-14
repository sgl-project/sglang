# Created by OpenAI
"""TP4/EP1 large-prefill adapter for the shared full-expert kernel family."""

from .fused_moe_tp8_m4193_16768 import fused_moe as _fused_moe


def fused_moe(*args, expert_start=0, fuse_shared_expert=False, **kwargs):
    """Run the bucketed 32K kernel; TP4/EP1 always owns experts from rank zero.

    NextN keeps the shared expert native and passes a zero-filled appended slot,
    while target layers append the real shared expert.  The underlying full-bank
    kernel supports both layouts.
    """
    assert expert_start == 0
    return _fused_moe(*args, **kwargs)
