"""FlashInfer-compatible facade for the copied MNNVL CuTe DSL backend."""

from __future__ import annotations

import flashinfer.comm as _upstream_comm

from .mnnvl_cutedsl_ar import (
    MNNVLCuteDSLAllReduceFusionWorkspace,
    _mnnvl_cutedsl_allreduce_fusion,
)

AllReduceFusionPattern = _upstream_comm.AllReduceFusionPattern


def allreduce_fusion(*, input, workspace, pattern, **kwargs):
    """Use the copied backend locally and preserve upstream dispatch otherwise."""
    if isinstance(workspace, MNNVLCuteDSLAllReduceFusionWorkspace):
        # The unified API documents this as ignored by MNNVL backends and does
        # not forward it to the backend-specific implementation.
        kwargs.pop("trigger_completion_at_end", None)
        return _mnnvl_cutedsl_allreduce_fusion(
            input=input,
            workspace=workspace,
            pattern=pattern,
            **kwargs,
        )
    return _upstream_comm.allreduce_fusion(
        input=input,
        workspace=workspace,
        pattern=pattern,
        **kwargs,
    )


__all__ = [
    "AllReduceFusionPattern",
    "MNNVLCuteDSLAllReduceFusionWorkspace",
    "allreduce_fusion",
]
