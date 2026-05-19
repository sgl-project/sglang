"""Shared boilerplate for ``test_a_*`` and the smoke test.

Centralises the model id, the CUDA-availability guard, and the
:func:`register_cuda_ci` call shape so individual scenario files do
not duplicate the same 3 lines and so a future fleet-wide change
(e.g. switching to a smaller dummy-weights model) lands in one place.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

PSEUDO_MODE_MODEL: str = "Qwen/Qwen3-0.6B"


def register_pseudo_a_ci() -> None:
    """Register a type-a scenario test with the project's CI metadata.

    Type-a scenarios all share the same wall-clock budget and runner
    profile; encoding that once here keeps cross-scenario drift out.
    """
    register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


requires_cuda = unittest.skipUnless(
    torch.cuda.is_available(), "PseudoEngine requires CUDA"
)
