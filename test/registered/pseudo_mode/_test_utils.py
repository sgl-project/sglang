"""Shared boilerplate for ``test_a_*`` / ``test_b_*`` / the smoke test.

Centralises the model id and the CUDA-availability guard so individual
scenario files do not duplicate the same 2 lines and so a future
fleet-wide change (e.g. switching to a smaller dummy-weights model)
lands in one place.

The ``register_cuda_ci(...)`` call itself stays inlined in each test
file: ``scripts/ci/check_registered_tests.py`` validates the CI
metadata via AST parsing and only recognises literal calls to the
backend-registration functions in ``REGISTER_MAPPING``. Wrapping the
call in a helper would silently strip every test file from CI
discovery.
"""

from __future__ import annotations

import unittest

import torch

PSEUDO_MODE_MODEL: str = "Qwen/Qwen3-0.6B"

requires_cuda = unittest.skipUnless(
    torch.cuda.is_available(), "PseudoEngine requires CUDA"
)
