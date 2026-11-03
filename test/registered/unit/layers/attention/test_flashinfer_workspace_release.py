"""The shared FlashInfer workspace is dropped once no backend uses it.

Hybrid QSA models build a FlashInferAttnBackend that the model-level wrapper
then replaces; the process-wide workspace it fetched must not outlive it.
"""

import unittest

import torch

from sglang.srt.layers.attention import flashinfer_backend
from sglang.srt.runtime_context import get_buffer, pop_buffer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORKSPACE_BYTES = 4096


class _WorkspaceUser:
    """Registers like a FlashInferAttnBackend that uses the shared workspace."""

    def __init__(self):
        self.workspace_buffer = get_buffer(
            "flashinfer_workspace",
            lambda: torch.empty(WORKSPACE_BYTES, dtype=torch.uint8),
        )
        flashinfer_backend._global_workspace_users.add(self)
        # The real backend's helpers point back at it, so only gc reclaims it.
        self.self_ref = self


class TestFlashInferWorkspaceRelease(CustomTestCase):
    def setUp(self):
        pop_buffer("flashinfer_workspace")
        self.addCleanup(pop_buffer, "flashinfer_workspace")

    def test_live_backend_keeps_the_workspace(self):
        backend = _WorkspaceUser()
        self.assertEqual(flashinfer_backend.release_unused_global_workspace(), 0)
        self.assertIs(
            get_buffer("flashinfer_workspace", lambda: None), backend.workspace_buffer
        )

    def test_replaced_backend_releases_the_workspace(self):
        _WorkspaceUser()  # dropped at once, as when a wrapper replaces it
        released = flashinfer_backend.release_unused_global_workspace()
        self.assertEqual(released, WORKSPACE_BYTES)
        self.assertIsNone(pop_buffer("flashinfer_workspace"))
        # A backend built later allocates a fresh workspace.
        later = _WorkspaceUser()
        self.assertEqual(later.workspace_buffer.numel(), WORKSPACE_BYTES)

    def test_nothing_to_release(self):
        self.assertEqual(flashinfer_backend.release_unused_global_workspace(), 0)


if __name__ == "__main__":
    unittest.main()
