from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.kv_canary import api
from sglang.srt.kv_canary.api import torch_reference_conflicts_with_decode_graph
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTorchReferenceConflictsWithDecodeGraph(CustomTestCase):
    """The refusal that keeps a graph-captured torch reference from passing silently.

    The reference path does host work and D2H, so its launches leave nothing in a
    captured decode graph and every replay verifies clean. Each case below pins one
    branch of the gate; the platform capability is patched rather than probed so the
    CPU lane exercises all four.
    """

    def _publish_decode_backend(self, backend: str) -> None:
        override = get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend=backend))
        )
        override.install()
        self.addCleanup(override.restore)

    def _patch_graph_support(self, supported: bool) -> None:
        patcher = mock.patch.object(
            api.current_platform, "support_cuda_graph", return_value=supported
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reference_device_with_captured_decode_conflicts(self) -> None:
        self._patch_graph_support(True)
        self._publish_decode_backend(Backend.FULL)
        self.assertTrue(
            torch_reference_conflicts_with_decode_graph(torch.device("xpu"))
        )

    def test_reference_device_with_decode_graph_disabled_is_allowed(self) -> None:
        self._patch_graph_support(True)
        self._publish_decode_backend(Backend.DISABLED)
        self.assertFalse(
            torch_reference_conflicts_with_decode_graph(torch.device("xpu"))
        )

    def test_platform_without_graph_capture_is_allowed(self) -> None:
        """A device that never captures (CPU) keeps canary on the reference path."""
        self._patch_graph_support(False)
        self._publish_decode_backend(Backend.FULL)
        self.assertFalse(
            torch_reference_conflicts_with_decode_graph(torch.device("cpu"))
        )

    def test_cuda_device_is_never_refused(self) -> None:
        """CUDA/HIP run the real kernels, so the gate must not fire on them."""
        self._patch_graph_support(True)
        self._publish_decode_backend(Backend.FULL)
        self.assertFalse(
            torch_reference_conflicts_with_decode_graph(torch.device("cuda"))
        )


if __name__ == "__main__":
    unittest.main()
