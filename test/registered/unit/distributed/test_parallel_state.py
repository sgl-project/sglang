"""Unit tests for sglang.srt.distributed.parallel_state — no server, no model loading."""

import unittest

import sglang.srt.platforms as platforms_mod
from sglang.srt.distributed.parallel_state import get_default_distributed_backend
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


# Worked example: an out-of-tree platform that overrides the torch backend.
# Future plugin authors can pattern-match against this in their own tests.
class _OverridingPlatform(SRTPlatform):
    device_type = "cuda"

    def get_torch_distributed_backend_str(self) -> str:
        return "fake_backend"


# A platform whose override raises an unexpected error at init time.
class _RaisingPlatform(SRTPlatform):
    device_type = "cuda"

    def get_torch_distributed_backend_str(self) -> str:
        raise RuntimeError("network init failure")


# Mirrors all in-tree platforms today: no override, base raises NotImplementedError.
class _DefaultPlatform(SRTPlatform):
    device_type = "cuda"


class TestGetDefaultDistributedBackend(CustomTestCase):
    """Cover all paths of get_default_distributed_backend.

    The function looks up `_DEVICE_TO_DISTRIBUTED_BACKEND[device]` by default,
    but when `device == current_platform.device_type` it first asks
    `current_platform.get_torch_distributed_backend_str()`. Tests use real
    SRTPlatform subclasses so the assertions stay close to how an actual
    out-of-tree plugin would interact with this dispatcher.

    `current_platform` is exposed via __getattr__ on the platforms module
    backed by the `_current_platform` singleton, so the test overrides that
    singleton directly and restores it in tearDown.
    """

    def setUp(self):
        self._saved_platform = platforms_mod._current_platform

    def tearDown(self):
        platforms_mod._current_platform = self._saved_platform

    def _install(self, platform: SRTPlatform) -> None:
        platforms_mod._current_platform = platform

    def test_overriding_platform_supplies_backend(self):
        self._install(_OverridingPlatform())
        self.assertEqual(get_default_distributed_backend("cuda"), "fake_backend")

    def test_overriding_platform_skipped_for_non_active_device(self):
        # Even when current_platform implements get_torch_distributed_backend_str,
        # callers asking for a different device (e.g. an auxiliary "cpu" gloo
        # group on a CUDA process) must keep going through the dict.
        self._install(_OverridingPlatform())
        self.assertEqual(get_default_distributed_backend("cpu"), "gloo")

    def test_unexpected_exception_warns_and_falls_back_to_dict(self):
        self._install(_RaisingPlatform())
        with self.assertLogs(
            "sglang.srt.distributed.parallel_state", level="WARNING"
        ) as cm:
            self.assertEqual(get_default_distributed_backend("cuda"), "nccl")
        self.assertTrue(
            any("network init failure" in line for line in cm.output),
            f"warning should mention the underlying error, got {cm.output!r}",
        )

    def test_default_platform_preserves_in_tree_behavior(self):
        # In-tree platforms today don't override get_torch_distributed_backend_str
        # (the base raises NotImplementedError). The dispatcher must silently
        # fall through to the dict so cuda still resolves to nccl, etc.
        self._install(_DefaultPlatform())
        self.assertEqual(get_default_distributed_backend("cuda"), "nccl")

    def test_unknown_device_returns_gloo_default(self):
        self._install(_DefaultPlatform())
        self.assertEqual(get_default_distributed_backend("unobtanium"), "gloo")


if __name__ == "__main__":
    unittest.main()
