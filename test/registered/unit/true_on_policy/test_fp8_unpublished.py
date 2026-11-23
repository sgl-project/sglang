import unittest
from unittest.mock import patch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import _selected_k_cache_owners
from sglang.srt.runtime_context import get_server_args
from sglang.srt.true_on_policy import fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTopFp8WithoutServerArgs(unittest.TestCase):
    """Kernel tests, benchmarks and offline scripts run the FP8 layers with no server args published;
    the TOP hooks must stay off there rather than raise."""

    def setUp(self):
        with self.assertRaises(ValueError):
            get_server_args()

    def test_top_fp8_is_off_and_not_resolved(self):
        self.assertIsNone(fp8.served_block())
        self.assertFalse(fp8.enabled())
        self.assertFalse(fp8.enabled_b32())
        # Nothing is cached, so a later published config still decides.
        self.assertEqual(fp8._SERVED_BLOCK, [])

    def test_the_layer_hooks_fall_through_to_native(self):
        self.assertFalse(fp8.requant_ue8m0(weight=None, weight_scale=None))
        self.assertIsNone(fp8.maybe_linear(None, None, None, None))
        self.assertIsNone(
            fp8.maybe_expert_forward(
                None, None, weight_dtype=None, activation=None, swiglu_limit=None
            )
        )


class TestTopFp8Device(unittest.TestCase):
    def _validate(self, capability):
        with patch("torch.cuda.get_device_capability", return_value=capability):
            fp8.validate_device()

    def test_hopper_is_admitted_silently(self):
        with self.assertNoLogs("sglang.srt.true_on_policy.fp8", level="WARNING"):
            self._validate((9, 0))

    def test_blackwell_is_admitted_with_a_not_verified_warning(self):
        with self.assertLogs("sglang.srt.true_on_policy.fp8", level="WARNING") as logs:
            self._validate((10, 0))
        self.assertIn("admitted but not verified", logs.output[0])

    def test_other_architectures_are_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "Hopper"):
            self._validate((8, 0))


class TestSelectedOwnersCompilesOnce(unittest.TestCase):
    def test_count_is_a_runtime_argument(self):
        """A constexpr count recompiles the kernel for every new decode batch size."""
        params = {p.name: p for p in _selected_k_cache_owners.params}
        self.assertFalse(params["count"].is_constexpr)


if __name__ == "__main__":
    unittest.main()
