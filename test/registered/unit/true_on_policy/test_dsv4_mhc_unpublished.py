import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.runtime_context import get_server_args
from sglang.srt.true_on_policy.dsv4 import top_mhc_pre_num_splits
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMhcSplitWithoutServerArgs(unittest.TestCase):
    """mhc_pre and mhc_fused_post_pre run in kernel tests and benchmarks with no server args published;
    the TOP split pin must stay off there instead of raising (test_mhc_kernels.py)."""

    def setUp(self):
        with self.assertRaises(ValueError):
            get_server_args()

    def test_no_pin_without_server_args(self):
        self.assertIsNone(top_mhc_pre_num_splits())

    def test_the_mhc_pre_split_is_the_native_heuristic(self):
        try:
            from sglang.kernels.ops.layernorm.mhc import _compute_num_split_for_mhc_pre
        except ImportError as e:
            self.skipTest(f"mhc kernels cannot be imported here: {e!r}")
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=132),
        ):
            # 64 tokens fill one block row: min(132 // 1, (4 * 4096 / 64) // 4).
            self.assertEqual(_compute_num_split_for_mhc_pre(64, 4 * 4096), 64)


if __name__ == "__main__":
    unittest.main()
