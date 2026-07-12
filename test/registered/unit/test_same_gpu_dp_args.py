"""Unit tests for same-GPU data parallelism argument validation (gpu_id_step=0)."""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kwargs):
    return ServerArgs(model_path="dummy", **kwargs)


class TestSameGpuDpArgs(CustomTestCase):
    def test_same_gpu_dp_accepted(self):
        args = _args(dp_size=2, gpu_id_step=0, max_total_tokens=65536)
        args._check_same_gpu_dp()
        self.assertEqual(args.gpu_id_step, 0)

    def test_default_step_unchanged(self):
        args = _args()
        args._check_same_gpu_dp()
        self.assertEqual(args.gpu_id_step, 1)

    def test_negative_step_rejected(self):
        with self.assertRaisesRegex(AssertionError, "non-negative"):
            _args(gpu_id_step=-1)._check_same_gpu_dp()

    def test_same_gpu_dp_requires_dp_size(self):
        with self.assertRaisesRegex(AssertionError, "dp_size > 1"):
            _args(gpu_id_step=0, max_total_tokens=65536)._check_same_gpu_dp()

    def test_same_gpu_dp_requires_tp1(self):
        with self.assertRaisesRegex(AssertionError, "tp_size and pp_size"):
            _args(
                dp_size=2, tp_size=2, gpu_id_step=0, max_total_tokens=65536
            )._check_same_gpu_dp()

    def test_same_gpu_dp_requires_token_cap(self):
        with self.assertRaisesRegex(AssertionError, "max-total-tokens"):
            _args(dp_size=2, gpu_id_step=0)._check_same_gpu_dp()


if __name__ == "__main__":
    unittest.main()
