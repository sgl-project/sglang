"""CPU-checkable policy tests for the KDA packed-decode CUDA fast path."""

import unittest

from sglang.kernels.ops.attention.kda_packed_decode import (
    _prefer_native_for_capability,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKdaPackedDecodeDispatch(CustomTestCase):
    def test_hopper_keeps_triton_for_all_batch_buckets(self):
        for batch_size in (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            with self.subTest(batch_size=batch_size):
                self.assertFalse(_prefer_native_for_capability(batch_size, (9, 0)))

    def test_sm10_keeps_existing_batch_crossover(self):
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability):
                self.assertFalse(_prefer_native_for_capability(7, capability))
                self.assertTrue(_prefer_native_for_capability(8, capability))

    def test_unknown_architecture_defaults_to_triton(self):
        self.assertFalse(_prefer_native_for_capability(1024, (8, 0)))
        self.assertFalse(_prefer_native_for_capability(1024, (11, 0)))
        self.assertFalse(_prefer_native_for_capability(1024, (12, 0)))
        self.assertFalse(_prefer_native_for_capability(1024, (13, 0)))


if __name__ == "__main__":
    unittest.main()
