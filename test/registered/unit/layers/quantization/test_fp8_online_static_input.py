"""Fp8Config takes the online static-activation scheme at construction."""

import unittest

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFp8OnlineStaticInput(CustomTestCase):
    def test_default_is_dynamic(self):
        self.assertFalse(Fp8Config().online_static_input)

    def test_constructor_enables_static(self):
        cfg = Fp8Config(online_static_input=True)
        self.assertTrue(cfg.online_static_input)
        self.assertFalse(cfg.is_checkpoint_fp8_serialized)


if __name__ == "__main__":
    unittest.main()
