import unittest
from unittest.mock import patch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSADenseOneShotSupport(CustomTestCase):
    @patch.object(dsa_backend, "_IS_GFX95", False)
    def test_cuda_architecture_support(self):
        for device_sm in (90, 100, 103, 107, 120):
            with self.subTest(device_sm=device_sm):
                self.assertTrue(dsa_backend._supports_dsa_mha_one_shot(device_sm))

        for device_sm in (80, 89, 110, 121):
            with self.subTest(device_sm=device_sm):
                self.assertFalse(dsa_backend._supports_dsa_mha_one_shot(device_sm))

    @patch.object(dsa_backend, "_IS_GFX95", True)
    def test_gfx95_support(self):
        self.assertTrue(dsa_backend._supports_dsa_mha_one_shot(0))


if __name__ == "__main__":
    unittest.main()
