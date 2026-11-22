import unittest
from unittest.mock import patch

from sglang.srt.layers.attention import dsa_backend as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaMhaOneShotDeviceGate(unittest.TestCase):
    """SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD is honored only on devices
    that have a dense one-shot kernel; SM12x now has one (SM12x FA4)."""

    def supported(self, device_sm, *, cuda=True, gfx95=False):
        with (
            patch.object(module, "_is_cuda", cuda),
            patch.object(module, "_IS_GFX95", gfx95),
        ):
            return module.dsa_mha_one_shot_device_supported(device_sm)

    def test_cuda_devices_with_a_dense_kernel(self):
        for sm in (90, 100, 103, 107, 120, 121):
            with self.subTest(sm=sm):
                self.assertTrue(self.supported(sm))

    def test_cuda_devices_without_a_dense_kernel(self):
        for sm in (80, 86, 89, 110, 130):
            with self.subTest(sm=sm):
                self.assertFalse(self.supported(sm))

    def test_gfx95_is_supported_and_other_rocm_is_not(self):
        self.assertTrue(self.supported(0, cuda=False, gfx95=True))
        # gfx1250 also reports a major of 12; without CUDA it must not match.
        self.assertFalse(self.supported(120, cuda=False, gfx95=False))


if __name__ == "__main__":
    unittest.main()
