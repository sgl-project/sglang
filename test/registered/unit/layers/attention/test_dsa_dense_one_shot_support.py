import unittest
from unittest.mock import patch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSADenseOneShotSupport(CustomTestCase):
    def supported(self, device_sm, *, cuda=True, gfx95=False):
        with (
            patch.object(dsa_backend, "_is_cuda", cuda),
            patch.object(dsa_backend, "_IS_GFX95", gfx95),
        ):
            return dsa_backend._supports_dsa_mha_one_shot(device_sm)

    def test_cuda_architecture_support(self):
        for device_sm in (90, 100, 103, 107, 120):
            with self.subTest(device_sm=device_sm):
                self.assertTrue(self.supported(device_sm))

        for device_sm in (80, 89, 110, 121):
            with self.subTest(device_sm=device_sm):
                self.assertFalse(self.supported(device_sm))

    def test_non_cuda_architecture_support(self):
        self.assertTrue(self.supported(0, cuda=False, gfx95=True))
        # gfx1250 also reports major 12; it must not select the CUDA FA4 path.
        self.assertFalse(self.supported(120, cuda=False, gfx95=False))

    def test_runtime_options_keep_sparse_semantics(self):
        supported = dsa_backend._dsa_mha_one_shot_runtime_options_supported

        self.assertTrue(
            supported(
                device_sm=120,
                return_indexer_topk=False,
                skip_softmax_prefill_threshold_scale_factor=None,
            )
        )
        self.assertFalse(
            supported(
                device_sm=120,
                return_indexer_topk=True,
                skip_softmax_prefill_threshold_scale_factor=None,
            )
        )
        self.assertFalse(
            supported(
                device_sm=120,
                return_indexer_topk=False,
                skip_softmax_prefill_threshold_scale_factor=0.0,
            )
        )
        # Existing device paths keep their established behavior.
        self.assertTrue(
            supported(
                device_sm=100,
                return_indexer_topk=True,
                skip_softmax_prefill_threshold_scale_factor=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
