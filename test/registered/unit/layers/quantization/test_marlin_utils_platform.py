"""Unit tests for the Marlin platform gate — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.test.test_utils import CustomTestCase

_MODULE = "sglang.srt.layers.quantization.marlin_utils"
_GET_CAP = f"{_MODULE}.get_device_capability"
_IS_CUDA = f"{_MODULE}._is_cuda"

_, scalar_types = get_scalar_types()


class TestMarlinSupportedPlatform(CustomTestCase):
    """Regression for #33015: Marlin selected on ROCm.

    The Marlin kernels are PTX-only, but support was decided from
    `get_device_capability()` alone. On ROCm that returns the gfx arch, so
    gfx1150 reads back as (11, 5) -> 115 >= 80 and a GPTQ/AutoRound model got
    routed into `gptq_marlin_repack`, which then failed to compile under hipcc
    with "use of undeclared identifier '__cvta_generic_to_shared'".
    """

    def test_not_supported_on_rocm(self):
        with patch(_IS_CUDA, False), patch(_GET_CAP, return_value=(11, 5)):
            self.assertFalse(
                check_marlin_supported(scalar_types.uint4b8, group_size=128)
            )

    def test_supported_on_ampere(self):
        with patch(_IS_CUDA, True), patch(_GET_CAP, return_value=(8, 6)):
            self.assertTrue(
                check_marlin_supported(scalar_types.uint4b8, group_size=128)
            )


if __name__ == "__main__":
    unittest.main()
