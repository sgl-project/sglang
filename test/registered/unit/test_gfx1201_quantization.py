import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGfx1201QuantizationSupport(unittest.TestCase):
    def test_triton_kernels_requires_compatible_matmul_api(self):
        def find_spec_without_matmul_ogs(module):
            return None if module == "triton_kernels.matmul_ogs" else object()

        common.is_triton_kernels_available.cache_clear()
        self.addCleanup(common.is_triton_kernels_available.cache_clear)
        with patch.object(
            common.importlib.util,
            "find_spec",
            side_effect=find_spec_without_matmul_ogs,
        ):
            self.assertFalse(common.is_triton_kernels_available())

    def test_mxfp4_requires_triton_kernels(self):
        props = SimpleNamespace(gcnArchName="gfx1201")
        with (
            patch.object(common.torch.version, "hip", "7.2"),
            patch.object(
                common.torch.cuda, "get_device_properties", return_value=props
            ),
            patch.object(common, "is_triton_kernels_available", return_value=False),
        ):
            self.assertFalse(common.mxfp_supported())

        with (
            patch.object(common.torch.version, "hip", "7.2"),
            patch.object(
                common.torch.cuda, "get_device_properties", return_value=props
            ),
            patch.object(common, "is_triton_kernels_available", return_value=True),
        ):
            self.assertTrue(common.mxfp_supported())


if __name__ == "__main__":
    unittest.main()
