import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRocmNoGpuGuards(CustomTestCase):
    def tearDown(self):
        from sglang.srt.layers.quantization import fp8_kernel
        from sglang.srt.utils import common

        common._clear_hip_gcn_arch_name_cache()
        fp8_kernel.is_fp8_fnuz.cache_clear()

    def test_common_gfx_helpers_return_false_without_visible_gpu(self):
        from sglang.srt.utils import common

        common._clear_hip_gcn_arch_name_cache()
        with patch.object(common.torch.version, "hip", "7.2.0"), patch.object(
            common.torch.cuda, "is_available", return_value=False
        ), patch.object(
            common.torch.cuda,
            "get_device_properties",
            side_effect=AssertionError("must not query device properties"),
        ):
            self.assertFalse(common.mxfp_supported())
            self.assertFalse(common.is_gfx95_supported())
            self.assertFalse(common.is_gfx942_supported())

    def test_common_gfx_helpers_swallow_no_hip_gpu_runtime_error(self):
        from sglang.srt.utils import common

        common._clear_hip_gcn_arch_name_cache()
        with patch.object(common.torch.version, "hip", "7.2.0"), patch.object(
            common.torch.cuda, "is_available", return_value=True
        ), patch.object(
            common.torch.cuda,
            "get_device_properties",
            side_effect=RuntimeError("No HIP GPUs are available"),
        ):
            self.assertFalse(common.mxfp_supported())
            self.assertFalse(common.is_gfx95_supported())
            self.assertFalse(common.is_gfx942_supported())

    def test_common_gfx_helpers_preserve_real_gpu_detection(self):
        from sglang.srt.utils import common

        common._clear_hip_gcn_arch_name_cache()
        fake_props = SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        with patch.object(common.torch.version, "hip", "7.2.0"), patch.object(
            common.torch.cuda, "is_available", return_value=True
        ), patch.object(
            common.torch.cuda, "get_device_properties", return_value=fake_props
        ):
            self.assertTrue(common.mxfp_supported())
            self.assertTrue(common.is_gfx95_supported())
            self.assertFalse(common.is_gfx942_supported())

    def test_common_gfx_helpers_do_not_cache_no_gpu_false(self):
        from sglang.srt.utils import common

        common._clear_hip_gcn_arch_name_cache()
        fake_props = SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        with patch.object(common.torch.version, "hip", "7.2.0"):
            with patch.object(common.torch.cuda, "is_available", return_value=False):
                self.assertFalse(common.is_gfx95_supported())

            with patch.object(common.torch.cuda, "is_available", return_value=True):
                with patch.object(
                    common.torch.cuda, "get_device_properties", return_value=fake_props
                ):
                    self.assertTrue(common.is_gfx95_supported())

    def test_fp8_fnuz_returns_false_without_visible_gpu(self):
        from sglang.srt.layers.quantization import fp8_kernel

        fp8_kernel.is_fp8_fnuz.cache_clear()
        with patch.object(fp8_kernel, "_is_hip", True), patch.object(
            fp8_kernel.torch.cuda, "is_available", return_value=False
        ), patch.object(
            fp8_kernel.torch.cuda,
            "get_device_properties",
            side_effect=AssertionError("must not query device properties"),
        ):
            self.assertFalse(fp8_kernel.is_fp8_fnuz())

    def test_fp8_fnuz_preserves_gfx942_detection(self):
        from sglang.srt.layers.quantization import fp8_kernel

        fp8_kernel.is_fp8_fnuz.cache_clear()
        fake_props = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with patch.object(fp8_kernel, "_is_hip", True), patch.object(
            fp8_kernel.torch.cuda, "is_available", return_value=True
        ), patch.object(
            fp8_kernel.torch.cuda, "get_device_properties", return_value=fake_props
        ):
            self.assertTrue(fp8_kernel.is_fp8_fnuz())

    def test_rowwise_scaled_mm_returns_false_without_device_capability(self):
        from sglang.srt.layers.quantization import fp8_utils

        with patch.object(fp8_utils, "_is_hip", True), patch.object(
            fp8_utils, "get_device_capability", return_value=(None, None)
        ):
            self.assertFalse(fp8_utils.use_rowwise_torch_scaled_mm())

    def test_rowwise_scaled_mm_preserves_supported_capability(self):
        from sglang.srt.layers.quantization import fp8_utils

        with patch.object(fp8_utils, "_is_hip", True), patch.object(
            fp8_utils, "get_device_capability", return_value=(9, 5)
        ), patch.object(fp8_utils, "torch_release", (2, 7)):
            self.assertTrue(fp8_utils.use_rowwise_torch_scaled_mm())

    def test_quark_mxfp4_linear_no_gpu_stub_raises_not_implemented(self):
        from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4

        if getattr(quark_w4a4_mxfp4, "_has_visible_hip_device", False):
            self.skipTest("visible HIP device imports real AITer kernels")

        with self.assertRaisesRegex(NotImplementedError, "visible at import time"):
            quark_w4a4_mxfp4.dynamic_mxfp4_quant(None)

    def test_quark_mxfp4_moe_no_gpu_processing_raises_not_implemented(self):
        from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4_moe

        if getattr(quark_w4a4_mxfp4_moe, "_has_visible_hip_device", False):
            self.skipTest("visible HIP device imports real AITer kernels")

        scheme = object.__new__(quark_w4a4_mxfp4_moe.QuarkW4A4MXFp4MoE)
        with self.assertRaisesRegex(NotImplementedError, "visible AMD ROCm device"):
            scheme.process_weights_after_loading(object())

    def test_quark_int4fp8_moe_no_gpu_processing_raises_not_implemented(self):
        import sglang.srt.layers.quantization.quark_int4fp8_moe as quark_int4fp8_moe

        if getattr(quark_int4fp8_moe, "_has_visible_hip_device", False):
            self.skipTest("visible HIP device imports real AITer kernels")

        method = object.__new__(quark_int4fp8_moe.QuarkInt4Fp8MoEMethod)
        with self.assertRaisesRegex(NotImplementedError, "visible AMD ROCm device"):
            method.process_weights_after_loading(object())


if __name__ == "__main__":
    unittest.main()
