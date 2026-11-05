"""Weight-only FP8 (W8A16) dense linear on XPU (``XPUFp8W8A16LinearMethod`` ->
``sgl_kernel.fp8_linear``): online quantization and pre-quantized 128x128 block-FP8
checkpoints."""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEV = "xpu"
M_VALUES = (1, 8, 300)


def _make_layer(weight: torch.Tensor):
    """A bare ``LinearBase`` holding a bf16 weight, as after online loading."""
    from sglang.srt.layers.linear import LinearBase

    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    # Set by LinearBase.__init__ / create_weights for a single-partition layer.
    layer.params_dtype = torch.bfloat16
    layer.deepgemm_mxfp8_input = False
    layer.logical_widths = [weight.shape[0]]
    layer.input_scale = None
    return layer


def _block_config():
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])


def _make_block_checkpoint_layer(weight: torch.Tensor, logical_widths=None):
    """A bare ``LinearBase`` holding a block-FP8 weight and its scales, as loaded
    from a checkpoint; returns the layer and the exactly dequantized weight."""
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    qweight, scale = per_block_cast_to_fp8(weight)
    layer = _make_layer(qweight)
    layer.weight_scale_inv = torch.nn.Parameter(scale.float(), requires_grad=False)
    layer.logical_widths = logical_widths or [weight.shape[0]]
    n, k = weight.shape
    expand = scale.float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return layer, (qweight.float() * expand[:, :k]).to(torch.bfloat16)


def _online_method():
    from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
        XPUFp8W8A16LinearMethod,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    return XPUFp8W8A16LinearMethod(Fp8Config())


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


@unittest.skipUnless(is_xpu(), "requires XPU")
class TestXPUFp8W8A16Dispatch(CustomTestCase):
    """Which method Fp8Config picks; sgl-kernel-xpu's answer is mocked, so every
    case runs on any XPU runner."""

    def _methods(self, device_supported: bool):
        import sgl_kernel

        from sglang.srt.layers.quantization.fp8 import Fp8Config

        layer = _make_layer(torch.empty(64, 64, dtype=torch.bfloat16))
        configs = {
            "online": Fp8Config(),
            "block": _block_config(),
            "per_tensor_checkpoint": Fp8Config(is_checkpoint_fp8_serialized=True),
            "mxfp8": Fp8Config(use_mxfp8=True),
        }
        with mock.patch.object(
            sgl_kernel,
            "fp8_linear_supported",
            create=True,
            return_value=device_supported,
        ):
            return {
                name: config.get_quant_method(layer, prefix="")
                for name, config in configs.items()
            }

    def test_supported_device(self):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
        )

        methods = self._methods(device_supported=True)
        self.assertIsInstance(methods["online"], XPUFp8W8A16LinearMethod)
        self.assertIsInstance(methods["block"], XPUFp8W8A16LinearMethod)
        self.assertNotIsInstance(
            methods["per_tensor_checkpoint"], XPUFp8W8A16LinearMethod
        )
        self.assertNotIsInstance(methods["mxfp8"], XPUFp8W8A16LinearMethod)

    def test_other_devices_keep_fp8_linear_method(self):
        # Devices the kernel does not take (CRI included) get Fp8LinearMethod
        # itself, exactly as without this change.
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        for name, method in self._methods(device_supported=False).items():
            if name != "mxfp8":
                self.assertIs(type(method), Fp8LinearMethod, name)

    def test_kernel_without_fp8_linear(self):
        # An sgl-kernel-xpu release without fp8_linear keeps Fp8LinearMethod.
        import builtins

        from sglang.srt.hardware_backend.xpu.quantization import fp8_w8a16

        real_import = builtins.__import__

        def no_fp8_linear(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sgl_kernel" and fromlist and "fp8_linear_supported" in fromlist:
                raise ImportError("no fp8_linear_supported")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(builtins, "__import__", side_effect=no_fp8_linear):
            self.assertFalse(fp8_w8a16._fp8_linear_supported_on_device())


@unittest.skipUnless(is_xpu(), "requires XPU")
class TestXPUFp8W8A16Linear(CustomTestCase):
    """The W8A16 path itself, on a device sgl-kernel-xpu runs fp8_linear on."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            _fp8_linear_supported_on_device,
        )

        if not _fp8_linear_supported_on_device():
            raise unittest.SkipTest("sgl_kernel.fp8_linear does not run on this device")

    def test_online_per_channel(self):
        import sgl_kernel

        torch.manual_seed(0)
        n, k = 576, 2048
        weight = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02
        bias = torch.randn(n, dtype=torch.bfloat16, device=DEV)
        layer = _make_layer(weight.clone())
        method = _online_method()
        method.process_weights_after_loading(layer)

        # Fp8LinearMethod's online weight preparation: per-channel scales.
        self.assertTrue(layer.use_fp8_w8a16)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight.t().shape), (n, k))
        self.assertEqual(layer.weight_scale.numel(), n)
        with mock.patch.object(
            sgl_kernel, "fp8_linear", wraps=sgl_kernel.fp8_linear
        ) as kernel:
            for m in M_VALUES:
                x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
                out = method.apply(layer, x, bias)
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertEqual(tuple(out.shape), (m, n))
                self.assertLess(_rel_err(out, F.linear(x, weight, bias)), 0.05)
        self.assertEqual(kernel.call_count, len(M_VALUES))

    def test_3d_input(self):
        weight = torch.randn(256, 256, dtype=torch.bfloat16, device=DEV) * 0.02
        layer = _make_layer(weight.clone())
        method = _online_method()
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 5, 256, dtype=torch.bfloat16, device=DEV)
        out = method.apply(layer, x)
        self.assertEqual(tuple(out.shape), (2, 5, 256))
        self.assertLess(_rel_err(out, F.linear(x, weight)), 0.05)

    def test_unsupported_layer_keeps_existing_path(self):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        # Linear-attention gate projections (N = 32) do not fit the kernel.
        weight = torch.randn(32, 2048, dtype=torch.bfloat16, device=DEV) * 0.02
        layer = _make_layer(weight.clone())
        method = _online_method()
        method.process_weights_after_loading(layer)
        self.assertFalse(layer.use_fp8_w8a16)

        x = torch.randn(4, 2048, dtype=torch.bfloat16, device=DEV)
        # Same weight preparation and GEMM as Fp8LinearMethod.
        reference = Fp8LinearMethod(method.quant_config).apply(layer, x)
        torch.testing.assert_close(method.apply(layer, x), reference)

    def test_block_checkpoint_uses_stored_weights(self):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
        )

        torch.manual_seed(0)
        n, k = 576, 2048
        layer, dequant = _make_block_checkpoint_layer(
            torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02
        )
        stored = layer.weight.data.clone()
        method = XPUFp8W8A16LinearMethod(_block_config())
        method.process_weights_after_loading(layer)

        self.assertTrue(layer.use_fp8_w8a16)
        # No requantization: the checkpoint's FP8 weight and block scales are used.
        self.assertTrue(
            torch.equal(layer.weight.data.view(torch.uint8), stored.view(torch.uint8))
        )
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (5, 16))
        bias = torch.randn(n, dtype=torch.bfloat16, device=DEV)
        for m in M_VALUES:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
            out = method.apply(layer, x, bias)
            self.assertLess(_rel_err(out, F.linear(x, dequant, bias)), 0.02)

    def test_block_checkpoint_misaligned_partitions_keep_existing_path(self):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        # A fused layer whose first partition (96 rows) does not end on a block
        # boundary: its block scales do not tile the fused weight.
        weight = torch.randn(256, 1024, dtype=torch.bfloat16, device=DEV) * 0.02
        layer, _ = _make_block_checkpoint_layer(weight, logical_widths=[96, 160])
        reference_layer, _ = _make_block_checkpoint_layer(
            weight, logical_widths=[96, 160]
        )
        method = XPUFp8W8A16LinearMethod(_block_config())
        method.process_weights_after_loading(layer)
        self.assertFalse(layer.use_fp8_w8a16)

        # Same weights and scale layout as Fp8LinearMethod.
        reference = Fp8LinearMethod(_block_config())
        reference.process_weights_after_loading(reference_layer)
        self.assertEqual(
            layer.weight_scale_inv.stride(), reference_layer.weight_scale_inv.stride()
        )
        x = torch.randn(4, 1024, dtype=torch.bfloat16, device=DEV)
        torch.testing.assert_close(
            method.apply(layer, x), reference.apply(reference_layer, x)
        )


if __name__ == "__main__":
    unittest.main()
