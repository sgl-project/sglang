"""Weight-only FP8 (W8A16) execution of FP8 dense layers on XPU: ``Fp8LinearMethod`` runs
``sgl_kernel.fp8_linear`` for online quantization and pre-quantized 128x128 block-FP8
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


def _method(config=None):
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    return Fp8LinearMethod(config or Fp8Config())


def _w8a8_method(config=None):
    """Fp8LinearMethod with the W8A16 backend off: the existing W8A8 path."""
    method = _method(config)
    method.use_xpu_w8a16 = False
    return method


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


@unittest.skipUnless(is_xpu(), "requires XPU")
class TestXPUFp8W8A16Dispatch(CustomTestCase):
    """Which execution backend Fp8LinearMethod picks; sgl-kernel-xpu's answer is
    mocked, so every case runs on any XPU runner."""

    def _backends(self, device_supported: bool):
        import sgl_kernel

        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

        layer = _make_layer(torch.empty(64, 64, dtype=torch.bfloat16))
        configs = {
            "online": Fp8Config(),
            "block": _block_config(),
            "per_tensor_checkpoint": Fp8Config(is_checkpoint_fp8_serialized=True),
            "mxfp8": Fp8Config(use_mxfp8=True),
        }
        backends = {}
        with mock.patch.object(
            sgl_kernel,
            "fp8_linear_supported",
            create=True,
            return_value=device_supported,
        ):
            for name, config in configs.items():
                method = config.get_quant_method(layer, prefix="")
                # W8A16 is an execution backend of Fp8LinearMethod, not a new method.
                self.assertIs(type(method), Fp8LinearMethod, name)
                backends[name] = method.use_xpu_w8a16
        return backends

    def test_supported_device(self):
        backends = self._backends(device_supported=True)
        self.assertTrue(backends["online"])
        self.assertTrue(backends["block"])
        self.assertFalse(backends["per_tensor_checkpoint"])
        self.assertFalse(backends["mxfp8"])

    def test_other_devices_keep_w8a8(self):
        # Devices the kernel does not take (CRI included) keep the W8A8 path.
        self.assertFalse(any(self._backends(device_supported=False).values()))

    def test_kernel_without_fp8_linear(self):
        # An sgl-kernel-xpu release without fp8_linear keeps the W8A8 path.
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
        method = _method()
        method.process_weights_after_loading(layer)

        # Fp8LinearMethod's online weight preparation: per-channel scales.
        self.assertTrue(layer.xpu_fp8_w8a16)
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
        method = _method()
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 5, 256, dtype=torch.bfloat16, device=DEV)
        out = method.apply(layer, x)
        self.assertEqual(tuple(out.shape), (2, 5, 256))
        self.assertLess(_rel_err(out, F.linear(x, weight)), 0.05)

    def test_unsupported_layer_keeps_existing_path(self):
        # Linear-attention gate projections (N = 32) do not fit the kernel.
        weight = torch.randn(32, 2048, dtype=torch.bfloat16, device=DEV) * 0.02
        layer = _make_layer(weight.clone())
        method = _method()
        method.process_weights_after_loading(layer)
        self.assertFalse(layer.xpu_fp8_w8a16)

        x = torch.randn(4, 2048, dtype=torch.bfloat16, device=DEV)
        # Same weight preparation and GEMM as Fp8LinearMethod.
        reference = _w8a8_method().apply(layer, x)
        torch.testing.assert_close(method.apply(layer, x), reference)

    def test_block_checkpoint_uses_stored_weights(self):

        torch.manual_seed(0)
        n, k = 576, 2048
        layer, dequant = _make_block_checkpoint_layer(
            torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02
        )
        stored = layer.weight.data.clone()
        method = _method(_block_config())
        method.process_weights_after_loading(layer)

        self.assertTrue(layer.xpu_fp8_w8a16)
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
        # A fused layer whose first partition (96 rows) does not end on a block
        # boundary: its block scales do not tile the fused weight.
        weight = torch.randn(256, 1024, dtype=torch.bfloat16, device=DEV) * 0.02
        layer, _ = _make_block_checkpoint_layer(weight, logical_widths=[96, 160])
        reference_layer, _ = _make_block_checkpoint_layer(
            weight, logical_widths=[96, 160]
        )
        method = _method(_block_config())
        method.process_weights_after_loading(layer)
        self.assertFalse(layer.xpu_fp8_w8a16)

        # Same weights, scale layout and output as the W8A8 path.
        reference = _w8a8_method(_block_config())
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
