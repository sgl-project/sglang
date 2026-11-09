"""XPU W8A8 int8 linear through the real loading path: compressed-tensors config ->
ColumnParallelLinear (scheme selection and the XPU capability gate) -> weight loader ->
process_weights_after_loading -> forward. Covers 2D and 3D inputs, empty input, and the
checkpoints XPU rejects at load: per-tensor weights, asymmetric or static activations.
GEMM numerics (bias dtypes, tile tails) are in
kernels/ops/gemm/test_int8_scaled_mm_via_int_mm.py."""

import unittest

import torch

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    XPUCompressedTensorsW8A8Int8,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
)
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"
N, K = 640, 1280


def _config(
    weight_strategy="channel", act_strategy="token", dynamic=True, symmetric=True
):
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "ignore": [],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": weight_strategy,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": act_strategy,
                        "symmetric": symmetric,
                        "dynamic": dynamic,
                    },
                }
            },
        }
    )


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an XPU device"
)
class TestXPUW8A8Int8Linear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _layer(self, **config_kwargs):
        # Construction selects the scheme and runs the capability gate against the
        # real device (sgl_kernel.query_device), so a rejected checkpoint raises here.
        return ColumnParallelLinear(
            input_size=K,
            output_size=N,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=_config(**config_kwargs),
            prefix="model.layers.0.mlp.fc1",
            parallel_group="replicated",
        )

    def _build(self):
        torch.manual_seed(0)
        layer = self._layer().to(DEVICE)
        self.assertIsInstance(layer.scheme, XPUCompressedTensorsW8A8Int8)
        w = torch.randn(N, K, device=DEVICE) / 10
        w_scale = w.abs().amax(dim=1, keepdim=True) / 127
        w_int8 = torch.round(w / w_scale).clamp(-127, 127).to(torch.int8)
        load_linear_weights(layer, weight=w_int8, weight_scale=w_scale)
        b = torch.randn(N, device=DEVICE, dtype=torch.bfloat16) / 10
        layer.bias.data.copy_(b)
        layer.quant_method.process_weights_after_loading(layer)
        return layer, w_int8.float() * w_scale, b

    def test_forward_matches_reference(self):
        layer, w_dequant, b = self._build()
        for shape in ((5, K), (2, 3, K)):  # 2D and 3D inputs
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
                out, _ = layer(x)
                self.assertEqual(tuple(out.shape), (*shape[:-1], N))
                assert_output_close(
                    self,
                    out,
                    x.float() @ w_dequant.t() + b.float(),
                    rtol=5e-2,
                    atol=1e-1,
                )

    def test_empty_input(self):
        layer, *_ = self._build()
        out, _ = layer(torch.empty(0, K, device=DEVICE, dtype=torch.bfloat16))
        self.assertEqual(tuple(out.shape), (0, N))

    def test_rejects_per_tensor_weights(self):
        """Regression: per-tensor int8 weights reached XPU and went through
        requantize_with_max_scale, which requantizes them through FP8."""
        with self.assertRaisesRegex(NotImplementedError, "per-channel"):
            self._layer(weight_strategy="tensor")

    def test_rejects_asymmetric_activations(self):
        with self.assertRaisesRegex(NotImplementedError, "symmetric"):
            self._layer(symmetric=False)

    def test_rejects_static_activations(self):
        # XPU has no kernel path for static per-tensor activation scales: the checkpoint
        # must fail at load (today through the capability gate) instead of being served.
        with self.assertRaises((RuntimeError, NotImplementedError)):
            self._layer(act_strategy="tensor", dynamic=False)


if __name__ == "__main__":
    unittest.main()
