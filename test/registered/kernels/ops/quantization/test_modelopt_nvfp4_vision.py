"""Serialized NVFP4 vision linears preserve token order across leading dimensions.

Before the ND activation fix, a [sequence, batch, hidden] input reaches the
matrix-only quantization/GEMM path without flattening and cannot return the
same result as its explicit matrix view. These tests use real FlashInfer
kernels and checkpoint-format weights through the layer's weight loader.
"""

import unittest

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.runtime_context import create_context, use_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import (
    break_fp4_bytes,
    dequantize_nvfp4_to_dtype,
)
from sglang.test.test_utils import CustomTestCase

# The single-GPU small pool is Blackwell (5090); no model download is needed.
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestModelOptNvfp4Vision(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "FlashInfer NVFP4 CUTLASS requires NVIDIA Blackwell"
            )
        if torch.cuda.get_device_capability() not in (
            (10, 0),
            (10, 3),
            (12, 0),
            (12, 1),
        ):
            raise unittest.SkipTest(
                "FlashInfer NVFP4 CUTLASS regression targets SM100/103/120/121"
            )

    def _make_layer(self):
        n, k = 160, 256
        prefix = "model.visual.blocks.0.mlp.linear_fc1"
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "packed_modules_mapping": {},
                "quantized_layers": {prefix: {"quant_algo": "NVFP4"}},
            }
        )
        with torch.device("cuda"):
            layer = ReplicatedLinear(
                input_size=k,
                output_size=n,
                bias=True,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=prefix,
            )
            rows = torch.arange(n)[:, None]
            columns = torch.arange(k // 2)[None, :]
            low = (rows + columns) % 16
            high = (2 * rows + 3 * columns + 1) % 16
            weight = (low | (high << 4)).to(torch.uint8)
            scale = ((rows + torch.arange(k // 16)[None, :]) % 4 + 1).float() / 4
            scale = scale.to(torch.float8_e4m3fn)
            scale_2 = torch.tensor(0.125, dtype=torch.float32)
            bias = torch.linspace(-0.25, 0.25, n, dtype=torch.bfloat16)
            checkpoint = {
                "weight": weight,
                "weight_scale": scale,
                "weight_scale_2": scale_2,
                "input_scale": torch.tensor(0.002, dtype=torch.float32),
                "bias": bias,
            }
            for name, loaded_weight in checkpoint.items():
                parameter = getattr(layer, name)
                parameter.weight_loader(parameter, loaded_weight)
            self.assertIsInstance(layer.quant_method, ModelOptFp4LinearMethod)
            self.assertEqual(layer.quant_method.quant_mode, "w4a4")
            layer.quant_method.process_weights_after_loading(layer)

        weight_reference = (
            break_fp4_bytes(weight)
            * scale.float().repeat_interleave(16, dim=-1)
            * scale_2
        )
        return layer, weight_reference

    @torch.inference_mode()
    def test_nd_and_prequantized_inputs_match_matrix_and_dequantized_reference(self):
        context = create_context(
            ServerArgs(
                model_path="dummy", fp4_gemm_runner_backend="flashinfer_cutlass"
            ),
            role="test",
        )
        previous_backend = fp4_utils.FP4_GEMM_RUNNER_BACKEND
        with use_context(context):
            try:
                fp4_utils.initialize_fp4_gemm_config()
                layer, weight_reference = self._make_layer()
                generator = torch.Generator(device="cuda").manual_seed(7)
                noncontiguous = (
                    torch.randn(
                        (2, 3, layer.input_size),
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    / 10
                ).transpose(0, 1)
                self.assertFalse(noncontiguous.is_contiguous())
                contiguous = noncontiguous.contiguous()
                matrix = contiguous.reshape(-1, layer.input_size)
                matrix_output, output_bias = layer(matrix)
                self.assertIsNone(output_bias)
                self.assertTrue(torch.isfinite(matrix_output).all().item())

                quantized = fp4_utils.fp4_quantize(matrix, layer.input_scale_inv)
                activation_reference = dequantize_nvfp4_to_dtype(
                    *quantized, layer.input_scale_inv, torch.float32
                )
                reference = (activation_reference @ weight_reference.T).to(
                    torch.bfloat16
                ) + layer.bias
                torch.testing.assert_close(
                    matrix_output.float(), reference.float(), rtol=5e-2, atol=5e-2
                )

                expected = matrix_output.reshape(3, 2, layer.output_size)
                for name, activation in (
                    ("contiguous", contiguous),
                    ("noncontiguous", noncontiguous),
                ):
                    with self.subTest(layout=name):
                        output, output_bias = layer(activation)
                        self.assertIsNone(output_bias)
                        self.assertEqual(output.shape, (3, 2, layer.output_size))
                        self.assertTrue(torch.isfinite(output).all().item())
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)

                layer._accepts_prequantized_fp4 = True
                prequantized_output, output_bias = layer(quantized)
                self.assertIsNone(output_bias)
                torch.testing.assert_close(
                    prequantized_output, matrix_output, rtol=0, atol=0
                )
            finally:
                fp4_utils.FP4_GEMM_RUNNER_BACKEND = previous_backend


if __name__ == "__main__":
    unittest.main()
