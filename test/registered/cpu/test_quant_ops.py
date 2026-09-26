import unittest
from types import SimpleNamespace

import sgl_kernel  # noqa: F401
import torch
from gguf import GGMLQuantizationType

from sglang.srt.layers.quantization.awq.awq import AWQMarlinConfig
from sglang.srt.layers.quantization.awq.schemes.awq_cpu import (
    AWQIntelAMXLinearScheme,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_cpu import (
    GPTQIntelAMXLinearScheme,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


def _pack_q8_0(weight: torch.Tensor) -> torch.Tensor:
    rows = []
    for row in weight.float():
        packed_blocks = []
        for block in row.reshape(-1, 32):
            scale = block.abs().max() / 127.0
            q = torch.clamp(torch.round(block / scale), -127, 127).to(torch.int8)
            packed_blocks.append(
                torch.cat(
                    [
                        scale.reshape(1).to(torch.float16).view(torch.uint8),
                        q.view(torch.uint8),
                    ]
                )
            )
        rows.append(torch.cat(packed_blocks))
    return torch.stack(rows).contiguous()


def _pack_q4_0(weight: torch.Tensor) -> torch.Tensor:
    rows = []
    for row in weight.float():
        packed_blocks = []
        for block in row.reshape(-1, 32):
            scale = block.abs().max() / -8.0
            q = torch.clamp(torch.round(block / scale) + 8, 0, 15).to(torch.uint8)
            packed = q[:16] | (q[16:] << 4)
            packed_blocks.append(
                torch.cat(
                    [scale.reshape(1).to(torch.float16).view(torch.uint8), packed]
                )
            )
        rows.append(torch.cat(packed_blocks))
    return torch.stack(rows).contiguous()


class TestCPUQuantOps(CustomTestCase):
    def test_fused_mul_mat_gguf_cpu_q8_0(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        weight = torch.randn(5, 64, dtype=torch.float32) / 4

        qweight = _pack_q8_0(weight)
        out = fused_mul_mat_gguf(x, qweight, GGMLQuantizationType.Q8_0)

        torch.testing.assert_close(
            out.float(), x.float() @ weight.T, atol=0.08, rtol=0.08
        )

    def test_fused_mul_mat_gguf_cpu_q4_0(self):
        x = torch.randn(2, 64, dtype=torch.float32)
        weight = torch.randn(4, 64, dtype=torch.float32) / 4

        qweight = _pack_q4_0(weight)
        out = fused_mul_mat_gguf(x, qweight, GGMLQuantizationType.Q4_0)

        torch.testing.assert_close(out, x @ weight.T, atol=0.3, rtol=0.3)

    def test_marlin_configs_select_cpu_int4_schemes(self):
        gptq_config = GPTQMarlinConfig(
            weight_bits=4,
            group_size=32,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )
        awq_config = AWQMarlinConfig(
            weight_bits=4,
            group_size=32,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        )

        self.assertIsInstance(
            gptq_config.get_linear_scheme(object()), GPTQIntelAMXLinearScheme
        )
        self.assertIsInstance(
            awq_config.get_linear_scheme(object()), AWQIntelAMXLinearScheme
        )

    def test_gptq_marlin_cpu_rejects_unsupported_desc_act(self):
        gptq_config = GPTQMarlinConfig(
            weight_bits=4,
            group_size=32,
            desc_act=True,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )
        scheme = GPTQIntelAMXLinearScheme(gptq_config)
        layer = torch.nn.Module()

        with self.assertRaisesRegex(ValueError, "desc_act"):
            scheme.create_weights(
                layer=layer,
                input_size_per_partition=64,
                output_partition_sizes=[64],
                input_size=64,
                output_size=64,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *args, **kwargs: None,
            )

    def test_compressed_tensors_wna16_cpu_apply(self):
        x = torch.rand(2, 4096, dtype=torch.bfloat16)
        qweight = torch.randint(-128, 128, (512, 4096), dtype=torch.int32)
        qzeros = torch.randint(0, 10, (32, 512), dtype=torch.int32)
        scales = torch.rand(32, 4096, dtype=torch.bfloat16) / 10
        packed_weight, packed_zero, packed_scales = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                qweight, qzeros, scales, 1
            )
        )
        layer = SimpleNamespace(
            qweight=packed_weight,
            qzeros=packed_zero,
            scales=packed_scales,
        )
        scheme = CompressedTensorsWNA16(
            strategy="group",
            num_bits=4,
            group_size=128,
            symmetric=False,
        )

        out = scheme.apply_weights(layer, x, None)
        ref = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x, packed_weight, packed_zero, packed_scales, None
        )

        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    unittest.main()
