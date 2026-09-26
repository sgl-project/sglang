import unittest

import sgl_kernel  # noqa: F401
import torch
from gguf import GGMLQuantizationType

from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf
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


if __name__ == "__main__":
    unittest.main()
