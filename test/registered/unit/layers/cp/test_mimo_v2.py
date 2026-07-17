import unittest

import torch

from sglang.srt.layers.cp.mimo_v2 import repack_mimo_v2_fused_qkv_block_fp8
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _block_quantize_fixture(weight, block_size):
    block_n, block_k = block_size
    n, k = weight.shape
    padded = torch.zeros(
        (
            (n + block_n - 1) // block_n * block_n,
            (k + block_k - 1) // block_k * block_k,
        ),
        dtype=torch.float32,
    )
    padded[:n, :k] = weight
    blocks = padded.view(
        padded.shape[0] // block_n,
        block_n,
        padded.shape[1] // block_k,
        block_k,
    )
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
    quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.view_as(padded)[:n, :k].contiguous(), scale.contiguous()


class TestMiMoV2CPWeightAdapter(CustomTestCase):
    def test_repack_fused_tp4_qkv_requantizes_across_block_boundaries(self):
        block_size = [4, 2]
        q_values = [1.0, 2.0, 4.0, 8.0]
        k_values = [16.0, 32.0, 64.0, 128.0]
        v_values = [-1.0, -2.0, -4.0, -8.0]

        checkpoint_groups = []
        for rank in range(4):
            checkpoint_groups.append(
                torch.cat(
                    [
                        torch.full((8, 4), q_values[rank]),
                        torch.full((6, 4), k_values[rank]),
                        torch.full((4, 4), v_values[rank]),
                    ]
                )
            )

        quantized_groups = [
            _block_quantize_fixture(group, block_size) for group in checkpoint_groups
        ]
        checkpoint_weight = torch.cat([item[0] for item in quantized_groups])
        checkpoint_scale = torch.cat([item[1] for item in quantized_groups])

        repacked_weight, repacked_scale = repack_mimo_v2_fused_qkv_block_fp8(
            checkpoint_weight,
            checkpoint_scale,
            q_rows=32,
            k_rows=24,
            v_rows=16,
            checkpoint_tp_size=4,
            block_size=block_size,
            output_dtype=torch.float32,
        )
        actual = block_quant_dequant(
            repacked_weight,
            repacked_scale,
            block_size,
            torch.float32,
        )
        expected = torch.cat(
            [
                *(torch.full((8, 4), value) for value in q_values),
                *(torch.full((6, 4), value) for value in k_values),
                *(torch.full((4, 4), value) for value in v_values),
            ]
        )

        self.assertEqual(tuple(checkpoint_weight.shape), (72, 4))
        self.assertEqual(tuple(checkpoint_scale.shape), (20, 2))
        self.assertEqual(tuple(repacked_weight.shape), (72, 4))
        self.assertEqual(tuple(repacked_scale.shape), (18, 2))
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
