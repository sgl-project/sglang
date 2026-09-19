# SPDX-License-Identifier: Apache-2.0
"""H3 SwiGLU rounding and storage contracts for quantized and BF16 linears."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def eager_swiglu(hidden):
    gate, up = hidden.chunk(2, dim=-1)
    return F.silu(gate) * up


class TestMiniMaxH3SwiGLU(CustomTestCase):
    def test_quantized_input_is_preserved_and_output_is_contiguous(self):
        torch.manual_seed(42)
        for rows in (1, 17, 1024):
            with self.subTest(rows=rows):
                hidden = torch.randn(rows, 28672, device="cuda", dtype=torch.bfloat16)
                original = hidden.clone()
                expected = eager_swiglu(hidden)
                with patch.object(
                    h3,
                    "silu_and_mul_with_activation_rounding",
                    wraps=h3.silu_and_mul_with_activation_rounding,
                ) as fused:
                    actual = h3._silu_mul(hidden, reuse_input=False)
                fused.assert_called_once()
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                self.assertTrue(torch.equal(hidden, original))
                self.assertTrue(actual.is_contiguous())
                self.assertNotEqual(actual.data_ptr(), hidden.data_ptr())

    def test_all_finite_bf16_gate_values_preserve_rounding(self):
        gate = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        gate = gate[torch.isfinite(gate)]
        hidden = torch.cat(
            (
                gate[:, None].expand(-1, 8),
                torch.ones(gate.numel(), 8, device="cuda", dtype=torch.bfloat16),
            ),
            dim=-1,
        )
        expected = eager_swiglu(hidden)
        actual = h3._silu_mul(hidden, reuse_input=False)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )

    def test_existing_inplace_path_preserves_up_half(self):
        hidden = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        expected, up = eager_swiglu(hidden), hidden[:, 128:].clone()
        actual = h3._silu_mul(hidden, reuse_input=True)
        self.assertEqual(actual.data_ptr(), hidden.data_ptr())
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(hidden[:, 128:], up))

    def test_unsupported_inputs_use_eager(self):
        inputs = [
            torch.randn(17, 256, device="cuda", dtype=torch.float32),
            torch.randn(17, 256, device="cuda", dtype=torch.float16),
            torch.randn(17, 14, device="cuda", dtype=torch.bfloat16),
            torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)[:, ::2],
            torch.randn(17, 256, dtype=torch.bfloat16),
        ]
        for hidden in inputs:
            with self.subTest(
                device=hidden.device, dtype=hidden.dtype, stride=hidden.stride()
            ):
                expected = eager_swiglu(hidden)
                with patch.object(h3, "silu_and_mul_with_activation_rounding") as fused:
                    actual = h3._silu_mul(hidden, reuse_input=False)
                fused.assert_not_called()
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
