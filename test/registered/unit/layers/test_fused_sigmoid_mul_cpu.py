import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.elementwise as elementwise
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _KernelLaunchForbidden:
    def __getitem__(self, _grid):
        raise AssertionError("CPU fallback should not launch the Triton kernel")


def _reference(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    hidden_dim = attn_output.shape[-1]
    attn_view = attn_output.reshape(-1, hidden_dim)
    gate_view = gate.reshape(-1, hidden_dim)
    return (attn_view * torch.sigmoid(gate_view)).reshape_as(attn_output)


class TestFusedSigmoidMulCPU(unittest.TestCase):
    def test_flat_out_of_place_matches_reference(self):
        torch.manual_seed(0)
        attn_output = torch.randn(5, 16, dtype=torch.float32)
        gate = torch.randn(5, 16, dtype=torch.float32)
        original = attn_output.clone()
        expected = _reference(attn_output, gate)

        with patch.object(
            elementwise, "_fused_sigmoid_mul_kernel", _KernelLaunchForbidden()
        ):
            out = elementwise.fused_sigmoid_mul(attn_output, gate, inplace=False)

        self.assertNotEqual(out.data_ptr(), attn_output.data_ptr())
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(attn_output, original)

    def test_flat_inplace_reuses_storage(self):
        torch.manual_seed(1)
        attn_output = torch.randn(4, 32, dtype=torch.float32)
        gate = torch.randn(4, 32, dtype=torch.float32)
        expected = _reference(attn_output.clone(), gate)
        input_ptr = attn_output.data_ptr()

        with patch.object(
            elementwise, "_fused_sigmoid_mul_kernel", _KernelLaunchForbidden()
        ):
            out = elementwise.fused_sigmoid_mul(attn_output, gate, inplace=True)

        self.assertEqual(out.data_ptr(), input_ptr)
        torch.testing.assert_close(out, expected)

    def test_strided_chunk_gate_bfloat16_uses_cpu_fallback(self):
        torch.manual_seed(2)
        num_tokens, num_heads, head_dim = 6, 4, 8
        q_gate = torch.randn(num_tokens, num_heads, head_dim * 2, dtype=torch.bfloat16)
        _, gate = torch.chunk(q_gate, 2, dim=-1)
        attn_output = torch.randn(
            num_tokens, num_heads * head_dim, dtype=torch.bfloat16
        )
        expected = _reference(attn_output.clone(), gate)
        input_ptr = attn_output.data_ptr()

        self.assertEqual(gate.shape, (num_tokens, num_heads, head_dim))
        self.assertFalse(gate.is_contiguous())

        with patch.object(
            elementwise, "_fused_sigmoid_mul_kernel", _KernelLaunchForbidden()
        ):
            out = elementwise.fused_sigmoid_mul(attn_output, gate, inplace=True)

        self.assertEqual(out.data_ptr(), input_ptr)
        self.assertTrue(torch.equal(out, expected))
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
