"""CPU checks for the BF16 wo_a linear path and in-place weight updates."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_wo_a import (
    apply_npu_wo_a_bf16,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuWoABf16Linear(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)

    @staticmethod
    def _layer(rank=32, dim=128):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.randn(rank, dim, dtype=torch.bfloat16), requires_grad=False
            ),
        )
        return layer

    def test_forward_preserves_weight_storage_values_and_loader_attributes(self):
        layer = self._layer()
        parameter = layer.weight
        original = parameter.detach().clone()
        loader = object()
        parameter.weight_loader = loader
        parameter.output_dim = 0
        pointer = parameter.data_ptr()
        stride = parameter.stride()
        o = torch.randn(8, 1, 128, dtype=torch.bfloat16)

        actual = apply_npu_wo_a_bf16(o, parameter)

        expected = torch.einsum("tgd,grd->tgr", o, original.unsqueeze(0))
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0002)
        self.assertIs(layer.weight, parameter)
        self.assertIs(layer.weight.weight_loader, loader)
        self.assertEqual(layer.weight.output_dim, 0)
        self.assertEqual(layer.weight.data_ptr(), pointer)
        self.assertEqual(layer.weight.stride(), stride)
        torch.testing.assert_close(layer.weight, original, rtol=0, atol=0)
        self.assertTrue(layer.weight.is_contiguous())
        self.assertEqual(layer.weight.untyped_storage().nbytes(), original.numel() * 2)
        self.assertEqual(list(layer.state_dict()), ["weight"])

    def test_matches_einsum_and_fp32_for_contiguous_and_sliced_activations(self):
        for tokens, rank, dim, padding in (
            (1, 32, 128, 1),
            (2, 64, 256, 8),
            (4, 64, 256, 8),
            (8, 1024, 4096, 1),
            (8, 1024, 4096, 8),  # sliced output of attention with padded heads
            (16, 64, 256, 8),
            (37, 64, 256, 1),
        ):
            with self.subTest(tokens=tokens, rank=rank, dim=dim, padding=padding):
                layer = self._layer(rank, dim)
                original = layer.weight.detach().clone()
                o = torch.randn(tokens, padding, dim, dtype=torch.bfloat16)[:, :1, :]
                expected = torch.einsum("tgd,grd->tgr", o, original.unsqueeze(0))
                actual = apply_npu_wo_a_bf16(o, layer.weight)
                # Different GEMM paths can change BF16 rounding near zero.
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0002)
                reference_fp32 = torch.einsum(
                    "tgd,grd->tgr", o.float(), original.float().unsqueeze(0)
                )
                torch.testing.assert_close(
                    actual.float(), reference_fp32, rtol=0.008, atol=0.0002
                )
                self.assertEqual(actual.shape, (tokens, 1, rank))
                self.assertTrue(actual.is_contiguous())
                # Prefill retains einsum and the original weight layout.
                prefill = torch.einsum(
                    "tgd,grd->tgr", o, layer.weight.view(1, rank, dim)
                )
                torch.testing.assert_close(prefill, expected, rtol=0.008, atol=0.0002)

    def test_reload_updates_existing_views_without_changing_address(self):
        for tp_rank in (0, 3, 7):
            with self.subTest(tp_rank=tp_rank):
                layer = self._layer()
                captured_weight = layer.weight.detach()
                pointer = captured_weight.data_ptr()
                o = torch.randn(8, 1, 128, dtype=torch.bfloat16)
                before = apply_npu_wo_a_bf16(o, layer.weight)

                # Standard TP loading and direct updates both copy into .data.
                checkpoint = torch.randn(8 * 32, 128, dtype=torch.bfloat16)
                local_weight = checkpoint.narrow(0, tp_rank * 32, 32)
                layer.weight.data.copy_(local_weight)

                self.assertEqual(layer.weight.data_ptr(), pointer)
                expected = torch.einsum("tgd,grd->tgr", o, local_weight.unsqueeze(0))
                actual = apply_npu_wo_a_bf16(o, layer.weight)
                replay_view = apply_npu_wo_a_bf16(o, captured_weight)
                self.assertFalse(torch.equal(before, actual))
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0002)
                torch.testing.assert_close(
                    replay_view, expected, rtol=0.008, atol=0.0002
                )


if __name__ == "__main__":
    unittest.main()
