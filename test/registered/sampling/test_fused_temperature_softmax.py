"""Correctness tests for fused_temperature_softmax Triton kernel.

Compares the fused kernel output against the reference PyTorch implementation
(logits.div_(temperatures) followed by torch.softmax) across a range of batch
sizes, vocab sizes, dtypes, and temperature values.
"""

import unittest

import torch

from sglang.srt.layers.fused_sampling import (
    fused_temperature_softmax,
    fused_temperature_softmax_inplace,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-small-1-gpu")


def reference_temperature_softmax(logits, temperatures):
    """Reference implementation: div + softmax (separate kernels)."""
    logits = logits.clone()
    logits.div_(temperatures)
    return torch.softmax(logits, dim=-1).float()


class TestFusedTemperatureSoftmax(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())
        torch.manual_seed(42)

    def _check_close(self, fused, ref, atol=1e-5, rtol=1e-5):
        """Assert outputs are close and both are valid probability distributions."""
        self.assertEqual(fused.shape, ref.shape)
        # Valid probabilities: non-negative, sum to ~1
        self.assertTrue((fused >= 0).all(), f"Negative probabilities in fused output")
        row_sums = fused.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(fused, ref, atol=atol, rtol=rtol)

    # --- out-of-place kernel ---

    def test_basic(self):
        logits = torch.randn(4, 1024, dtype=torch.bfloat16)
        temps = torch.tensor([0.7, 1.0, 1.5, 2.0], dtype=torch.float32).view(-1, 1)
        ref = reference_temperature_softmax(logits, temps)
        fused = fused_temperature_softmax(logits, temps)
        self._check_close(fused, ref, atol=1e-4, rtol=1e-3)

    def test_large_vocab(self):
        logits = torch.randn(8, 128256, dtype=torch.bfloat16)
        temps = torch.full((8, 1), 0.6, dtype=torch.float32)
        ref = reference_temperature_softmax(logits, temps)
        fused = fused_temperature_softmax(logits, temps)
        self._check_close(fused, ref, atol=1e-4, rtol=1e-3)

    def test_batch_sizes(self):
        for bs in [1, 2, 16, 64, 128, 512]:
            logits = torch.randn(bs, 32000, dtype=torch.bfloat16)
            temps = torch.rand(bs, 1, dtype=torch.float32) * 1.5 + 0.1
            ref = reference_temperature_softmax(logits, temps)
            fused = fused_temperature_softmax(logits, temps)
            self._check_close(fused, ref, atol=1e-4, rtol=1e-3)

    def test_temperature_one(self):
        """Temperature=1.0 should be equivalent to plain softmax."""
        logits = torch.randn(16, 32000, dtype=torch.bfloat16)
        temps = torch.ones(16, 1, dtype=torch.float32)
        ref = torch.softmax(logits.float(), dim=-1)
        fused = fused_temperature_softmax(logits, temps)
        self._check_close(fused, ref, atol=1e-4, rtol=1e-3)

    def test_very_low_temperature(self):
        """Very low temperature should produce near-one-hot distribution."""
        logits = torch.randn(4, 1024, dtype=torch.bfloat16)
        temps = torch.full((4, 1), 0.01, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        # Max probability should be very close to 1.0
        max_probs = fused.max(dim=-1).values
        self.assertTrue((max_probs > 0.99).all())

    def test_very_high_temperature(self):
        """Very high temperature should produce near-uniform distribution."""
        logits = torch.randn(4, 1024, dtype=torch.bfloat16)
        temps = torch.full((4, 1), 100.0, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        uniform = 1.0 / 1024
        self.assertTrue(
            (fused - uniform).abs().max() < 0.01,
            "High temperature should produce near-uniform distribution",
        )

    def test_fp16_input(self):
        logits = torch.randn(8, 32000, dtype=torch.float16)
        temps = torch.rand(8, 1, dtype=torch.float32) * 1.5 + 0.1
        ref = reference_temperature_softmax(logits, temps)
        fused = fused_temperature_softmax(logits, temps)
        self._check_close(fused, ref, atol=1e-3, rtol=1e-2)

    def test_fp32_input(self):
        logits = torch.randn(8, 32000, dtype=torch.float32)
        temps = torch.rand(8, 1, dtype=torch.float32) + 0.5
        ref = reference_temperature_softmax(logits, temps)
        fused = fused_temperature_softmax(logits, temps)
        self._check_close(fused, ref, atol=1e-5, rtol=1e-5)

    def test_mixed_temperatures(self):
        """Each row has a different temperature."""
        logits = torch.randn(8, 32000, dtype=torch.bfloat16)
        temps = torch.tensor(
            [0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 5.0], dtype=torch.float32
        ).view(-1, 1)
        ref = reference_temperature_softmax(logits, temps)
        fused = fused_temperature_softmax(logits, temps)
        self._check_close(fused, ref, atol=1e-4, rtol=1e-3)

    def test_empty_batch(self):
        logits = torch.randn(0, 32000, dtype=torch.bfloat16)
        temps = torch.ones(0, 1, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        self.assertEqual(fused.shape, (0, 32000))

    # --- in-place kernel ---

    def test_inplace_basic(self):
        logits = torch.randn(8, 32000, dtype=torch.float32)
        temps = torch.rand(8, 1, dtype=torch.float32) * 1.5 + 0.1
        ref = reference_temperature_softmax(logits, temps)
        fused_temperature_softmax_inplace(logits, temps)
        # In-place writes back to logits in the original dtype
        self._check_close(logits.float(), ref, atol=1e-5, rtol=1e-5)

    def test_inplace_bf16(self):
        logits = torch.randn(8, 32000, dtype=torch.bfloat16)
        temps = torch.rand(8, 1, dtype=torch.float32) + 0.5
        ref = reference_temperature_softmax(logits, temps)
        fused_temperature_softmax_inplace(logits, temps)
        self._check_close(logits.float(), ref, atol=2e-3, rtol=2e-3)

    def test_inplace_large_vocab(self):
        logits = torch.randn(4, 128256, dtype=torch.bfloat16)
        temps = torch.full((4, 1), 0.8, dtype=torch.float32)
        ref = reference_temperature_softmax(logits, temps)
        fused_temperature_softmax_inplace(logits, temps)
        self._check_close(logits.float(), ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
