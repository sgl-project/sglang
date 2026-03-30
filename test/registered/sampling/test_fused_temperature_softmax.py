"""Correctness tests for fused_temperature_softmax Triton kernel."""

import unittest

import torch
from flashinfer.sampling import softmax as flashinfer_softmax

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

    # --- exact known-value correctness ---

    def test_known_uniform_logits(self):
        """Identical logits must produce uniform distribution regardless of temperature."""
        logits = torch.zeros(2, 5, dtype=torch.float32)
        temps = torch.tensor([0.5, 2.0], dtype=torch.float32).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        expected = torch.full((2, 5), 0.2, dtype=torch.float32, device=fused.device)
        torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-6)

    def test_known_softmax_values(self):
        """Verify against hand-computed softmax(logits / T)."""
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        temps = torch.tensor([[1.0]], dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        # softmax([1,2,3]) = exp([1,2,3]) / sum(exp([1,2,3]))
        e = torch.exp(logits)
        expected = (e / e.sum(dim=-1, keepdim=True)).to(fused.device)
        torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-6)

    def test_known_softmax_with_temperature(self):
        """Verify softmax([1,2,3] / 0.5) against hand computation."""
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        temps = torch.tensor([[0.5]], dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        scaled = logits / 0.5
        e = torch.exp(scaled)
        expected = (e / e.sum(dim=-1, keepdim=True)).to(fused.device)
        torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-6)

    # --- argmax preservation ---

    def test_argmax_preserved(self):
        """argmax must be invariant to temperature for finite T > 0."""
        logits = torch.randn(64, 32000, dtype=torch.bfloat16)
        original_argmax = logits.float().argmax(dim=-1)
        for t_val in [0.1, 0.5, 1.0, 2.0, 10.0]:
            temps = torch.full((64, 1), t_val, dtype=torch.float32)
            fused = fused_temperature_softmax(logits, temps)
            fused_argmax = fused.argmax(dim=-1)
            self.assertTrue(
                (original_argmax == fused_argmax).all(),
                f"argmax changed at temperature={t_val}",
            )

    # --- numerical stability ---

    def test_large_logits_no_nan(self):
        """Extreme logit magnitudes must not produce NaN or Inf."""
        logits = torch.tensor(
            [[1e6, -1e6, 0.0], [1e4, 1e4 + 1, 1e4 - 1]], dtype=torch.float32
        )
        temps = torch.tensor([[1.0], [0.01]], dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        self.assertFalse(torch.isnan(fused).any(), "NaN in output")
        self.assertFalse(torch.isinf(fused).any(), "Inf in output")
        row_sums = fused.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_large_logits_inplace_no_nan(self):
        """In-place variant: extreme logits must not produce NaN or Inf."""
        logits = torch.tensor(
            [[1e6, -1e6, 0.0], [1e4, 1e4 + 1, 1e4 - 1]], dtype=torch.float32
        )
        temps = torch.tensor([[1.0], [0.01]], dtype=torch.float32)
        fused_temperature_softmax_inplace(logits, temps)
        self.assertFalse(torch.isnan(logits).any(), "NaN in output")
        self.assertFalse(torch.isinf(logits).any(), "Inf in output")

    # --- comparison with flashinfer.sampling.softmax ---

    def test_vs_flashinfer_basic(self):
        logits = torch.randn(4, 1024, dtype=torch.bfloat16)
        temps = torch.tensor([0.7, 1.0, 1.5, 2.0], dtype=torch.float32).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        fi = flashinfer_softmax(logits, temperature=temps.view(-1))
        self._check_close(fused, fi, atol=1e-4, rtol=1e-3)

    def test_vs_flashinfer_large_vocab(self):
        logits = torch.randn(8, 128256, dtype=torch.bfloat16)
        temps = torch.full((8, 1), 0.6, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        fi = flashinfer_softmax(logits, temperature=temps.view(-1))
        self._check_close(fused, fi, atol=1e-4, rtol=1e-3)

    def test_vs_flashinfer_batch_sizes(self):
        for bs in [1, 16, 64, 128, 512]:
            logits = torch.randn(bs, 32000, dtype=torch.bfloat16)
            temps = torch.rand(bs, 1, dtype=torch.float32) * 1.5 + 0.1
            fused = fused_temperature_softmax(logits, temps)
            fi = flashinfer_softmax(logits, temperature=temps.view(-1))
            self._check_close(fused, fi, atol=1e-4, rtol=1e-3)

    def test_vs_flashinfer_scalar_temperature(self):
        logits = torch.randn(16, 32000, dtype=torch.bfloat16)
        temps_2d = torch.full((16, 1), 0.8, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps_2d)
        fi = flashinfer_softmax(logits, temperature=0.8)
        self._check_close(fused, fi, atol=1e-4, rtol=1e-3)

    def test_vs_flashinfer_mixed_temperatures(self):
        logits = torch.randn(8, 32000, dtype=torch.bfloat16)
        temps = torch.tensor(
            [0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 5.0], dtype=torch.float32
        ).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        fi = flashinfer_softmax(logits, temperature=temps.view(-1))
        self._check_close(fused, fi, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
