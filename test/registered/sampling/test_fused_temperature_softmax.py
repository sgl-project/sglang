"""Correctness tests for fused_temperature_softmax Triton kernel.

Two reference implementations are used:

  1. fp32 reference — logits.float()/temp then softmax in fp32.
     The Triton kernel also promotes to fp32 internally, so this reference
     shares the same precision path and can be checked with tight tolerance.
     This proves **kernel correctness**.

  2. Native-dtype reference — logits.div_(temp) in the original dtype then
     softmax.  The in-place div_ truncates intermediates to bf16/fp16, so
     a looser tolerance is needed.  This covers the **dtype truncation gap**
     between the fused kernel and the existing PyTorch sampling path.
"""

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

# Tolerance table — chosen per dtype to cover known rounding gaps.
# fp32 ref is tight (kernel also runs in fp32 internally).
# Native-dtype ref is looser (div_ truncates to bf16/fp16 before softmax).
_TOL = {
    torch.bfloat16: {"fp32_ref": (1e-5, 1e-5), "native_ref": (2e-2, 1e-1)},
    torch.float16: {"fp32_ref": (1e-5, 1e-5), "native_ref": (1e-3, 1e-2)},
    torch.float32: {"fp32_ref": (1e-5, 1e-5), "native_ref": (1e-5, 1e-5)},
}


def reference_fp32(logits, temperatures):
    """fp32 reference: promotes to fp32 first, matching the kernel's internal precision."""
    return torch.softmax(logits.float() / temperatures.float(), dim=-1)


def reference_native(logits, temperatures):
    """Native-dtype reference: div_ in original dtype, matching the existing PyTorch path."""
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
        self.assertTrue((fused >= 0).all(), "Negative probabilities in fused output")
        row_sums = fused.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(fused, ref, atol=atol, rtol=rtol)

    def _check_both_refs(self, logits, temps, fused, dtype):
        """Check fused output against both fp32 and native-dtype references."""
        tol = _TOL[dtype]
        ref_f32 = reference_fp32(logits, temps)
        ref_nat = reference_native(logits, temps)
        self._check_close(fused, ref_f32, *tol["fp32_ref"])
        self._check_close(fused, ref_nat, *tol["native_ref"])

    # ------------------------------------------------------------------
    # Out-of-place kernel
    # ------------------------------------------------------------------

    def test_basic(self):
        logits = torch.randn(4, 1024, dtype=torch.bfloat16)
        temps = torch.tensor([0.7, 1.0, 1.5, 2.0], dtype=torch.float32).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        self._check_both_refs(logits, temps, fused, torch.bfloat16)

    def test_large_vocab(self):
        logits = torch.randn(8, 128256, dtype=torch.bfloat16)
        temps = torch.full((8, 1), 0.6, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        self._check_both_refs(logits, temps, fused, torch.bfloat16)

    def test_batch_sizes(self):
        for bs in [1, 2, 16, 64, 128, 512]:
            logits = torch.randn(bs, 32000, dtype=torch.bfloat16)
            temps = torch.rand(bs, 1, dtype=torch.float32) * 1.5 + 0.1
            fused = fused_temperature_softmax(logits, temps)
            self._check_both_refs(logits, temps, fused, torch.bfloat16)

    def test_temperature_one(self):
        """Temperature=1.0 should be equivalent to plain softmax."""
        logits = torch.randn(16, 32000, dtype=torch.bfloat16)
        temps = torch.ones(16, 1, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        self._check_both_refs(logits, temps, fused, torch.bfloat16)

    def test_very_low_temperature(self):
        """Very low temperature should produce near-one-hot distribution."""
        logits = torch.randn(4, 100, dtype=torch.float32)
        temps = torch.full((4, 1), 0.01, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
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
        fused = fused_temperature_softmax(logits, temps)
        self._check_both_refs(logits, temps, fused, torch.float16)

    def test_fp32_input(self):
        logits = torch.randn(8, 32000, dtype=torch.float32)
        temps = torch.rand(8, 1, dtype=torch.float32) + 0.5
        fused = fused_temperature_softmax(logits, temps)
        self._check_both_refs(logits, temps, fused, torch.float32)

    def test_mixed_temperatures(self):
        """Each row has a different temperature."""
        logits = torch.randn(8, 32000, dtype=torch.bfloat16)
        temps = torch.tensor(
            [0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 5.0], dtype=torch.float32
        ).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        self._check_both_refs(logits, temps, fused, torch.bfloat16)

    def test_empty_batch(self):
        logits = torch.randn(0, 32000, dtype=torch.bfloat16)
        temps = torch.ones(0, 1, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        self.assertEqual(fused.shape, (0, 32000))

    # ------------------------------------------------------------------
    # In-place kernel
    # ------------------------------------------------------------------

    def test_inplace_basic(self):
        logits = torch.randn(8, 32000, dtype=torch.float32)
        temps = torch.rand(8, 1, dtype=torch.float32) * 1.5 + 0.1
        ref = reference_fp32(logits, temps)
        fused_temperature_softmax_inplace(logits, temps)
        self._check_close(logits.float(), ref, *_TOL[torch.float32]["fp32_ref"])

    def test_inplace_bf16(self):
        logits = torch.randn(8, 32000, dtype=torch.bfloat16)
        temps = torch.rand(8, 1, dtype=torch.float32) + 0.5
        ref_f32 = reference_fp32(logits, temps)
        ref_nat = reference_native(logits, temps)
        fused_temperature_softmax_inplace(logits, temps)
        # In-place stores fp32 probabilities into bf16 buffer, adding another
        # truncation step. Use native-dtype tolerance for both references.
        tol_nat = _TOL[torch.bfloat16]["native_ref"]
        self._check_close(logits.float(), ref_f32, *tol_nat)
        self._check_close(logits.float(), ref_nat, *tol_nat)

    def test_inplace_large_vocab(self):
        logits = torch.randn(4, 128256, dtype=torch.bfloat16)
        temps = torch.full((4, 1), 0.8, dtype=torch.float32)
        ref_f32 = reference_fp32(logits, temps)
        fused_temperature_softmax_inplace(logits, temps)
        tol_nat = _TOL[torch.bfloat16]["native_ref"]
        self._check_close(logits.float(), ref_f32, *tol_nat)

    # ------------------------------------------------------------------
    # Exact known-value correctness (fp32 only — no dtype truncation)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Argmax preservation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

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
            row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4
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

    # ------------------------------------------------------------------
    # Comparison with flashinfer.sampling.softmax
    # ------------------------------------------------------------------

    def test_vs_flashinfer_basic(self):
        logits = torch.randn(4, 1024, dtype=torch.bfloat16)
        temps = torch.tensor([0.7, 1.0, 1.5, 2.0], dtype=torch.float32).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        fi = flashinfer_softmax(logits, temperature=temps.view(-1))
        self._check_close(fused, fi, *_TOL[torch.bfloat16]["native_ref"])

    def test_vs_flashinfer_large_vocab(self):
        logits = torch.randn(8, 128256, dtype=torch.bfloat16)
        temps = torch.full((8, 1), 0.6, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps)
        fi = flashinfer_softmax(logits, temperature=temps.view(-1))
        self._check_close(fused, fi, *_TOL[torch.bfloat16]["native_ref"])

    def test_vs_flashinfer_batch_sizes(self):
        for bs in [1, 16, 64, 128, 512]:
            logits = torch.randn(bs, 32000, dtype=torch.bfloat16)
            temps = torch.rand(bs, 1, dtype=torch.float32) * 1.5 + 0.1
            fused = fused_temperature_softmax(logits, temps)
            fi = flashinfer_softmax(logits, temperature=temps.view(-1))
            self._check_close(fused, fi, *_TOL[torch.bfloat16]["native_ref"])

    def test_vs_flashinfer_scalar_temperature(self):
        logits = torch.randn(16, 32000, dtype=torch.bfloat16)
        temps_2d = torch.full((16, 1), 0.8, dtype=torch.float32)
        fused = fused_temperature_softmax(logits, temps_2d)
        fi = flashinfer_softmax(logits, temperature=0.8)
        self._check_close(fused, fi, *_TOL[torch.bfloat16]["native_ref"])

    def test_vs_flashinfer_mixed_temperatures(self):
        logits = torch.randn(8, 32000, dtype=torch.bfloat16)
        temps = torch.tensor(
            [0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 5.0], dtype=torch.float32
        ).view(-1, 1)
        fused = fused_temperature_softmax(logits, temps)
        fi = flashinfer_softmax(logits, temperature=temps.view(-1))
        self._check_close(fused, fi, *_TOL[torch.bfloat16]["native_ref"])


if __name__ == "__main__":
    unittest.main()
