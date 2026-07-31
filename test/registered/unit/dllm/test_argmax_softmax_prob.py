"""Unit tests for ``dllm/algorithm/base.argmax_and_softmax_prob``.

The dLLM denoise step needs the per-position argmax token and the softmax
probability of that token. ``argmax_and_softmax_prob`` has two portable
implementations of that pair and picks between them by input dtype:

* fp32 input -- ``argmax`` + ``softmax`` + ``gather``, the pre-existing dLLM
  formulation, kept so that platforms which still hand the algorithms an fp32
  ``[tokens, vocab]`` tensor are bitwise unchanged;
* otherwise -- a max reduction plus a vocab-chunked exp-sum, returning
  ``1 / sumexp``. This one never materializes ``[tokens, vocab]`` in fp32, and
  it relies on two things that a "looks equivalent" rewrite silently breaks:
  the argmax token's shifted exponent is exactly 1 (so its probability is the
  reciprocal of the denominator), and the running max must be re-applied when
  folding each chunk into the accumulator.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm.base import argmax_and_softmax_prob

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference(logits: torch.Tensor):
    """The formulation the dLLM algorithms used before this helper existed."""
    ids = torch.argmax(logits, dim=-1)
    prob = torch.gather(
        torch.softmax(logits, dim=-1), dim=-1, index=ids.unsqueeze(-1)
    ).squeeze(-1)
    return ids, prob


# A device may provide a fused kernel; these tests pin the portable paths, so
# the fused one is disabled for all of them.
@patch("sglang.srt.dllm.algorithm.base._argmax_softmax_prob_fused", None)
class TestArgmaxAndSoftmaxProb(CustomTestCase):
    def _logits(self, rows: int, vocab: int, dtype=torch.float32) -> torch.Tensor:
        torch.manual_seed(0)
        # Scaled so the max is well separated: mirrors real logits, where the
        # argmax probability is O(0.1..1) rather than O(1/vocab).
        return (torch.randn(rows, vocab) * 4).to(dtype)

    def test_fp32_input_is_bitwise_the_pre_existing_formulation(self):
        # The fp32 branch exists so non-NPU dLLM keeps its exact numerics.
        # Anything but bitwise equality here is a silent accuracy change for
        # every platform that hands the algorithms fp32 logits.
        logits = self._logits(12, 4099)
        ids, prob = argmax_and_softmax_prob(logits)
        want_ids, want_prob = _reference(logits)
        self.assertTrue(torch.equal(ids, want_ids))
        self.assertTrue(torch.equal(prob, want_prob))

    def test_chunked_path_matches_softmax_across_chunk_boundaries(self):
        # float64 routes to the chunked branch (it is not fp32) and keeps the
        # arithmetic exact, so a mismatch here is the chunking, not rounding.
        # The vocab sizes straddle the chunk boundary in both directions.
        for vocab, chunk in ((4099, 1024), (2048, 2048), (300, 1024), (4096, 1024)):
            with self.subTest(vocab=vocab, vocab_chunk=chunk):
                logits = self._logits(6, vocab, dtype=torch.float64)
                ids, prob = argmax_and_softmax_prob(logits, vocab_chunk=chunk)
                want_ids, want_prob = _reference(logits)
                self.assertTrue(torch.equal(ids, want_ids))
                torch.testing.assert_close(
                    prob, want_prob.float(), rtol=1e-6, atol=1e-7
                )

    def test_chunked_path_survives_a_max_in_the_last_chunk(self):
        # Folding a new chunk must rescale the accumulator by exp(old-new max).
        # Dropping that rescale is invisible unless the running max actually
        # grows late, so put the winner in the final (partial) chunk.
        logits = torch.zeros(1, 2500, dtype=torch.float64)
        logits[0, 2499] = 50.0
        ids, prob = argmax_and_softmax_prob(logits, vocab_chunk=1024)
        want_ids, want_prob = _reference(logits)
        self.assertEqual(ids.tolist(), want_ids.tolist())
        torch.testing.assert_close(prob, want_prob.float(), rtol=1e-6, atol=1e-7)

    def test_low_precision_input_does_not_flip_unmask_decisions(self):
        # The chunked path is what lets a caller keep logits in lm_head dtype.
        # The confidence is compared against a threshold (0.5 by default in
        # JointThreshold), so what has to survive the precision loss is the
        # side of the threshold each position lands on, not the exact value.
        logits = self._logits(64, 4099)
        _, want_prob = _reference(logits)
        ids, prob = argmax_and_softmax_prob(logits.bfloat16())
        self.assertTrue(torch.equal(ids, torch.argmax(logits, dim=-1)))
        self.assertEqual(((prob > 0.5) != (want_prob > 0.5)).sum().item(), 0)

    def test_lm_head_dtype_input_never_materializes_a_full_width_fp32(self):
        # This is the memory claim, and it must hold without the fused kernel --
        # that is the whole reason the caller may hand over lm_head-dtype logits
        # before a kernel package ships. Route a bf16 [rows, vocab] through the
        # portable path and assert no allocation is as wide as the input: the
        # chunked reduction's transients are [rows, vocab_chunk].
        rows, vocab, chunk = 8, 4096, 1024
        logits = self._logits(rows, vocab, dtype=torch.bfloat16)
        widths = []
        real_empty, real_exp = torch.empty_like, torch.Tensor.exp_

        def record(t, *a, **kw):
            widths.append(t.shape[-1] if t.dim() == 2 else 0)
            return real_empty(t, *a, **kw)

        with patch("torch.empty_like", record):
            ids, prob = argmax_and_softmax_prob(logits, vocab_chunk=chunk)

        self.assertEqual(prob.shape, (rows,))
        self.assertTrue(all(w <= chunk for w in widths), widths)


if __name__ == "__main__":
    unittest.main()
