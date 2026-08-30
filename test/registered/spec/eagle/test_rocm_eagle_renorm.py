"""Numerical-equivalence tests for the ROCm EAGLE top-k/top-p renorm.

Guards ``eagle_utils._renorm_top_k_top_p_hip`` (the Triton replacement for the
CUDA/MUSA-only ``sgl_kernel`` ``top_k_renorm_prob`` + ``top_p_renorm_prob``)
against an independent torch nucleus reference. A regression in the pivot search
(wrong iteration count, wrong threshold, broken top-k, or a ``top_p == 1.0`` that
stops being an exact no-op) turns a case red. The kernel is device-agnostic, so
this runs on any triton-capable GPU (CUDA in CI); skipped without one.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Runs on any triton GPU; AMD registration exercises the actual ROCm target path.
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_RUNNABLE = torch.cuda.is_available() and _HAS_TRITON


def _torch_nucleus_reference(probs, top_ks, top_ps):
    """top-k renorm then nucleus (top-p) renorm, mirroring sgl_kernel semantics.

    Always keeps the argmax token (like the pivot kernel and sgl_kernel), so it
    agrees on the top_p -> 0 edge instead of zeroing the row.
    """
    vocab = probs.shape[-1]
    order = torch.arange(vocab, device=probs.device).view(1, -1)
    # top-k
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    sorted_probs = torch.where(
        order < top_ks.view(-1, 1), sorted_probs, torch.zeros_like(sorted_probs)
    )
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    p = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    # top-p (nucleus)
    sorted_probs, sorted_idx = p.sort(dim=-1, descending=True)
    csum = sorted_probs.cumsum(dim=-1)
    keep = (csum - sorted_probs) < top_ps.view(-1, 1)
    keep[..., 0] = True
    sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.zeros_like(p).scatter_(-1, sorted_idx, sorted_probs)


@unittest.skipUnless(_RUNNABLE, "requires a triton-capable GPU")
class TestRocmEagleRenorm(CustomTestCase):
    VOCAB = 4096
    ROWS = 16

    def setUp(self):
        torch.manual_seed(0)

    def _rand_probs(self, rows=None):
        rows = rows or self.ROWS
        return torch.softmax(torch.randn(rows, self.VOCAB, device="cuda"), dim=-1)

    def _unrestricted_top_ks(self, rows=None):
        rows = rows or self.ROWS
        return torch.full((rows,), self.VOCAB, dtype=torch.int32, device="cuda")

    def test_top_p_matches_nucleus(self):
        from sglang.srt.speculative.eagle_utils import _renorm_top_k_top_p_hip

        probs = self._rand_probs()
        top_ks = self._unrestricted_top_ks()
        for top_p in (0.5, 0.9, 0.95):
            top_ps = torch.full((self.ROWS,), top_p, device="cuda")
            got = _renorm_top_k_top_p_hip(probs.clone(), top_ks, top_ps)
            ref = _torch_nucleus_reference(probs.clone(), top_ks, top_ps)
            self.assertLess((got - ref).abs().max().item(), 1e-6, f"top_p={top_p}")

    def test_top_p_one_is_exact_noop(self):
        from sglang.srt.speculative.eagle_utils import _renorm_top_k_top_p_hip

        probs = self._rand_probs()
        top_ks = self._unrestricted_top_ks()
        top_ps = torch.ones(self.ROWS, device="cuda")
        got = _renorm_top_k_top_p_hip(probs.clone(), top_ks, top_ps)
        self.assertEqual((got - probs).abs().max().item(), 0.0)

    def test_top_k_restriction(self):
        from sglang.srt.speculative.eagle_utils import _renorm_top_k_top_p_hip

        probs = self._rand_probs()
        top_ks = torch.full((self.ROWS,), 8, dtype=torch.int32, device="cuda")
        top_ps = torch.ones(self.ROWS, device="cuda")  # top_p no-op -> isolate top-k
        got = _renorm_top_k_top_p_hip(probs.clone(), top_ks, top_ps)
        self.assertTrue(((got > 0).sum(-1) == 8).all().item())
        ref = _torch_nucleus_reference(probs.clone(), top_ks, top_ps)
        self.assertLess((got - ref).abs().max().item(), 1e-6)

    def test_edge_top_p_to_zero_keeps_top1(self):
        from sglang.srt.speculative.eagle_utils import _renorm_top_k_top_p_hip

        probs = self._rand_probs(rows=1)
        top_ks = self._unrestricted_top_ks(rows=1)
        top_ps = torch.full((1,), 1e-4, device="cuda")
        got = _renorm_top_k_top_p_hip(probs.clone(), top_ks, top_ps)
        self.assertEqual((got > 0).sum().item(), 1)
        self.assertEqual(got.argmax().item(), probs.argmax().item())

    def test_edge_single_nonmasked_token(self):
        from sglang.srt.speculative.eagle_utils import _renorm_top_k_top_p_hip

        one_hot = torch.zeros(1, self.VOCAB, device="cuda")
        one_hot[0, 123] = 1.0
        got = _renorm_top_k_top_p_hip(
            one_hot.clone(),
            self._unrestricted_top_ks(rows=1),
            torch.full((1,), 0.9, device="cuda"),
        )
        self.assertEqual((got > 0).sum().item(), 1)
        self.assertAlmostEqual(got[0, 123].item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
