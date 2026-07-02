import unittest

import torch

from sglang.srt.layers.attention.linear.gdn_backend import _solve_fwd_sub
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="stage-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestSolveForwardSubstitution(unittest.TestCase):
    """Triton forward-substitution triangular solve that replaces the 64-iteration
    Python loop in torch_chunk_gated_delta_rule.

    Not required to be bit-identical to the old loop (it uses a tree reduction, not
    the loop's ILP=4 order). It MUST be deterministic, independent of the number of
    chunks (so Megatron prefill and SGLang match each other), equal to (I - A)^-1 to
    fp32 tolerance, and differentiable.
    """

    def _rand_strict_lower(self, shape, seed, C=64, scale=0.15):
        torch.manual_seed(seed)
        return (torch.randn(*shape, C, C, device="cuda", dtype=torch.float32) * scale).tril(-1)

    def test_deterministic(self):
        for seed in range(4):
            A = self._rand_strict_lower((1, 48, 12), seed)
            self.assertTrue(torch.equal(_solve_fwd_sub(A.clone()), _solve_fwd_sub(A.clone())))

    def test_chunk_count_invariant(self):
        # Output for a given (b, h, chunk) must not depend on the number of chunks.
        A = self._rand_strict_lower((1, 48, 12), 0)
        self.assertTrue(
            torch.equal(_solve_fwd_sub(A[:, :, :3].clone()), _solve_fwd_sub(A.clone())[:, :, :3])
        )

    def test_matches_inverse(self):
        C = 64
        eye = torch.eye(C, device="cuda")
        for seed in range(6):
            A = self._rand_strict_lower((1, 48, 4), seed)
            T = _solve_fwd_sub(A.clone()) + eye
            ref = torch.linalg.inv(eye - A)
            self.assertLess((T - ref).abs().max().item(), 1e-5)

    def test_gradients(self):
        C = 64
        eye = torch.eye(C, device="cuda")
        for seed in range(4):
            base = self._rand_strict_lower((48,), seed, scale=0.1)
            a1 = base.clone().requires_grad_(True)
            a2 = base.clone().requires_grad_(True)
            T1 = _solve_fwd_sub(a1) + eye
            T2 = torch.linalg.inv(eye - a2.tril(-1))
            g = torch.randn_like(T1)
            T1.backward(g)
            T2.backward(g.clone())
            self.assertTrue(torch.allclose(a1.grad, a2.grad, atol=1e-5, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
