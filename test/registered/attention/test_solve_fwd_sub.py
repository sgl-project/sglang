import unittest

import torch

from sglang.srt.layers.attention.linear.gdn_backend import _solve_fwd_sub, _SolveFwdSub
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

    def test_forward_correlation(self):
        # Forward is ~1 fp32 ULP off the old loop (different reduction order), so check
        # it tracks the exact inverse: pearson corr == 1 and max abs err at fp32 noise.
        C = 64
        eye = torch.eye(C, device="cuda")
        for seed in range(5):
            A = self._rand_strict_lower((1, 48, 4), seed)
            T = (_solve_fwd_sub(A.clone()) + eye).flatten().double()
            ref = torch.linalg.inv(eye - A).flatten().double()
            corr = torch.corrcoef(torch.stack([T, ref]))[0, 1].item()
            self.assertGreater(corr, 1 - 1e-9)

    def test_gradcheck(self):
        # Formal finite-difference gradcheck of the analytic VJP (fp64 exact-inverse
        # forward sharing _SolveFwdSub's backward). Confirms grad_A = tril(T^T g T^T, -1).
        class _RefSolve(torch.autograd.Function):
            @staticmethod
            def forward(ctx, attn):
                C = attn.shape[-1]
                eye = torch.eye(C, dtype=attn.dtype, device=attn.device)
                T = torch.linalg.inv(eye - attn.tril(-1))
                ctx.save_for_backward(T)
                return T - eye

            @staticmethod
            def backward(ctx, grad_sub):
                return _SolveFwdSub.backward(ctx, grad_sub)

        torch.manual_seed(0)
        A = (torch.randn(2, 8, 8, device="cuda", dtype=torch.float64) * 0.1).tril(-1)
        A.requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(_RefSolve.apply, (A,), atol=1e-6, rtol=1e-4))

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
