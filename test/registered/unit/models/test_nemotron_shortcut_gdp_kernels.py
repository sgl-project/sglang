"""GDP (Gated DeltaProduct) on FlashInfer's kernels vs a PyTorch recurrence of
the reference mixer (FP32 state, V-major [H, V, K])."""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra-b", runner_config="4-gpu-b200")

_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (
    9,
    10,
)

H, K, V, N = 8, 128, 64, 3  # heads, key dim, value dim, householder steps
POOL = 10
TOL = 2e-2


def _l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _reference(q, k, v, alpha, beta, state):
    """q [T,H,K], k [T,N,H,K], v [T,N,H,V], alpha [T,H], beta [T,N,H]."""
    s = state.clone().float()
    q = _l2norm(q.float()) * K**-0.5
    k, v = _l2norm(k.float()), v.float()
    outs = []
    for t in range(q.shape[0]):
        s = s * alpha[t].float()[:, None, None]
        for j in range(N):
            u = torch.einsum("hvk,hk->hv", s, k[t, j])
            delta = beta[t, j].float()[:, None] * (v[t, j] - u)
            s = s + delta[:, :, None] * k[t, j][:, None, :]
        outs.append(torch.einsum("hvk,hk->hv", s, q[t]))
    return torch.stack(outs), s


def _rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def _inputs(tokens):
    dev, dt = "cuda", torch.bfloat16
    return (
        torch.randn(tokens, H, K, device=dev, dtype=dt),
        torch.randn(tokens, N, H, K, device=dev, dtype=dt),
        torch.randn(tokens, N, H, V, device=dev, dtype=dt),
        torch.randn(tokens, H, device=dev, dtype=dt),
        torch.randn(tokens, N * H, device=dev, dtype=dt),
    )


@unittest.skipUnless(_SUPPORTED, "FlashInfer GDP kernels need SM90 or SM100")
class TestGDPKernels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.linear import gdp_backend

        cls.gdp = gdp_backend
        torch.manual_seed(0)
        a_log = torch.empty(H, device="cuda").uniform_(0.0, 16.0).clamp_min(1e-3).log()
        dt_bias = torch.randn(H, device="cuda") * 0.5 - 3.0
        cls.layer = types.SimpleNamespace(A_log=a_log, dt_bias=dt_bias)
        cls.slots = torch.tensor([3, 1, 7, 5], device="cuda", dtype=torch.int32)
        cls.pool = torch.randn(POOL, H, V, K, device="cuda") * 0.05
        cls.pool[1] = 0  # a fresh sequence

    def _gates(self, a, b):
        return self.gdp.gdp_gates(
            A_log=self.layer.A_log,
            dt_bias=self.layer.dt_bias,
            a=a,
            b=b,
            num_householder=N,
        )

    def test_prefill_matches_reference_and_leaves_other_slots(self):
        seq_lens = [1, 7, 64, 300]
        total = sum(seq_lens)
        q, k, v, a, b = _inputs(total)
        alpha, beta = self._gates(a, b)
        cu = [0]
        for length in seq_lens:
            cu.append(cu[-1] + length)
        pool = self.pool.clone()
        backend = self.gdp._gdp_prefill_backend(torch.device("cuda"))
        out = self.gdp.gdp_prefill_flashinfer(
            query=q,
            key=k,
            value=v,
            alpha=alpha,
            beta=beta,
            ssm_states=pool,
            cache_indices=self.slots,
            query_start_loc=torch.tensor(cu, device="cuda", dtype=torch.int32),
            backend=backend,
        )
        for i, slot in enumerate(self.slots.tolist()):
            s, e = cu[i], cu[i + 1]
            ref_out, ref_state = _reference(
                q[s:e],
                k[s:e],
                v[s:e],
                alpha[s:e],
                beta.view(total, N, H)[s:e],
                self.pool[slot],
            )
            self.assertLess(_rel(out[s:e], ref_out), TOL)
            self.assertLess(_rel(pool[slot], ref_state), TOL)
        untouched = [i for i in range(POOL) if i not in self.slots.tolist()]
        self.assertTrue(torch.equal(pool[untouched], self.pool[untouched]))

    def test_decode_matches_reference_and_skips_padding_rows(self):
        rows = self.slots.numel()
        q, k, v, a, b = _inputs(rows)
        pad = lambda x: torch.cat([x, x[:1]])  # noqa: E731
        slots = torch.cat(
            [self.slots, torch.tensor([-1], device="cuda", dtype=torch.int32)]
        )
        pool = self.pool.clone()
        out = self.gdp.gdp_decode_flashinfer(
            layer=self.layer,
            query=pad(q),
            key=pad(k),
            value=pad(v),
            a=pad(a),
            b=pad(b),
            ssm_states=pool,
            cache_indices=slots,
        )
        alpha, beta = self._gates(a, b)
        for i, slot in enumerate(self.slots.tolist()):
            ref_out, ref_state = _reference(
                q[i : i + 1],
                k[i : i + 1],
                v[i : i + 1],
                alpha[i : i + 1],
                beta.view(rows, N, H)[i : i + 1],
                self.pool[slot],
            )
            self.assertLess(_rel(out[i], ref_out[0]), TOL)
            self.assertLess(_rel(pool[slot], ref_state), TOL)
        self.assertEqual(out[rows].abs().max().item(), 0.0)

    def test_verify_writes_every_draft_state_to_scratch_only(self):
        rows, steps = self.slots.numel(), 3
        q, k, v, a, b = _inputs(rows * steps)
        scratch = torch.zeros(POOL + 1, steps, H, V, K, device="cuda")
        scratch_rows = torch.tensor([2, 0, 6, 4], device="cuda", dtype=torch.int32)
        pool = self.pool.clone()
        out = self.gdp.gdp_verify_flashinfer(
            layer=self.layer,
            query=q,
            key=k,
            value=v,
            a=a,
            b=b,
            ssm_states=pool,
            cache_indices=self.slots,
            scratch=scratch,
            scratch_rows=scratch_rows,
            num_draft_tokens=steps,
        )
        alpha, beta = self._gates(a, b)
        beta = beta.view(rows * steps, N, H)
        for i, slot in enumerate(self.slots.tolist()):
            state = self.pool[slot]
            for t in range(steps):
                j = i * steps + t
                ref_out, state = _reference(
                    q[j : j + 1],
                    k[j : j + 1],
                    v[j : j + 1],
                    alpha[j : j + 1],
                    beta[j : j + 1],
                    state,
                )
                self.assertLess(_rel(out[j], ref_out[0]), TOL)
                self.assertLess(_rel(scratch[scratch_rows[i], t], state), TOL)
        # The accepted prefix is committed after verify, not by the kernel.
        self.assertTrue(torch.equal(pool, self.pool))


if __name__ == "__main__":
    unittest.main()
