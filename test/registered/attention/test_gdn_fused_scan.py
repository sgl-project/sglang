import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    torch_chunk_gated_delta_rule,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large")

DEV = "cuda"


def _make(seq_lens, HK, HV, Dk, Dv, seed, nonzero_state, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    T = sum(seq_lens)
    q = torch.randn(1, T, HK, Dk, device=DEV, dtype=dtype)
    k = torch.randn(1, T, HK, Dk, device=DEV, dtype=dtype)
    v = torch.randn(1, T, HV, Dv, device=DEV, dtype=dtype)
    a = torch.randn(1, T, HV, device=DEV, dtype=dtype)
    b = torch.randn(1, T, HV, device=DEV, dtype=dtype)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.to(torch.bfloat16).sigmoid()
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), dtype=torch.int32, device=DEV
    )
    n_seq = len(seq_lens)
    ssm = None
    if nonzero_state:
        ssm = torch.randn(n_seq, HV, Dk, Dv, device=DEV, dtype=torch.float32) * 0.2
    cidx = torch.arange(n_seq, dtype=torch.long, device=DEV)
    return q, k, v, g, beta, ssm, cidx, qsl


def _run(args, want_bnd, grad):
    q, k, v, g, beta, ssm, cidx, qsl = args
    torch.set_grad_enabled(grad)
    return torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        ssm_states=None if ssm is None else ssm.clone(),
        cache_indices=cidx, query_start_loc=qsl, return_boundary_state=want_bnd,
    )


def _md(a, b):
    return (a.float() - b.float()).abs().max().item()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNFusedScan(unittest.TestCase):
    """The cross-chunk recurrence has two entries into the SAME fused Triton scan (_chunk_scan_kernel):
    grad off (inference / log-prob dump) calls _fused_chunk_scan directly; grad on (training) wraps
    the whole WY-prep + scan in _ChunkGDR, an autograd Function whose backward is fla's analytic
    chunk_gated_delta_rule_bwd driven from OUR saved tensors (our T-matrix as fla's `A`, our fp32
    cumsum as fla's `g`). Both forwards run identical kernels, so sglang and Megatron are bit-identical
    in both modes. These tests cover: inference self-determinism, train-forward == inference-forward
    (same kernels), and that the fla-driven backward produces finite grads matching autograd through a
    pure-torch replica of OUR forward to bf16-kernel tolerance (rel ~1e-3).
    """

    def test_inference_self_deterministic(self):
        with set_batch_invariant_mode(True):
            for seq_lens in [[2048], [512] * 8, [2048] * 8, [300, 64, 1]]:
                args = _make(seq_lens, 4, 12, 128, 128, 3, True)
                c1, l1, _ = _run(args, False, False)
                c2, l2, _ = _run(args, False, False)
                self.assertEqual(_md(c1, c2), 0.0, msg=f"core lens={seq_lens}")
                self.assertEqual(_md(l1, l2), 0.0, msg=f"last lens={seq_lens}")

    def test_train_forward_matches_inference(self):
        # _ChunkGDR (grad on) and the raw fused scan (grad off) run the same kernels on the same
        # fp32 inputs, so their forwards are bf16-bit-identical for a single sequence. Use fp32
        # inputs so the two paths share the exact same l2norm rounding.
        with set_batch_invariant_mode(True):
            for HK, HV in [(4, 12), (16, 48), (2, 2)]:
                for seq_lens in [[2048], [2000], [64], [127], [4096]]:
                    for nz in (False, True):
                        args = _make(seq_lens, HK, HV, 128, 128, 7, nz, dtype=torch.float32)
                        ci, li, _ = _run(args, False, False)
                        ct, lt, _ = _run(args, False, True)
                        self.assertLess(_md(ci.to(torch.bfloat16), ct.to(torch.bfloat16)), 5e-6,
                                        msg=f"core HK={HK} lens={seq_lens} nz={nz}")
                        self.assertLess(_md(li, lt), 5e-6,
                                        msg=f"last HK={HK} lens={seq_lens} nz={nz}")

    def test_backward_matches_torch_reference(self):
        # Grads from _ChunkGDR (fla analytic bwd) vs autograd through a pure-torch replica of OUR
        # forward, on a single aligned sequence, uniform heads (no GQA), fp32. bf16-kernel level.
        from sglang.srt.layers.attention.linear.gdn_backend import _ChunkGDR
        C = 64

        def ref(q, k, v, g, beta, S0):
            N, HV, T, Dk = q.shape
            Dv = v.shape[-1]
            NC = T // C
            scale = 1.0 / Dk ** 0.5
            qs = (q * scale).reshape(N, HV, NC, C, Dk)
            kc = k.reshape(N, HV, NC, C, Dk)
            vc = v.reshape(N, HV, NC, C, Dv)
            bc = beta.reshape(N, HV, NC, C)
            vb = vc * bc.unsqueeze(-1)
            kb = kc * bc.unsqueeze(-1)
            gcum = g.reshape(N, HV, NC, C).cumsum(-1)
            eye = torch.eye(C, device=DEV)
            m0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=DEV), 0)
            decay = ((gcum.unsqueeze(-1) - gcum.unsqueeze(-2)).tril().exp()).tril()
            L = -((kb @ kc.transpose(-1, -2)) * decay).masked_fill(m0, 0)
            Tm = torch.inverse(eye - L)
            u = Tm @ vb
            w = Tm @ (kb * gcum.exp().unsqueeze(-1))
            S = S0
            m1 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=DEV), 1)
            outs = []
            for i in range(NC):
                qi, ki, ui, wi, gi = qs[:, :, i], kc[:, :, i], u[:, :, i], w[:, :, i], gcum[:, :, i]
                attn = (qi @ ki.transpose(-1, -2) * decay[:, :, i]).masked_fill(m1, 0)
                vn = ui - wi @ S
                outs.append((qi * gi[..., None].exp()) @ S + attn @ vn)
                S = S * gi[:, :, -1, None, None].exp() + (ki * (gi[:, :, -1, None] - gi).exp()[..., None]).transpose(-1, -2) @ vn
            return torch.stack(outs, dim=2).reshape(N, HV, T, Dv), S

        N, HV, Dk, Dv, T = 1, 6, 128, 128, 256
        torch.manual_seed(0)

        def mk():
            q = torch.nn.functional.normalize(torch.randn(N, HV, T, Dk, device=DEV), dim=-1)
            k = torch.nn.functional.normalize(torch.randn(N, HV, T, Dk, device=DEV), dim=-1)
            v = torch.randn(N, HV, T, Dv, device=DEV)
            g = -torch.nn.functional.softplus(torch.randn(N, HV, T, device=DEV))
            beta = torch.rand(N, HV, T, device=DEV).sigmoid()
            S0 = torch.randn(N, HV, Dk, Dv, device=DEV) * 0.1
            return q, k, v, g, beta, S0

        q0, k0, v0, g0, b0, s0 = mk()
        do = torch.randn(N, HV, T, Dv, device=DEV)
        dht = torch.randn(N, HV, Dk, Dv, device=DEV)
        with set_batch_invariant_mode(True):
            for use_dht in (False, True):
                gr, gc = [], []
                for fn in (ref, lambda *a: _ChunkGDR.apply(*a, C)):
                    ins = [t.clone().requires_grad_(True) for t in (q0, k0, v0, g0, b0, s0)]
                    core, S = fn(*ins)
                    loss = (core * do).sum() + ((S * dht).sum() if use_dht else 0.0)
                    loss.backward()
                    (gr if fn is ref else gc).append([t.grad.clone() for t in ins])
                for name, a, b in zip("q k v g beta S0".split(), gr[0], gc[0]):
                    rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-30)
                    self.assertLess(rel.item(), 5e-3, msg=f"d{name} dht={use_dht} rel={rel:.2e}")
                    self.assertTrue(torch.isfinite(b).all().item(), msg=f"d{name} non-finite")


if __name__ == "__main__":
    unittest.main()
