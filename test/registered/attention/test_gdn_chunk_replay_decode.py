import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    _gdn_fused_replay,
    torch_chunk_gated_delta_rule,
    torch_gdn_gating,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-large")

C = 64


class _FakeLayer:
    """Minimal stand-in for RadixLinearAttention holding the dims/params the replay uses."""

    def __init__(self, HV, HK, K, V, A_log, dt_bias):
        self.layer_id = 0
        self.num_k_heads = HK
        self.num_v_heads = HV
        self.head_k_dim = K
        self.head_v_dim = V
        self.head_q_dim = K
        self.num_q_heads = HK
        self.q_dim = HK * K
        self.k_dim = HK * K
        self.v_dim = HV * V
        self.A_log = A_log
        self.dt_bias = dt_bias


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNChunkReplayDecode(unittest.TestCase):
    """Fused single-launch chunk-replay decode (batch-invariant GDN path).

    Each decode token must reproduce, bit-for-bit, row `position` of a single full-sequence
    `torch_chunk_gated_delta_rule` forward (what Megatron computes when teacher-forcing the whole
    prompt+response). This depends on the fp32 bmm routing to the M-invariant Triton kernel (no
    cuBLAS short-circuit); the fused kernel does only tl.dot matmuls + libdevice.exp + the shared
    forward-sub solve, with the divergent elementwise reductions precomputed in torch. Decode is
    inference-only so the fused path carries no autograd.
    """

    def _make_backend(self):
        be = GDNAttnBackend.__new__(GDNAttnBackend)
        be.gdn_replay_cache = {}
        return be

    def _full_ref(self, mixed_all, a_all, b_all, layer):
        T = mixed_all.shape[0]
        qf, kf, vf = torch.split(
            mixed_all, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q = qf.view(1, T, layer.num_k_heads, layer.head_k_dim)
        k = kf.view(1, T, layer.num_k_heads, layer.head_k_dim)
        v = vf.view(1, T, layer.num_v_heads, layer.head_v_dim)
        g, beta = torch_gdn_gating(layer.A_log, a_all, b_all, layer.dt_bias)
        core, _, _ = torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            ssm_states=None,
            cache_indices=torch.zeros(1, dtype=torch.long, device="cuda"),
            query_start_loc=torch.tensor([0, T], dtype=torch.int32, device="cuda"),
        )
        return core[0]  # [T, HV, V]

    def test_bit_identical_to_full_sequence(self):
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 12, 4, 128, 128  # GQA repeat 3, per-rank TP=4 shapes
            A_log = torch.randn(HV, device="cuda")
            dt_bias = torch.randn(HV, device="cuda")
            layer = _FakeLayer(HV, HK, K, V, A_log, dt_bias)
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            # prompt_len % 64 both == 0 and != 0; single crossing and multi-crossing.
            for pl, nd in [(64, 8), (60, 8), (30, 20), (128, 10), (63, 4), (1, 70), (127, 4)]:
                for seed in range(2):
                    torch.manual_seed(seed * 977 + pl)
                    T = pl + nd
                    mixed = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16)
                    a = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
                    ref = self._full_ref(mixed, a, b, layer)

                    be = self._make_backend()
                    cidx = torch.tensor([7], dtype=torch.long, device="cuda")
                    be._seed_gdn_replay_cache(
                        layer,
                        mixed[:pl],
                        a[:pl],
                        b[:pl],
                        torch.tensor([0, pl], dtype=torch.int32, device="cuda"),
                        cidx,
                        None,
                    )
                    for p in range(pl, T):
                        out = be._gdn_chunk_replay_decode(
                            layer, mixed[p : p + 1], a[p : p + 1], b[p : p + 1], cidx
                        )
                        self.assertTrue(
                            torch.equal(out[0, 0].to(torch.bfloat16), ref[p]),
                            msg=f"pl={pl} nd={nd} seed={seed} token={p} not bit-identical",
                        )

    def test_fused_replay_matches_torch_chunk_seeded(self):
        """Direct fused-kernel vs torch_chunk replay over a single partial chunk seeded by a
        nonzero boundary state S (the misaligned-prompt path decode continues from).

        Asserts FP32 bit-identity (not just bf16). torch_chunk casts its output to bf16, so an
        fp32 compare needs torch_chunk's pre-cast fp32 value; we recompute it via the same
        _gdn_replay_prep + _solve_fwd_sub the kernel uses (both proven == torch_chunk internals).
        An fp32 gap here is what previously tipped a bf16 boundary on ~1% of real decode tokens
        while bf16-random tests stayed green, so the fp32 assertion is the real regression guard.
        Runs real Qwen3.6 GDN dims (HV=48, HK=16) plus the small GQA shape.
        """
        with set_batch_invariant_mode(True):
            for HV, HK, K, V in [(48, 16, 128, 128), (12, 4, 128, 128)]:
                torch.manual_seed(HV)
                A_log = torch.randn(HV, device="cuda")
                dt_bias = torch.randn(HV, device="cuda")
                for w in [1, 5, 30, 32, 63, 64]:
                    torch.manual_seed(w + 100)
                    qh = torch.randn(w, HK, K, device="cuda", dtype=torch.bfloat16)
                    kh = torch.randn(w, HK, K, device="cuda", dtype=torch.bfloat16)
                    vh = torch.randn(w, HV, V, device="cuda", dtype=torch.bfloat16)
                    a = torch.randn(w, HV, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn(w, HV, device="cuda", dtype=torch.bfloat16)
                    S = torch.randn(1, HV, K, V, device="cuda", dtype=torch.float32) * 0.2
                    commit = w == C

                    out, Snew = _gdn_fused_replay(qh, kh, vh, a, b, A_log, dt_bias, S, commit)

                    # bf16 vs torch_chunk public output
                    g, beta = torch_gdn_gating(A_log, a, b, dt_bias)
                    ref, last, _ = torch_chunk_gated_delta_rule(
                        qh.view(1, w, HK, K),
                        kh.view(1, w, HK, K),
                        vh.view(1, w, HV, V),
                        g=g,
                        beta=beta,
                        ssm_states=S,
                        cache_indices=torch.zeros(1, dtype=torch.long, device="cuda"),
                        query_start_loc=torch.tensor([0, w], dtype=torch.int32, device="cuda"),
                    )
                    self.assertTrue(
                        torch.equal(out[0].to(torch.bfloat16), ref[0, -1]),
                        msg=f"HV={HV} w={w} fused replay not bf16-identical to torch_chunk",
                    )

                    # fp32 vs torch_chunk's pre-bf16-cast core (the tight guard)
                    core_fp32, snew_fp32 = self._torch_chunk_fp32(
                        qh, kh, vh, a, b, A_log, dt_bias, S
                    )
                    self.assertTrue(
                        torch.equal(out[0], core_fp32[:, w - 1]),
                        msg=f"HV={HV} w={w} fused replay not FP32-identical to torch_chunk core",
                    )
                    if commit:
                        self.assertTrue(
                            torch.equal(Snew[0], snew_fp32),
                            msg=f"HV={HV} w={w} fused replay boundary Snew not FP32-identical",
                        )

    def _torch_chunk_fp32(self, qh, kh, vh, a, b, A_log, dt_bias, S):
        """torch_chunk single-chunk math in fp32 WITHOUT the final bf16 cast, returning
        (core [HV,64,V], Snew [HV,K,V]). Uses the same _gdn_replay_prep + _solve_fwd_sub the
        kernel does; both are proven == torch_chunk's internal q/k/kb/vb/gcum and solve."""
        import torch.nn.functional as F

        from sglang.srt.layers.attention.linear.gdn_backend import (
            _gdn_replay_prep,
            _solve_fwd_sub,
        )

        qn, kn, kb, vb, gcum = _gdn_replay_prep(qh, kh, vh, a, b, A_log, dt_bias)
        w, HV, Dk = qn.shape
        dev = qn.device

        def pad(x):  # [w,HV,D] -> [HV,64,D]
            return F.pad(x.transpose(0, 1).contiguous(), (0, 0, 0, C - w)).contiguous()

        qn, kn, kb, vb = pad(qn), pad(kn), pad(kb), pad(vb)
        gc = gcum.transpose(0, 1).contiguous()  # [HV,w]
        # torch_chunk pads g pre-cumsum, so padded positions flatten at the boundary value.
        gc = torch.cat([gc, gc[:, -1:].expand(-1, C - w)], dim=1) if w < C else gc
        decay = ((gc.unsqueeze(-1) - gc.unsqueeze(-2)).tril().exp().float()).tril()
        tri0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), 0)
        A = -((kb @ kn.transpose(-1, -2)) * decay).masked_fill(tri0, 0)
        T = _solve_fwd_sub(A) + torch.eye(C, device=dev)
        value = T @ vb
        kcd = T @ (kb * gc.exp().unsqueeze(-1))
        Sh = S[0]
        vnew = value - kcd @ Sh
        tri1 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), 1)
        attn2 = ((qn @ kn.transpose(-1, -2)) * decay).masked_fill(tri1, 0)
        core = (qn * gc.exp().unsqueeze(-1)) @ Sh + attn2 @ vnew
        glast = gc[:, -1]
        kdec = kn * (glast[:, None] - gc).exp().unsqueeze(-1)
        snew = Sh * glast[:, None, None].exp() + kdec.transpose(-1, -2) @ vnew
        return core, snew

    def test_pad_slot_returns_zeros(self):
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 12, 4, 128, 128
            layer = _FakeLayer(
                HV, HK, K, V, torch.randn(HV, device="cuda"), torch.randn(HV, device="cuda")
            )
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            be = self._make_backend()
            out = be._gdn_chunk_replay_decode(
                layer,
                torch.randn(1, dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, HV, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, HV, device="cuda", dtype=torch.bfloat16),
                torch.tensor([PAD_SLOT_ID], dtype=torch.long, device="cuda"),
            )
            self.assertTrue(bool((out == 0).all()))
            self.assertEqual(len(be.gdn_replay_cache), 0)


if __name__ == "__main__":
    unittest.main()
