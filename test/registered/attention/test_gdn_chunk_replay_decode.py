import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
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

    def _make_backend(self, max_bs=8):
        # Bypass __init__ (needs a ModelRunner); set up just the arena state the replay path uses.
        be = GDNAttnBackend.__new__(GDNAttnBackend)
        be.device = torch.device("cuda")
        be.gdn_arena_max_bs = max_bs
        be.gdn_arena_rows = max_bs + 1
        be.gdn_pad_row = max_bs
        be.gdn_arena = {}
        be.gdn_slot_to_row = {}
        be.gdn_free_rows = list(range(max_bs))
        # Batch buffers span the largest dispatchable decode batch (>= arena rows). Use a value
        # bigger than max_bs so the split of "arena rows" vs "batch length" is actually exercised.
        be.gdn_max_batch = max_bs + 4
        be._gdn_row_map = torch.zeros(be.gdn_max_batch, dtype=torch.int32, device="cuda")
        be._gdn_row_map_host = torch.zeros(be.gdn_max_batch, dtype=torch.int32, device="cpu").pin_memory()
        be._gdn_valid = torch.zeros(be.gdn_max_batch, dtype=torch.float32, device="cuda")
        be._gdn_valid_host = torch.zeros(be.gdn_max_batch, dtype=torch.float32, device="cpu").pin_memory()
        be._gdn_out = None
        be._gdn_cur_row_map = None
        be._gdn_cur_valid = None
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
                        be._gdn_write_row_map(cidx.tolist())
                        out = be._gdn_chunk_replay_decode(
                            layer, mixed[p : p + 1], a[p : p + 1], b[p : p + 1], cidx
                        )
                        self.assertTrue(
                            torch.equal(out[0, 0].to(torch.bfloat16), ref[p]),
                            msg=f"pl={pl} nd={nd} seed={seed} token={p} not bit-identical",
                        )

    def test_batched_decode_bit_identical(self):
        """B>1 decode: several sequences with different prompt lengths (so their partial-chunk
        widths and fold phases differ) decoded together through one _gdn_chunk_replay_decode call
        per step. Each row must reproduce its own full-sequence torch_chunk row bit-for-bit. Guards
        the batched slot-arena path (the B=1 dumper never exercises cross-slot indexing)."""
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 48, 16, 128, 128  # real Qwen3.6 GDN dims
            A_log = torch.randn(HV, device="cuda")
            dt_bias = torch.randn(HV, device="cuda")
            layer = _FakeLayer(HV, HK, K, V, A_log, dt_bias)
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            # Distinct prompt lengths -> distinct widths/fold phases across the batch, incl a
            # PAD slot. nd chosen so at least one sequence crosses a 64-fold during decode.
            pls = [60, 30, 128, 1]
            slots = [7, 2, 11, PAD_SLOT_ID]
            nd = 10
            torch.manual_seed(1234)
            seqs = []
            for pl in pls:
                mixed = torch.randn(pl + nd, dim, device="cuda", dtype=torch.bfloat16)
                a = torch.randn(pl + nd, HV, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(pl + nd, HV, device="cuda", dtype=torch.bfloat16)
                ref = self._full_ref(mixed, a, b, layer)
                seqs.append((pl, mixed, a, b, ref))

            be = self._make_backend()
            cidx = torch.tensor(slots, dtype=torch.long, device="cuda")
            # Seed all non-pad sequences (packed, one query_start_loc covering the batch).
            valid = [(pl, m, a, b, r, s) for (pl, m, a, b, r), s in zip(seqs, slots) if s != PAD_SLOT_ID]
            qsl = [0]
            for pl, *_ in valid:
                qsl.append(qsl[-1] + pl)
            mixed_cat = torch.cat([m[:pl] for pl, m, *_ in valid], dim=0)
            a_cat = torch.cat([a[:pl] for pl, m, a, *_ in valid], dim=0)
            b_cat = torch.cat([b[:pl] for pl, m, a, b, *_ in valid], dim=0)
            be._seed_gdn_replay_cache(
                layer, mixed_cat, a_cat, b_cat,
                torch.tensor(qsl, dtype=torch.int32, device="cuda"),
                torch.tensor([s for *_, s in valid], dtype=torch.long, device="cuda"),
                None,
            )
            for t in range(nd):
                mixed_row = torch.stack([seqs[j][1][pls[j] + t] for j in range(len(pls))], dim=0)
                a_row = torch.stack([seqs[j][2][pls[j] + t] for j in range(len(pls))], dim=0)
                b_row = torch.stack([seqs[j][3][pls[j] + t] for j in range(len(pls))], dim=0)
                be._gdn_write_row_map(cidx.tolist())
                out = be._gdn_chunk_replay_decode(layer, mixed_row, a_row, b_row, cidx)
                for j in range(len(pls)):
                    if slots[j] == PAD_SLOT_ID:
                        self.assertTrue(bool((out[0, j] == 0).all()))
                        continue
                    self.assertTrue(
                        torch.equal(out[0, j].to(torch.bfloat16), seqs[j][4][pls[j] + t]),
                        msg=f"batched decode slot={slots[j]} pl={pls[j]} t={t} not bit-identical",
                    )

    def test_incremental_replay_matches_torch_chunk_seeded(self):
        """Incremental per-token replay vs torch_chunk over a single partial chunk seeded by a
        nonzero boundary state S (the misaligned-prompt path decode continues from). Advances the
        per-slot incremental state token by token and checks EVERY width's core row.

        Asserts FP32 bit-identity (not just bf16). torch_chunk casts its output to bf16, so an
        fp32 compare needs torch_chunk's pre-cast fp32 value; we recompute it via the same
        prep + _solve_fwd_sub the kernels use (both proven == torch_chunk internals).
        An fp32 gap here is what previously tipped a bf16 boundary on ~1% of real decode tokens
        while bf16-random tests stayed green, so the fp32 assertion is the real regression guard.
        Runs real Qwen3.6 GDN dims (HV=48, HK=16) plus the small GQA shape.
        """
        with set_batch_invariant_mode(True):
            for HV, HK, K, V in [(48, 16, 128, 128), (12, 4, 128, 128)]:
                torch.manual_seed(HV)
                A_log = torch.randn(HV, device="cuda")
                dt_bias = torch.randn(HV, device="cuda")
                layer = _FakeLayer(HV, HK, K, V, A_log, dt_bias)
                torch.manual_seed(HV + 100)
                qh = torch.randn(C, HK, K, device="cuda", dtype=torch.bfloat16)
                kh = torch.randn(C, HK, K, device="cuda", dtype=torch.bfloat16)
                vh = torch.randn(C, HV, V, device="cuda", dtype=torch.bfloat16)
                a = torch.randn(C, HV, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(C, HV, device="cuda", dtype=torch.bfloat16)
                S = torch.randn(1, HV, K, V, device="cuda", dtype=torch.float32) * 0.2

                be = self._make_backend()
                arena = be._gdn_arena_for(layer)
                out = be._gdn_out_for(layer)
                ar = 3  # arbitrary non-zero arena row to exercise row indexing
                arena["boundary"][ar].copy_(S[0])
                arena["i_buf"][ar].zero_()
                arena["inext_buf"][ar].zero_()
                row_map = torch.tensor([ar], dtype=torch.int32, device="cuda")
                for w in range(1, C + 1):
                    i = w - 1
                    be._gdn_launch_step(
                        arena, layer,
                        qh[i : i + 1].contiguous(), kh[i : i + 1].contiguous(),
                        vh[i : i + 1].contiguous(), a[i : i + 1].contiguous(),
                        b[i : i + 1].contiguous(), row_map, 1, out,
                    )
                    core = out[0].clone()
                    commit = w == C

                    # bf16 vs torch_chunk public output at this width
                    g, beta = torch_gdn_gating(A_log, a[:w], b[:w], dt_bias)
                    ref, last, _ = torch_chunk_gated_delta_rule(
                        qh[:w].view(1, w, HK, K),
                        kh[:w].view(1, w, HK, K),
                        vh[:w].view(1, w, HV, V),
                        g=g,
                        beta=beta,
                        ssm_states=S,
                        cache_indices=torch.zeros(1, dtype=torch.long, device="cuda"),
                        query_start_loc=torch.tensor([0, w], dtype=torch.int32, device="cuda"),
                    )
                    self.assertTrue(
                        torch.equal(core.to(torch.bfloat16), ref[0, -1]),
                        msg=f"HV={HV} w={w} incremental replay not bf16-identical to torch_chunk",
                    )

                    # fp32 vs torch_chunk's pre-bf16-cast core (the tight guard)
                    core_fp32, snew_fp32 = self._torch_chunk_fp32(
                        qh[:w], kh[:w], vh[:w], a[:w], b[:w], A_log, dt_bias, S
                    )
                    self.assertTrue(
                        torch.equal(core, core_fp32[:, w - 1]),
                        msg=f"HV={HV} w={w} incremental replay not FP32-identical to torch_chunk",
                    )
                    if commit:
                        # the commit folded Snew into the arena boundary row
                        self.assertTrue(
                            torch.equal(arena["boundary"][ar], snew_fp32),
                            msg=f"HV={HV} w={w} incremental boundary Snew not FP32-identical",
                        )

    def _prep_fp32(self, qh, kh, vh, a, b, A_log, dt_bias):
        """torch_chunk's per-token prep (shared l2norm, GQA repeat, scale, gating, per-chunk
        cumsum), returning qn,kn,kb [w,HV,Dk], vb [w,HV,Dv], gcum [w,HV] fp32."""
        from sglang.srt.layers.attention.linear.gdn_backend import chunk_cumsum, l2norm_bf16

        w, HK, Dk = qh.shape
        HV = vh.shape[1]
        rep = HV // HK
        scale = 1.0 / (Dk**0.5)
        qn = l2norm_bf16(qh).float().repeat_interleave(rep, dim=1) * scale
        kn = l2norm_bf16(kh).float().repeat_interleave(rep, dim=1)
        g, beta = torch_gdn_gating(A_log, a, b, dt_bias)
        g = g[0].contiguous()
        beta = beta[0].contiguous().float()
        kb = kn * beta[..., None]
        vb = vh.float() * beta[..., None]
        # Shared serial chunk_cumsum (what torch_chunk_gated_delta_rule now uses), not torch.cumsum:
        # the serial scan differs from torch.cumsum ~1 fp32 ULP and the decode append reproduces the
        # serial scan bit-for-bit, so the fp32 reference must scan the same way.
        gt = g.transpose(0, 1).contiguous()  # [HV,w]
        gcum = chunk_cumsum(gt.view(1, HV, 1, w))[0, :, 0].transpose(0, 1).contiguous()
        return qn.contiguous(), kn.contiguous(), kb.contiguous(), vb.contiguous(), gcum

    def _torch_chunk_fp32(self, qh, kh, vh, a, b, A_log, dt_bias, S):
        """torch_chunk single-chunk math in fp32 WITHOUT the final bf16 cast, returning
        (core [HV,64,V], Snew [HV,K,V]). Uses the same prep + _solve_fwd_sub the kernels do;
        both are proven == torch_chunk's internal q/k/kb/vb/gcum and solve."""
        import torch.nn.functional as F

        from sglang.srt.layers.attention.linear.gdn_backend import _solve_fwd_sub

        qn, kn, kb, vb, gcum = self._prep_fp32(qh, kh, vh, a, b, A_log, dt_bias)
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
            cidx = torch.tensor([PAD_SLOT_ID], dtype=torch.long, device="cuda")
            be._gdn_write_row_map(cidx.tolist())
            out = be._gdn_chunk_replay_decode(
                layer,
                torch.randn(1, dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, HV, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, HV, device="cuda", dtype=torch.bfloat16),
                cidx,
            )
            self.assertTrue(bool((out == 0).all()))
            # PAD slot claimed no arena row.
            self.assertEqual(len(be.gdn_slot_to_row), 0)


if __name__ == "__main__":
    unittest.main()
