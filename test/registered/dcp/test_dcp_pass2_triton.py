# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the DCP verify-cascade Triton pass-2 kernel.

``_forward_verify_dcp`` pass-2 folds each request's T (<= 8) draft-token
latents with CAUSAL chain masking. The old path called the full tokenspeed
decode kernel over a per-layer-assembled 64-slot draft page pool (~50 us of
pure launch/tiling floor for ~0 work). The new path
(``SGLANG_DCP_TRITON_PASS2=1``, default) runs
``kernels.dcp_pass2_causal_attn_triton`` straight off q/k/k_rope.

Tolerance policy (different kernels are NOT bit-comparable; both are asserted
against an fp32 reference computed in this file, never against each other):
  - Triton vs fp32 ref: max-abs 3e-2 on out / 5e-3 on LSE. The kernel computes
    entirely in fp32 on the same (already-quantized) inputs the reference
    reads, so the only divergence is the bf16 output store (rel step 2^-8),
    fp32 accumulation order, and exp2/log2 rounding.
  - tokenspeed vs fp32 ref (SM100-only cross-check): max-abs 1.5e-1 on out /
    1e-2 on LSE. The tokenspeed fp8 kernel quantizes the softmax matrix P to
    fp8-e4m3 before its PV MMA (mla_decode_fp8.py "before quantize/P-store"),
    a ~2^-4 relative step on each weight; its LSE is computed from the fp32
    row_sum BEFORE that quantization, hence the much tighter LSE bound.

LSE convention under test: base-2 of the NATURAL-scaled logits,
``lse = log2(sum_j exp(softmax_scale * s_j))`` — exactly what
``dcp_lse_combine_triton(is_lse_base_on_e=False)`` consumes and what the
tokenspeed epilogue emits (``log2(row_sum) + softmax_scale*log2(e)*row_max``).
The cascade test closes the loop through the real production combine kernel.

Usage:
    python -m pytest test_dcp_pass2_triton.py -v
"""

import importlib.util
import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")

LOG2_E = math.log2(math.e)
D_LATENT, D_ROPE = 512, 64
D_QK = D_LATENT + D_ROPE


def _tokenspeed_supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if importlib.util.find_spec("tokenspeed_mla") is None:
        return False, "tokenspeed_mla python package is not installed"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return False, f"tokenspeed_mla requires SM100+, got SM {major}.{minor}"
    return True, ""


_TS_SUPPORTED, _TS_SKIP_REASON = _tokenspeed_supported()


def _pass2_ref(q, k_latent, k_rope, seq_lens, softmax_scale, output_scale):
    """fp32 reference for causal draft-chain attention.

    q [bs, T, H, D_qk]; k_latent [bs*T, D_latent] (doubles as V);
    k_rope [bs*T, D_rope]; seq_lens [bs] int32.
    Row (b, qt, h) attends KV j iff j <= qt and j < seq_lens[b].
    Returns (out fp32 [bs, T, H, D_latent], lse fp32 base-2 [bs, T, H]).
    """
    bs, T, H, _ = q.shape
    qf = q.float().cpu()
    kl = k_latent.float().cpu().view(bs, T, D_LATENT)
    kr = k_rope.float().cpu().view(bs, T, -1)
    kf = torch.cat([kl, kr], dim=-1)  # [bs, T, D_qk]
    s = torch.einsum("bthd,bjd->bthj", qf, kf) * softmax_scale  # [bs,T,H,T]
    qt_idx = torch.arange(T).view(1, T, 1, 1)
    j_idx = torch.arange(T).view(1, 1, 1, T)
    allowed = (j_idx <= qt_idx) & (j_idx < seq_lens.cpu().view(bs, 1, 1, 1))
    s = s.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("bthj,bjd->bthd", p, kl) * output_scale
    lse = torch.logsumexp(s, dim=-1) * LOG2_E  # base-2 of natural logits
    return out, lse


def _rand_inputs(bs, T, H, dtype, device, seed=0, d_qk=D_QK):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(bs, T, H, d_qk, generator=g).to(dtype).to(device)
    k_latent = torch.randn(bs * T, D_LATENT, generator=g).to(dtype).to(device)
    k_rope = torch.randn(bs * T, d_qk - D_LATENT, generator=g).to(dtype).to(device)
    seq = torch.full((bs,), T, dtype=torch.int32, device=device)
    return q, k_latent, k_rope, seq


def _max_abs(a, b):
    return (a.float().cpu() - b.float().cpu()).abs().max().item()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestPass2TritonVsRef(unittest.TestCase):
    """Triton pass-2 vs fp32 reference (tolerances justified in module doc)."""

    OUT_TOL = 3e-2
    LSE_TOL = 5e-3

    # (bs, T, H, dtype) — includes the production Kimi-K2 DCP shape
    # (T=ndt=8 draft tokens, H_local=8 heads, fp8 q/k).
    CASES = [
        (2, 8, 8, torch.float8_e4m3fn),  # production shape
        (4, 8, 8, torch.bfloat16),
        (3, 4, 8, torch.float8_e4m3fn),
        (1, 1, 8, torch.bfloat16),  # single token, causal degenerate
        (5, 8, 16, torch.bfloat16),
    ]

    def _check(self, q, k_latent, k_rope, seq, scale, out_scale, tag):
        from sglang.srt.layers.dcp import dcp_pass2_causal_attn_triton

        out, lse = dcp_pass2_causal_attn_triton(
            q, k_latent, k_rope, seq, softmax_scale=scale, output_scale=out_scale
        )
        # Contract of the tokenspeed call this replaces.
        exp_out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        self.assertEqual(out.dtype, exp_out_dtype, tag)
        self.assertEqual(lse.dtype, torch.float32, tag)
        self.assertEqual(tuple(out.shape), (*q.shape[:3], D_LATENT), tag)
        self.assertEqual(tuple(lse.shape), tuple(q.shape[:3]), tag)

        ref_out, ref_lse = _pass2_ref(q, k_latent, k_rope, seq, scale, out_scale)
        self.assertLess(_max_abs(out, ref_out), self.OUT_TOL, f"{tag} out")
        self.assertLess(_max_abs(lse, ref_lse), self.LSE_TOL, f"{tag} lse")

    def test_kernel_matches_fp32_reference(self):
        for i, (bs, T, H, dtype) in enumerate(self.CASES):
            q, kl, kr, seq = _rand_inputs(bs, T, H, dtype, "cuda", seed=i)
            scale = 1.0 / math.sqrt(D_QK)
            self._check(q, kl, kr, seq, scale, 1.0, f"case{i} {dtype}")

    def test_nontrivial_scales(self):
        """softmax_scale carries k_scale and output_scale = k_scale on the
        fp8-KV path — exercise both plumbing points."""
        q, kl, kr, seq = _rand_inputs(3, 8, 8, torch.float8_e4m3fn, "cuda", seed=7)
        k_scale = 1.3
        scale = (1.0 / math.sqrt(D_QK)) * k_scale
        self._check(q, kl, kr, seq, scale, k_scale, "scaled")

    def test_short_seq_lens_mask(self):
        """seq_lens < T rows: KV j >= seq_lens[b] must be masked out (runtime
        tensor mask — production always has seq_lens == T, but the capture-safe
        masking path must be correct)."""
        bs, T, H = 4, 8, 8
        q, kl, kr, _ = _rand_inputs(bs, T, H, torch.bfloat16, "cuda", seed=11)
        seq = torch.tensor([8, 3, 1, 5], dtype=torch.int32, device="cuda")
        scale = 1.0 / math.sqrt(D_QK)
        # Restrict the comparison to rows with at least one valid position
        # (qt < seq_lens[b]); rows beyond the len are normalized over the
        # truncated chain by both kernel and reference, so compare everything.
        self._check(q, kl, kr, seq, scale, 1.0, "short lens")

    def test_strided_views(self):
        """Production passes q as a HEAD-SLICE view of the full-head gathered
        query (no .contiguous()); k/k_rope may be row-strided. The kernel takes
        raw strides, so all must match the contiguous result bit-for-bit."""
        from sglang.srt.layers.dcp import dcp_pass2_causal_attn_triton

        bs, T, H_local, W = 2, 8, 8, 8
        H_full = H_local * W
        scale = 1.0 / math.sqrt(D_QK)
        g = torch.Generator(device="cpu").manual_seed(13)
        q_full = (
            torch.randn(bs, T, H_full, D_QK, generator=g).to(torch.float8_e4m3fn).cuda()
        )
        rank = 3
        q_view = q_full[:, :, rank * H_local : (rank + 1) * H_local, :]
        # k tensors: column slices of wider buffers (row stride > D).
        kl_wide = (
            torch.randn(bs * T, D_LATENT + 64, generator=g)
            .to(torch.float8_e4m3fn)
            .cuda()
        )
        kr_wide = (
            torch.randn(bs * T, D_ROPE + 32, generator=g).to(torch.float8_e4m3fn).cuda()
        )
        kl_view, kr_view = kl_wide[:, 32 : 32 + D_LATENT], kr_wide[:, 16 : 16 + D_ROPE]
        seq = torch.full((bs,), T, dtype=torch.int32, device="cuda")

        out_v, lse_v = dcp_pass2_causal_attn_triton(
            q_view, kl_view, kr_view, seq, softmax_scale=scale
        )
        out_c, lse_c = dcp_pass2_causal_attn_triton(
            q_view.contiguous(),
            kl_view.contiguous(),
            kr_view.contiguous(),
            seq,
            softmax_scale=scale,
        )
        self.assertTrue(torch.equal(out_v, out_c), "strided out != contiguous out")
        self.assertTrue(torch.equal(lse_v, lse_c), "strided lse != contiguous lse")
        ref_out, ref_lse = _pass2_ref(q_view, kl_view, kr_view, seq, scale, 1.0)
        self.assertLess(_max_abs(out_v, ref_out), TestPass2TritonVsRef.OUT_TOL)
        self.assertLess(_max_abs(lse_v, ref_lse), TestPass2TritonVsRef.LSE_TOL)

    def test_causality(self):
        """Perturbing draft KV at position j must not change any out/lse row
        with qt < j (bit-identical — the masked loads never touch those
        addresses)."""
        from sglang.srt.layers.dcp import dcp_pass2_causal_attn_triton

        bs, T, H = 2, 8, 8
        q, kl, kr, seq = _rand_inputs(bs, T, H, torch.bfloat16, "cuda", seed=17)
        scale = 1.0 / math.sqrt(D_QK)
        out0, lse0 = dcp_pass2_causal_attn_triton(q, kl, kr, seq, softmax_scale=scale)
        j = 5
        kl2, kr2 = kl.clone(), kr.clone()
        for b in range(bs):
            kl2[b * T + j] = 100.0
            kr2[b * T + j] = -100.0
        out1, lse1 = dcp_pass2_causal_attn_triton(q, kl2, kr2, seq, softmax_scale=scale)
        self.assertTrue(torch.equal(out0[:, :j], out1[:, :j]), "causality leak (out)")
        self.assertTrue(torch.equal(lse0[:, :j], lse1[:, :j]), "causality leak (lse)")
        # And rows qt >= j MUST change (the perturbation is visible to them).
        self.assertFalse(torch.equal(out0[:, j:], out1[:, j:]), "mask too strong")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestPass2CascadeCombine(unittest.TestCase):
    """End-to-end cascade identity through the PRODUCTION combine kernel:

        full-causal verify over [prefix ++ draft chain]
          == dcp_lse_combine_triton( non-causal prefix fold (fp32 ref, base-2
             LSE), triton pass-2, is_lse_base_on_e=False )

    This is the decisive check that the Triton kernel's LSE base/scaling is
    the one ``_forward_verify_dcp``'s combine actually consumes — a wrong base
    passes kernel-vs-ref tests but fails here. Tolerance 4e-2 max-abs: both
    partials are stored bf16 (as in production) before the fp32 combine.
    """

    def test_cascade_identity(self):
        from sglang.srt.layers.dcp import (
            dcp_lse_combine_triton,
            dcp_pass2_causal_attn_triton,
        )

        bs, T, H, P = 3, 8, 8, 40
        dev, dtype = "cuda", torch.bfloat16
        scale = 1.0 / math.sqrt(D_QK)
        g = torch.Generator(device="cpu").manual_seed(23)
        q = torch.randn(bs, T, H, D_QK, generator=g).to(dtype).to(dev)
        prefix = torch.randn(bs, P, D_QK, generator=g).to(dtype).to(dev)
        kl = torch.randn(bs * T, D_LATENT, generator=g).to(dtype).to(dev)
        kr = torch.randn(bs * T, D_ROPE, generator=g).to(dtype).to(dev)
        seq = torch.full((bs,), T, dtype=torch.int32, device=dev)

        # --- full-causal fp32 reference over prefix ++ chain ---
        draft = torch.cat(
            [kl.view(bs, T, D_LATENT), kr.view(bs, T, D_ROPE)], dim=-1
        )  # [bs, T, D_qk]
        kv_all = torch.cat([prefix, draft], dim=1).float().cpu()  # [bs, P+T, D_qk]
        qf = q.float().cpu()
        s = torch.einsum("bthd,bjd->bthj", qf, kv_all) * scale  # [bs,T,H,P+T]
        qt_idx = torch.arange(T).view(1, T, 1, 1)
        j_idx = torch.arange(P + T).view(1, 1, 1, P + T)
        allowed = j_idx < (P + qt_idx + 1)  # prefix fully visible, chain causal
        s = s.masked_fill(~allowed, float("-inf"))
        p = torch.softmax(s, dim=-1)
        ref = torch.einsum("bthj,bjd->bthd", p, kv_all[..., :D_LATENT])

        # --- pass-1: fp32 NON-causal prefix fold with base-2 LSE ---
        s1 = torch.einsum("bthd,bjd->bthj", qf, prefix.float().cpu()) * scale
        p1 = torch.softmax(s1, dim=-1)
        o1 = torch.einsum("bthj,bjd->bthd", p1, prefix.float().cpu()[..., :D_LATENT])
        lse1 = torch.logsumexp(s1, dim=-1) * LOG2_E

        # --- pass-2: the Triton kernel under test ---
        o2, lse2 = dcp_pass2_causal_attn_triton(q, kl, kr, seq, softmax_scale=scale)

        # --- production combine (base-2), partials stored bf16 as in prod ---
        N = bs * T
        recv_out = torch.stack(
            [
                o1.reshape(N, H, D_LATENT).to(torch.bfloat16).to(dev),
                o2.reshape(N, H, D_LATENT),
            ],
            dim=0,
        )
        recv_lse = torch.stack(
            [lse1.reshape(N, H).float().to(dev), lse2.reshape(N, H)], dim=0
        )
        final, _ = dcp_lse_combine_triton(recv_out, recv_lse, is_lse_base_on_e=False)

        diff = (final.float().cpu() - ref.reshape(N, H, D_LATENT)).abs().max().item()
        self.assertLess(diff, 4e-2, f"cascade combine identity, max abs {diff}")


@unittest.skipIf(not _TS_SUPPORTED, _TS_SKIP_REASON)
class TestPass2TritonVsTokenspeed(unittest.TestCase):
    """Triton pass-2 vs the tokenspeed decode-kernel call it replaces, both
    asserted against the fp32 reference (never against each other; see the
    module docstring for the tolerance derivation). Reproduces the OLD
    ``_forward_verify_dcp`` pass-2 verbatim: 64-slot draft page pool, identity
    block table, causal_mask=True, cp_world=1, return_lse=True.
    """

    TRITON_OUT_TOL = 3e-2
    TRITON_LSE_TOL = 5e-3
    TS_OUT_TOL = 1.5e-1  # fp8-quantized P in the PV MMA
    TS_LSE_TOL = 1e-2  # LSE computed pre-quantization (fp32 row_sum)

    def test_both_within_fp32_reference(self):
        import tokenspeed_mla

        from sglang.srt.layers.dcp import dcp_pass2_causal_attn_triton

        bs, T, H, PAGE = 4, 8, 8, 64
        dev, dtype = torch.device("cuda"), torch.float8_e4m3fn
        k_scale = 1.3
        scale = (1.0 / math.sqrt(D_QK)) * k_scale
        q, kl, kr, seq = _rand_inputs(bs, T, H, dtype, dev, seed=31)
        ref_out, ref_lse = _pass2_ref(q, kl, kr, seq, scale, k_scale)

        # --- OLD path: tokenspeed over the assembled draft page pool ---
        draft_latent = torch.cat(
            [kl.view(bs * T, D_LATENT), kr.view(bs * T, D_ROPE)], dim=-1
        )
        draft_pool = draft_latent.new_zeros(bs, PAGE, D_QK)
        draft_pool[:, :T, :] = draft_latent.view(bs, T, D_QK)
        draft_pool = draft_pool.unsqueeze(1)  # [bs, 1, page_size, D_qk]
        draft_bt = torch.arange(bs, dtype=torch.int32, device=dev).view(bs, 1)
        ws_bytes = tokenspeed_mla.get_num_sm(dev) * H * max(T, 8) * (D_LATENT + 1) * 4
        ws = torch.empty(ws_bytes, dtype=torch.int8, device=dev)
        ts_out, ts_lse = tokenspeed_mla.tokenspeed_mla_decode(
            query=q,
            kv_cache=draft_pool,
            workspace_buffer=ws,
            kv_lora_rank=D_LATENT,
            qk_rope_head_dim=D_ROPE,
            block_tables=draft_bt,
            seq_lens=seq,
            max_seq_len=T,
            softmax_scale=scale,
            output_scale=k_scale,
            enable_pdl=False,
            return_lse=True,
            cp_world=1,
            cp_rank=0,
            causal_seqs=None,
            causal_mask=True,
        )

        # --- NEW path: Triton kernel straight off q/k/k_rope ---
        tr_out, tr_lse = dcp_pass2_causal_attn_triton(
            q,
            kl.view(bs * T, D_LATENT),
            kr.view(bs * T, D_ROPE),
            seq,
            softmax_scale=scale,
            output_scale=k_scale,
        )

        # Contract parity: same shapes/dtypes as the tokenspeed call.
        self.assertEqual(tr_out.dtype, ts_out.dtype)
        self.assertEqual(tuple(tr_out.shape), tuple(ts_out.shape))
        self.assertEqual(tr_lse.dtype, ts_lse.dtype)
        self.assertEqual(tuple(tr_lse.shape), tuple(ts_lse.shape))

        d_tr_out = _max_abs(tr_out, ref_out)
        d_tr_lse = _max_abs(tr_lse, ref_lse)
        d_ts_out = _max_abs(ts_out, ref_out)
        d_ts_lse = _max_abs(ts_lse, ref_lse)
        self.assertLess(d_tr_out, self.TRITON_OUT_TOL, f"triton out {d_tr_out}")
        self.assertLess(d_tr_lse, self.TRITON_LSE_TOL, f"triton lse {d_tr_lse}")
        self.assertLess(d_ts_out, self.TS_OUT_TOL, f"tokenspeed out {d_ts_out}")
        self.assertLess(d_ts_lse, self.TS_LSE_TOL, f"tokenspeed lse {d_ts_lse}")


if __name__ == "__main__":
    unittest.main()
