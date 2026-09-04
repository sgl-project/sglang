"""Real-device XPU tests for the encoder-decoder attention path.

The backend calls flash_attn_with_kvcache with a page_size=1 view; sgl-kernel-xpu
PR #454 detects that and gathers + runs varlen inside the kernel. This runs on an
actual XPU and guards what a mocked CPU test cannot: _forward_attn_flat_page_table
plus the real kernel produce correct attention for a scattered (non-page-aligned)
token-slot layout, for both cross-attn (non-causal) and decoder self-attn (causal).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="stage-b-test-1-gpu-xpu")


def _sdpa_ref(
    *, q, k_flat, v_flat, page_table, cache_seqlens, cu_seqlens_q, scale, causal
):
    """Per-request SDPA oracle: gather each request's valid slots straight from
    the page_table and attend. fp32 math for a stable bf16 comparison."""
    outs = []
    for i in range(page_table.shape[0]):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        sl = int(cache_seqlens[i])
        q_i = q[qs:qe].float()
        slots = page_table[i, :sl].long()
        k_i = k_flat.index_select(0, slots).float()
        v_i = v_flat.index_select(0, slots).float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        lq, lk = qe - qs, sl
        if causal and lq > 0 and lk > 0:
            row = torch.arange(lq, device=q.device).unsqueeze(1)
            col = torch.arange(lk, device=q.device).unsqueeze(0)
            keep = col <= (lk - lq + row)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_i))
    return torch.cat(outs, dim=0)


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an Intel XPU"
)
class TestXPUEncoderDecoderVarlen(CustomTestCase):
    # Whisper-large-v3 is MHA (num_kv_heads == num_heads), so use MHA here to keep
    # the reference exact; head_dim=64 satisfies the kernel's alignment.
    H, D, TOTAL_SLOTS = 8, 64, 64

    def setUp(self):
        torch.manual_seed(0)
        self.dev = torch.device("xpu")
        self.backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        self.backend.is_encoder_decoder = True
        # Deliberately scattered (non-page-aligned) slot indices: a paged kernel
        # would mis-read these; the page_size=1 gather must be alignment-agnostic.
        perm = torch.randperm(self.TOTAL_SLOTS)
        self.k_flat = torch.randn(
            self.TOTAL_SLOTS, self.H, self.D, dtype=torch.bfloat16, device=self.dev
        )[perm].contiguous()
        self.v_flat = torch.randn(
            self.TOTAL_SLOTS, self.H, self.D, dtype=torch.bfloat16, device=self.dev
        )[perm].contiguous()

    def _check(self, *, cache_seqlens, cu_seqlens_q, causal):
        cache_seqlens = cache_seqlens.to(self.dev)
        cu_seqlens_q = cu_seqlens_q.to(self.dev)
        num_rows = int(cu_seqlens_q[-1])
        m = int(cache_seqlens.max())
        # Rows packed valid-first; scatter distinct slots per request (build the
        # permutation on CPU, then move -- randperm(device="xpu") is unreliable).
        page_table = (
            torch.stack(
                [
                    torch.randperm(self.TOTAL_SLOTS)[:m]
                    for _ in range(cache_seqlens.numel())
                ]
            )
            .to(torch.int32)
            .to(self.dev)
        )
        q = torch.randn(num_rows, self.H, self.D, dtype=torch.bfloat16, device=self.dev)
        layer = SimpleNamespace(
            is_cross_attention=not causal,
            tp_q_head_num=self.H,
            tp_k_head_num=self.H,
            tp_v_head_num=self.H,
            head_dim=self.D,
            scaling=0.5,
            logit_cap=0.0,
        )
        key_cache = self.k_flat.view(-1, 1, self.H, self.D)
        value_cache = self.v_flat.view(-1, 1, self.H, self.D)

        # causal=True mirrors decoder self-attn, causal=False cross-attn; the
        # generic helper takes the (page_table, cache_seqlens, causal) that the
        # caller's _encoder_decoder_page_table dispatch would have selected.
        got = self.backend._forward_attn_flat_page_table(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            layer=layer,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            causal=causal,
        )
        torch.xpu.synchronize()
        want = _sdpa_ref(
            q=q,
            k_flat=self.k_flat,
            v_flat=self.v_flat,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            scale=layer.scaling,
            causal=causal,
        )
        self.assertEqual(tuple(got.shape), (num_rows, self.H, self.D))
        self.assertTrue(torch.isfinite(got).all(), "attention output must be finite")
        # bf16 kernel vs fp32 reference: loose tolerance.
        torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)

    def test_cross_attention_decode_on_xpu(self):
        # 1 query/request, attend all encoder KV, non-causal, unequal lengths.
        self._check(
            cache_seqlens=torch.tensor([5, 8], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            causal=False,
        )

    def test_decoder_self_attention_decode_on_xpu(self):
        self._check(
            cache_seqlens=torch.tensor([4, 6], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            causal=True,
        )

    def test_mixed_empty_batch_on_xpu(self):
        # Mixed batch: request 0 has no keys (cache_seqlens==0), request 1 has some.
        # On the real kernel the empty request's rows come back NaN/inf, so the
        # backend must zero them without corrupting request 1. The SDPA oracle
        # yields zeros for the empty request (empty-key contraction), so the shared
        # assert_close plus the finiteness check guard against a regression that
        # drops the zeroing and leaks NaN into the output.
        self._check(
            cache_seqlens=torch.tensor([0, 6], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            causal=False,
        )


if __name__ == "__main__":
    unittest.main()
