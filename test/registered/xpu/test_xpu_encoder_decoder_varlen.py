"""Real-device XPU tests for the encoder-decoder and sliding-window attention paths.

The backend calls flash_attn_with_kvcache with a page_size=1 view; sgl-kernel-xpu
PR #454 detects that and gathers + runs varlen inside the kernel. This runs on an
actual XPU and guards what a mocked CPU test cannot: _forward_encoder_decoder_attn
plus the real kernel produce correct attention for a scattered (non-page-aligned)
token-slot layout, for both cross-attn (non-causal) and decoder self-attn (causal).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="extra-test-1-gpu-xpu")


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
        scale, softcap = 0.5, 0.0
        key_cache = self.k_flat.view(-1, 1, self.H, self.D)
        value_cache = self.v_flat.view(-1, 1, self.H, self.D)

        # causal=True mirrors decoder self-attn, causal=False cross-attn; the
        # generic helper takes the (page_table, cache_seqlens, causal) that the
        # caller's _encoder_decoder_page_table dispatch would have selected.
        got = self.backend._forward_encoder_decoder_attn(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            scale=scale,
            softcap=softcap,
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
            scale=scale,
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


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestXPUSlidingWindowExtend(CustomTestCase):
    """Over a context longer than twice the window, a non-causal sliding-window
    layer attends |q - k| <= W keys and a causal one q - W <= k <= q."""

    HQ, HK, D, PAGE, WINDOW, SEQ = 4, 2, 256, 128, 1023, 2560

    def setUp(self):
        torch.manual_seed(0)
        self.dev = torch.device("xpu")
        shape = (self.SEQ, self.HK, self.D)
        self.k = torch.randn(shape, dtype=torch.bfloat16, device=self.dev)
        self.v = torch.randn(shape, dtype=torch.bfloat16, device=self.dev)
        # Peaked scores make each output depend on a few keys, so a wrong window
        # moves it far beyond the bf16 tolerance.
        self.q = 4 * torch.randn(
            self.SEQ, self.HQ, self.D, dtype=torch.bfloat16, device=self.dev
        )

    def _forward_extend(self, attn_type):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        seq = torch.tensor([self.SEQ], dtype=torch.int32, device=self.dev)
        cu_seqlens = torch.tensor([0, self.SEQ], dtype=torch.int32, device=self.dev)
        backend.use_mla = False
        backend.dcp_size = 1
        backend.attention_chunk_size = None
        backend.topk = 1
        backend.page_size = self.PAGE
        backend.num_splits = 0
        backend.is_encoder_decoder = False
        backend.use_sliding_window_kv_pool = False
        backend.bidirectional_block_ids = None
        backend.bidirectional_sliding_layers_only = False
        backend.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(
                read=lambda _: SimpleNamespace(kv=lambda: (self.k, self.v))
            )
        )
        backend.forward_metadata = SimpleNamespace(
            local_attn_metadata=None,
            swa_spec_metadata=None,
            page_table=torch.arange(
                self.SEQ // self.PAGE, dtype=torch.int32, device=self.dev
            ).unsqueeze(0),
            cu_seqlens_q=cu_seqlens,
            cache_seqlens_int32=seq,
            max_seq_len_q=self.SEQ,
            max_seq_len_k=self.SEQ,
            cu_seqlens_k=cu_seqlens,
        )
        layer = SimpleNamespace(
            layer_id=0,
            attn_type=attn_type,
            is_cross_attention=False,
            sliding_window_size=self.WINDOW,
            tp_q_head_num=self.HQ,
            tp_k_head_num=self.HK,
            tp_v_head_num=self.HK,
            head_dim=self.D,
            v_head_dim=self.D,
            scaling=self.D**-0.5,
            logit_cap=0.0,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            encoder_lens=None,
            extend_prefix_lens_cpu=[0],
            _attn_output=None,
        )
        out = backend.forward_extend(
            q=self.q.view(self.SEQ, -1),
            k=None,
            v=None,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=False,
        )
        torch.xpu.synchronize()
        return out.view(self.SEQ, self.HQ, self.D).float()

    def _band_reference(self, *, causal):
        pos = torch.arange(self.SEQ, device=self.dev)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        keep = (
            (dist >= 0) & (dist <= self.WINDOW) if causal else dist.abs() <= self.WINDOW
        )
        group = self.HQ // self.HK
        k = self.k.float().repeat_interleave(group, dim=1)
        v = self.v.float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", self.q.float(), k) * self.D**-0.5
        scores = scores.masked_fill(~keep, float("-inf"))
        return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)

    def test_bidirectional_sliding_layer_sees_both_sides(self):
        got = self._forward_extend(AttentionType.DECODER_BIDIRECTIONAL)
        want = self._band_reference(causal=False)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)

    def test_causal_sliding_layer_keeps_a_left_only_window(self):
        got = self._forward_extend(AttentionType.DECODER)
        want = self._band_reference(causal=True)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
