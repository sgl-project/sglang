import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.flash_attn",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention import xpu_backend
    from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _segmented_varlen_ref(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    softcap,
    return_softmax_lse,
):
    """Pure-torch stand-in for the XPU varlen kernel.

    Segments the ragged q/k/v by cu_seqlens, repeats KV heads for GQA, applies
    bottom-right causal masking, and returns concatenated per-segment SDPA. Used
    to exercise the real gather + cu_seqlens plumbing in ``_varlen_gather_attn``
    on CPU without the device kernel.
    """
    assert not return_softmax_lse
    g = q.shape[1] // k.shape[1]
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        q_i = q[qs:qe]  # (Lq, Hq, D)
        k_i = k[ks:ke].repeat_interleave(g, dim=1)  # (Lk, Hq, D)
        v_i = v[ks:ke].repeat_interleave(g, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_i.float(), k_i.float()) * softmax_scale
        if softcap and softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        lq, lk = qe - qs, ke - ks
        if causal and lq > 0 and lk > 0:
            row = torch.arange(lq).unsqueeze(1)
            col = torch.arange(lk).unsqueeze(0)
            keep = col <= (lk - lq + row)  # bottom-right causal
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_i).to(q.dtype))
    return torch.cat(outs, dim=0)


def _dense_reference(
    *,
    q,
    k_flat,
    v_flat,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    scale,
    causal,
    softcap=0.0,
):
    """Independent per-request SDPA that gathers valid slots straight from the
    page_table (no ragged packing) — the oracle the helper must reproduce."""
    g = q.shape[1] // k_flat.shape[1]
    outs = []
    for i in range(page_table.shape[0]):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        sl = int(cache_seqlens[i])
        q_i = q[qs:qe]
        slots = page_table[i, :sl].long()
        k_i = k_flat.index_select(0, slots).repeat_interleave(g, dim=1)
        v_i = v_flat.index_select(0, slots).repeat_interleave(g, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_i.float(), k_i.float()) * scale
        if softcap and softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        lq, lk = qe - qs, sl
        if causal and lq > 0 and lk > 0:
            row = torch.arange(lq).unsqueeze(1)
            col = torch.arange(lk).unsqueeze(0)
            keep = col <= (lk - lq + row)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_i).to(q.dtype))
    return torch.cat(outs, dim=0)


class TestEncoderDecoderVarlenGather(unittest.TestCase):
    """encoder-decoder attention on XPU gathers token-slot-indexed KV into
    a ragged buffer and runs the non-paged varlen kernel (page_size=1 semantics),
    because the paged kernel mis-indexes the token-granular encoder KV at
    page_size 64/128. These guard the gather/packing math and the zero-length
    (text-only warmup) short-circuit.
    """

    HQ, HK, D = 4, 2, 8
    TOTAL_SLOTS = 40

    def setUp(self):
        torch.manual_seed(0)
        self.backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        # Deliberately non-page-aligned slot indices: a paged kernel would
        # mis-read these; the dense gather must not care about alignment.
        perm = torch.randperm(self.TOTAL_SLOTS)
        self.k_flat = torch.randn(self.TOTAL_SLOTS, self.HK, self.D)[perm].contiguous()
        self.v_flat = torch.randn(self.TOTAL_SLOTS, self.HK, self.D)[perm].contiguous()

    def _run_and_compare(
        self, *, page_table, cache_seqlens, cu_seqlens_q, causal, softcap=0.0
    ):
        num_rows = int(cu_seqlens_q[-1])
        q = torch.randn(num_rows, self.HQ, self.D)
        scale = 0.5
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
        with patch.object(xpu_backend, "flash_attn_varlen_func", _segmented_varlen_ref):
            got = self.backend._varlen_gather_attn(
                q=q,
                k_flat=self.k_flat,
                v_flat=self.v_flat,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                scale=scale,
                softcap=softcap,
                causal=causal,
            )
        want = _dense_reference(
            q=q,
            k_flat=self.k_flat,
            v_flat=self.v_flat,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            scale=scale,
            causal=causal,
            softcap=softcap,
        )
        self.assertEqual(tuple(got.shape), (num_rows, self.HQ, self.D))
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_cross_attention_decode(self):
        # One query per request (decode), attends to all encoder KV (causal=False),
        # with unequal per-request encoder lengths (numel()==1 restriction lifted).
        cache_seqlens = torch.tensor([5, 12, 7], dtype=torch.int32)
        m = int(cache_seqlens.max())
        page_table = torch.stack([torch.arange(m) + i * m for i in range(3)]).to(
            torch.int32
        )
        cu_seqlens_q = torch.arange(0, 4, dtype=torch.int32)  # 1 q/req
        self._run_and_compare(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            causal=False,
        )

    def test_cross_attention_extend(self):
        # Multiple query tokens per request (decoder prompt), causal=False.
        cache_seqlens = torch.tensor([6, 9], dtype=torch.int32)
        m = int(cache_seqlens.max())
        page_table = torch.stack([torch.arange(m), torch.arange(m) + m]).to(torch.int32)
        cu_seqlens_q = torch.tensor([0, 3, 7], dtype=torch.int32)  # 3 + 4 queries
        self._run_and_compare(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            causal=False,
            softcap=30.0,
        )

    def test_decoder_self_attention_causal(self):
        # Decoder self-attn: causal, per-request KV, query_len == key_len (Whisper
        # has no decoder prefix so bottom-right == standard causal).
        cache_seqlens = torch.tensor([4, 6], dtype=torch.int32)
        m = int(cache_seqlens.max())
        page_table = torch.stack([torch.arange(m) + 3, torch.arange(m) + 20]).to(
            torch.int32
        )
        cu_seqlens_q = torch.tensor([0, 4, 10], dtype=torch.int32)
        self._run_and_compare(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            causal=True,
        )

    def test_zero_length_short_circuits(self):
        # Regression: an empty-KV (max_seqlen_k == 0) varlen call faults the XPU
        # device. The helper must return exact zeros and never launch the kernel.
        cache_seqlens = torch.zeros(2, dtype=torch.int32)
        page_table = torch.zeros(2, 5, dtype=torch.int32)
        cu_seqlens_q = torch.arange(0, 3, dtype=torch.int32)
        q = torch.randn(2, self.HQ, self.D)
        sentinel = MagicMock(side_effect=AssertionError("kernel must not run"))
        with patch.object(xpu_backend, "flash_attn_varlen_func", sentinel):
            got = self.backend._varlen_gather_attn(
                q=q,
                k_flat=self.k_flat,
                v_flat=self.v_flat,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                scale=0.5,
                softcap=0.0,
                causal=False,
            )
        sentinel.assert_not_called()
        self.assertTrue(torch.equal(got, torch.zeros(2, self.HQ, self.D)))

    def test_forward_dispatch_selects_cross_vs_self(self):
        # _forward_encoder_decoder_mha must route cross-attn to encoder_page_table
        # (causal=False) and self-attn to page_table (causal=True).
        enc_seqlens = torch.tensor([5], dtype=torch.int32)
        dec_seqlens = torch.tensor([4], dtype=torch.int32)
        enc_pt = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        dec_pt = (torch.arange(4, dtype=torch.int32) + 10).unsqueeze(0)
        metadata = SimpleNamespace(
            encoder_page_table=enc_pt,
            encoder_lens_int32=enc_seqlens,
            page_table=dec_pt,
            cache_seqlens_int32=dec_seqlens,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len_q=1,
        )
        layer = SimpleNamespace(
            tp_q_head_num=self.HQ,
            tp_k_head_num=self.HK,
            tp_v_head_num=self.HK,
            head_dim=self.D,
            scaling=0.5,
            logit_cap=0.0,
        )
        key_cache = self.k_flat.view(-1, 1, self.HK, self.D)
        value_cache = self.v_flat.view(-1, 1, self.HK, self.D)
        q = torch.randn(1, self.HQ * self.D)

        for is_cross, page_table, seqlens, causal in (
            (True, enc_pt, enc_seqlens, False),
            (False, dec_pt, dec_seqlens, True),
        ):
            layer.is_cross_attention = is_cross
            with patch.object(
                xpu_backend, "flash_attn_varlen_func", _segmented_varlen_ref
            ):
                got = self.backend._forward_encoder_decoder_mha(
                    q=q,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    layer=layer,
                    metadata=metadata,
                    decode=True,
                )
            want = _dense_reference(
                q=q.view(1, self.HQ, self.D),
                k_flat=self.k_flat,
                v_flat=self.v_flat,
                page_table=page_table,
                cache_seqlens=seqlens,
                cu_seqlens_q=metadata.cu_seqlens_q,
                scale=layer.scaling,
                causal=causal,
            )
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_init_forward_metadata_per_request_encoder_offset(self):
        # Guards the encoder_lens.numel()==1 removal: with UNEQUAL encoder lengths,
        # init_forward_metadata must place each request's decoder self-attn page
        # table at ITS OWN encoder_lens[i] offset (not a single batch max), and
        # slice encoder KV per request. It must also skip the //page_size stride
        # for enc-dec (token-slot indices feed the varlen kernel). Fails on the old
        # scalar-max-offset slice (row0 would start at col 5, not 3).
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.page_size = 128  # >1: also exercises the enc-dec stride-skip
        backend.is_encoder_decoder = True
        backend.use_mla = False
        backend.use_sliding_window_kv_pool = False
        backend.attention_chunk_size = None
        backend.topk = 0
        # req_to_token[i, j] = 100*i + j, so gathered values reveal (row, col).
        req_to_token = torch.arange(16).unsqueeze(0) + torch.tensor([[0], [100]])
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([2, 4], dtype=torch.int64),  # decoder lengths
            seq_lens_cpu=torch.tensor([2, 4]),
            batch_size=2,
            req_pool_indices=torch.tensor([0, 1]),
            encoder_lens=torch.tensor([3, 5], dtype=torch.int64),  # UNEQUAL
            spec_info=None,
            out_cache_loc=None,
        )
        backend.init_forward_metadata(fb)
        md = backend.forward_metadata

        # Encoder KV: columns [0 : max_enc=5] of each request's row; per-request
        # lengths + segment boundaries captured for the varlen gather.
        self.assertEqual(
            md.encoder_page_table.tolist(),
            [[0, 1, 2, 3, 4], [100, 101, 102, 103, 104]],
        )
        self.assertEqual(md.encoder_lens_int32.tolist(), [3, 5])
        self.assertEqual(md.encoder_cu_seqlens_k.tolist(), [0, 3, 8])

        # Decoder self-attn KV (text_max = max(seq_lens) = 4 columns each): request 0
        # starts at col 3 (its encoder_len), request 1 at col 5 (its encoder_len).
        # Token-granular (not //128), proving the stride transform was skipped.
        self.assertEqual(
            md.page_table.tolist(),
            [[3, 4, 5, 6], [105, 106, 107, 108]],
        )


if __name__ == "__main__":
    unittest.main()
