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


class TestEncoderDecoderForward(unittest.TestCase):
    """Encoder-decoder attention on XPU calls flash_attn_with_kvcache with a
    page_size=1 view; sgl-kernel-xpu PR #454 gathers the token-slot page_table and
    runs varlen inside the kernel. These guard the backend's own responsibilities
    -- the cross-vs-self dispatch, the page_size=1 view, the empty-KV zero guard,
    and the per-request encoder-offset metadata. Real-kernel numerical correctness
    is covered on device in test/registered/xpu/test_xpu_encoder_decoder_varlen.py.
    """

    HQ, HK, D, TOTAL_SLOTS = 4, 2, 8, 40

    def setUp(self):
        torch.manual_seed(0)
        self.backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        self.k_flat = torch.randn(self.TOTAL_SLOTS, self.HK, self.D)
        self.v_flat = torch.randn(self.TOTAL_SLOTS, self.HK, self.D)

    def _layer(self, is_cross):
        return SimpleNamespace(
            is_cross_attention=is_cross,
            tp_q_head_num=self.HQ,
            tp_k_head_num=self.HK,
            tp_v_head_num=self.HK,
            head_dim=self.D,
            scaling=0.5,
            logit_cap=0.0,
        )

    def test_dispatch_and_forward_cross_vs_self(self):
        # The caller picks (page_table, cache_seqlens, causal) via
        # _encoder_decoder_page_table -- cross-attn -> encoder_page_table +
        # encoder_lens_int32 + causal=False; self-attn -> page_table +
        # cache_seqlens_int32 + causal=True -- then hands them to the generic
        # _forward_attn_flat_page_table, which must forward them unchanged with a
        # page_size=1 k_cache (shape[1]==1) so PR #454 routes to the varlen gather.
        enc_pt = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        dec_pt = (torch.arange(4, dtype=torch.int32) + 10).unsqueeze(0)
        metadata = SimpleNamespace(
            encoder_page_table=enc_pt,
            encoder_lens_int32=torch.tensor([5], dtype=torch.int32),
            page_table=dec_pt,
            cache_seqlens_int32=torch.tensor([4], dtype=torch.int32),
        )
        key_cache = self.k_flat.view(-1, 1, self.HK, self.D)
        value_cache = self.v_flat.view(-1, 1, self.HK, self.D)
        q = torch.randn(1, self.HQ * self.D)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)

        for is_cross, exp_pt, exp_seqlens, exp_causal in (
            (True, enc_pt, metadata.encoder_lens_int32, False),
            (False, dec_pt, metadata.cache_seqlens_int32, True),
        ):
            layer = self._layer(is_cross)
            page_table, cache_seqlens, causal = (
                self.backend._encoder_decoder_page_table(layer, metadata)
            )
            self.assertTrue(torch.equal(page_table, exp_pt))
            self.assertTrue(torch.equal(cache_seqlens, exp_seqlens))
            self.assertEqual(causal, exp_causal)

            captured = {}

            def fake_kvcache(*_, **kw):
                captured.update(kw)
                return kw["q"].new_zeros(
                    (kw["q"].shape[0], kw["q"].shape[1], kw["v_cache"].shape[-1])
                )

            with patch.object(xpu_backend, "flash_attn_with_kvcache", fake_kvcache):
                self.backend._forward_attn_flat_page_table(
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
            self.assertTrue(torch.equal(captured["page_table"], exp_pt))
            self.assertTrue(torch.equal(captured["cache_seqlens"], exp_seqlens))
            self.assertEqual(captured["causal"], exp_causal)
            self.assertEqual(captured["k_cache"].shape[1], 1)  # page_size=1 view
            self.assertEqual(captured["max_seqlen_q"], 1)

    def test_all_empty_returns_zeros_without_kernel(self):
        # Whisper text-only warmup: all cache_seqlens == 0. PR #454's page_size=1
        # path returns NaN for an empty KV, so the backend must short-circuit to
        # zeros and never launch the kernel.
        key_cache = self.k_flat.view(-1, 1, self.HK, self.D)
        value_cache = self.v_flat.view(-1, 1, self.HK, self.D)
        q = torch.randn(1, self.HQ * self.D)
        sentinel = MagicMock(side_effect=AssertionError("kernel must not run"))
        with patch.object(xpu_backend, "flash_attn_with_kvcache", sentinel):
            out = self.backend._forward_attn_flat_page_table(
                q=q,
                key_cache=key_cache,
                value_cache=value_cache,
                layer=self._layer(True),
                page_table=torch.zeros(1, 0, dtype=torch.int32),
                cache_seqlens=torch.zeros(1, dtype=torch.int32),
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
                max_seqlen_q=1,
                causal=False,
            )
        sentinel.assert_not_called()
        self.assertTrue(torch.equal(out, torch.zeros(1, self.HQ, self.D)))

    def test_mixed_empty_zeros_only_empty_request_rows(self):
        # Mixed batch: request 0 has cache_seqlens==0 (no keys), request 1 has
        # keys. PR #454 returns NaN for the empty request's rows, so the backend
        # must zero exactly those rows and leave the rest untouched. Unequal query
        # counts (2 and 3) exercise the cu_seqlens_q -> per-request row mapping.
        key_cache = self.k_flat.view(-1, 1, self.HK, self.D)
        value_cache = self.v_flat.view(-1, 1, self.HK, self.D)
        q = torch.randn(5, self.HQ * self.D)

        def fake_kvcache(*_, **kw):
            # All-ones (never-NaN) sentinel so zeroed rows are distinguishable.
            return kw["q"].new_ones(
                (kw["q"].shape[0], kw["q"].shape[1], kw["v_cache"].shape[-1])
            )

        with patch.object(xpu_backend, "flash_attn_with_kvcache", fake_kvcache):
            out = self.backend._forward_attn_flat_page_table(
                q=q,
                key_cache=key_cache,
                value_cache=value_cache,
                layer=self._layer(True),
                page_table=torch.zeros(2, 4, dtype=torch.int32),
                cache_seqlens=torch.tensor([0, 4], dtype=torch.int32),
                cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
                max_seqlen_q=3,
                causal=False,
            )
        self.assertTrue(torch.equal(out[:2], torch.zeros(2, self.HQ, self.D)))
        self.assertTrue(torch.equal(out[2:], torch.ones(3, self.HQ, self.D)))

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
        # lengths + segment boundaries captured for the kernel's internal gather.
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
