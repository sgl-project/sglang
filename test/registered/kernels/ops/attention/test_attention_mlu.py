"""Eager MLU paged GQA, cached prefixes, mixed batches and rejected features."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401

from sglang.srt.hardware_backend.mlu.attention.mlu_backend import MLUAttnBackend
from sglang.srt.hardware_backend.mlu.memory_pool_mlu import MLUMHATokenToKVPool
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestAttentionMLU(CustomTestCase):
    def _fixture(self, mode, qlens):
        torch.manual_seed(42)
        pool = MLUMHATokenToKVPool(
            size=256,
            page_size=16,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=1,
            device="mlu",
            enable_memory_saver=False,
        )
        req = torch.zeros(2, 128, device="mlu", dtype=torch.int32)
        for i, pages in enumerate(((1, 5), (3, 7, 10))):
            req[i, : 16 * len(pages)] = torch.cat(
                [torch.arange(p * 16, (p + 1) * 16, device="mlu") for p in pages]
            )
        backend = MLUAttnBackend(
            NS(
                device="mlu",
                page_size=16,
                model_config=NS(context_len=128),
                req_to_token_pool=NS(req_to_token=req),
                token_to_kv_pool=pool,
                kv_cache_dtype=torch.bfloat16,
            )
        )
        layer = NS(
            layer_id=0,
            tp_q_head_num=8,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=128,
            v_head_dim=128,
            scaling=128**-0.5,
            sliding_window_size=-1,
            logit_cap=0.0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
        )
        lens = [21, 35]
        k = torch.randn(56, 2, 128, device="mlu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        loc = torch.cat(
            [req[i, n - m : n] for i, (n, m) in enumerate(zip(lens, qlens))]
        )
        pool.set_kv_buffer(layer, torch.cat([req[0, :21], req[1, :35]]), k, v)
        # Leave only historical KV in the fixture: forward must write current tokens.
        empty = torch.zeros(sum(qlens), 2, 128, device="mlu", dtype=k.dtype)
        pool.set_kv_buffer(layer, loc, empty, empty)
        fb = NS(
            batch_size=2,
            positions=torch.arange(sum(qlens), device="mlu"),
            req_pool_indices=torch.arange(2, device="mlu"),
            seq_lens=torch.tensor(lens, device="mlu"),
            seq_lens_cpu=torch.tensor(lens),
            extend_seq_lens=torch.tensor(qlens, device="mlu"),
            extend_seq_lens_cpu=qlens,
            forward_mode=mode,
            out_cache_loc=loc,
        )
        backend.init_forward_metadata(fb)
        q = torch.randn(sum(qlens), 8, 128, device="mlu", dtype=k.dtype)
        current_k = torch.cat([k[21 - qlens[0] : 21], k[56 - qlens[1] : 56]])
        current_v = torch.cat([v[21 - qlens[0] : 21], v[56 - qlens[1] : 56]])
        fn = backend.forward_decode if mode.is_decode() else backend.forward_extend
        return NS(
            pool=pool,
            backend=backend,
            layer=layer,
            fb=fb,
            q=q,
            k=k,
            v=v,
            current_k=current_k,
            current_v=current_v,
            fn=fn,
        )

    def _check(self, mode, qlens):
        f = self._fixture(mode, qlens)
        actual = f.fn(f.q, f.current_k, f.current_v, f.layer, f.fb, save_kv_cache=True)
        actual = actual.view(-1, 8, 128).cpu().float()
        offset = ko = 0
        for n, m in zip((21, 35), qlens):
            q = f.q[offset : offset + m].cpu().float().transpose(0, 1)
            k = f.k[ko : ko + n].cpu().float().repeat_interleave(4, 1).transpose(0, 1)
            v = f.v[ko : ko + n].cpu().float().repeat_interleave(4, 1).transpose(0, 1)
            scores = q @ k.transpose(-1, -2) * f.layer.scaling
            mask = torch.arange(n)[None, :] > torch.arange(n - m, n)[:, None]
            expected = (
                scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
            ).transpose(0, 1)
            torch.testing.assert_close(
                actual[offset : offset + m], expected, rtol=0.03, atol=0.03
            )
            offset += m
            ko += n

    def test_paged_gqa(self):
        for mode, qlens in (
            (ForwardMode.EXTEND, [21, 35]),
            (ForwardMode.EXTEND, [2, 3]),
            (ForwardMode.DECODE, [1, 1]),
            (ForwardMode.MIXED, [2, 1]),
            (ForwardMode.MIXED, [1, 1]),
        ):
            with self.subTest(mode=mode, qlens=qlens):
                self._check(mode, qlens)

    def test_unsupported_features_rejected_before_kernel(self):
        for mode, qlens in ((ForwardMode.EXTEND, [2, 3]), (ForwardMode.DECODE, [1, 1])):
            f = self._fixture(mode, qlens)
            cases = (
                ("logit_cap", 0.5, "logit_cap"),
                ("attn_type", AttentionType.ENCODER_ONLY, "causal decoder"),
                ("attn_type", AttentionType.DECODER_BIDIRECTIONAL, "causal decoder"),
                ("is_cross_attention", True, "cross-attention"),
                (
                    "sinks",
                    torch.zeros(8, device="mlu", dtype=f.q.dtype),
                    "attention sinks",
                ),
            )
            if mode.is_decode():
                cases = cases[-1:]
            for field, value, message in cases:
                with self.subTest(mode=mode, feature=field):
                    # Zero sinks still request unsupported semantics.
                    kwargs = {"sinks": value} if field == "sinks" else {}
                    with (
                        patch.object(f.layer, field, value, create=True),
                        patch("torch_mlu_ops.flash_attention") as prefill,
                        patch("torch_mlu_ops.single_query_cached_kv_attn") as decode,
                    ):
                        with self.assertRaisesRegex(NotImplementedError, message):
                            f.fn(
                                f.q,
                                f.current_k,
                                f.current_v,
                                f.layer,
                                f.fb,
                                save_kv_cache=True,
                                **kwargs,
                            )
                        prefill.assert_not_called()
                        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
