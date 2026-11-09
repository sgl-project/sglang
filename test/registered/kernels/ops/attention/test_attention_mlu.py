"""Eager MLU paged GQA, cached prefixes, mixed batches and rejected features."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401
import torch_mlu_ops

from sglang.srt.hardware_backend.mlu.attention.mlu_backend import MLUAttnBackend
from sglang.srt.hardware_backend.mlu.memory_pool_mlu import MLUMHATokenToKVPool
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestAttentionMLU(CustomTestCase):
    def _fixture(self, mode, qlens, mixed_prefill_reqs=None):
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
            extend_prefix_lens_cpu=[n - m for n, m in zip(lens, qlens)],
            mixed_prefill_reqs=mixed_prefill_reqs,
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

    def _check(self, mode, qlens, mixed_prefill_reqs, window=-1):
        f = self._fixture(mode, qlens, mixed_prefill_reqs)
        f.layer.sliding_window_size = window
        with (
            patch.object(
                torch_mlu_ops, "flash_attention", wraps=torch_mlu_ops.flash_attention
            ) as prefill,
            patch.object(
                torch_mlu_ops,
                "single_query_cached_kv_attn",
                wraps=torch_mlu_ops.single_query_cached_kv_attn,
            ) as decode,
        ):
            actual = f.fn(
                f.q, f.current_k, f.current_v, f.layer, f.fb, save_kv_cache=True
            )
        if mode.is_mixed():
            # Request identity, not query length, selects the prefill path.
            self.assertEqual(prefill.called, mixed_prefill_reqs > 0)
            self.assertEqual(decode.called, mixed_prefill_reqs < f.fb.batch_size)
            if 0 < mixed_prefill_reqs < f.fb.batch_size:
                split = sum(qlens[:mixed_prefill_reqs])
                self.assertEqual(
                    prefill.call_args.kwargs["out"].data_ptr(), actual.data_ptr()
                )
                self.assertEqual(
                    decode.call_args.kwargs["out"].data_ptr(), actual[split:].data_ptr()
                )
        if prefill.called:
            count = mixed_prefill_reqs if mode.is_mixed() else f.fb.batch_size
            if not any(f.fb.extend_prefix_lens_cpu[:count]):
                self.assertIsNone(prefill.call_args.kwargs["block_tables"])
                self.assertEqual(
                    prefill.call_args.kwargs["k"].data_ptr(), f.current_k.data_ptr()
                )
                self.assertEqual(
                    prefill.call_args.kwargs["v"].data_ptr(), f.current_v.data_ptr()
                )
            else:
                self.assertIsNotNone(prefill.call_args.kwargs["block_tables"])
        # A later consumer may reuse populated KV without receiving current K/V.
        cached = f.fn(f.q, None, None, f.layer, f.fb, save_kv_cache=False)
        torch.testing.assert_close(cached, actual, rtol=0.03, atol=0.03)
        actual = actual.view(-1, 8, 128).cpu().float()
        offset = ko = 0
        for n, m in zip((21, 35), qlens):
            q = f.q[offset : offset + m].cpu().float().transpose(0, 1)
            k = f.k[ko : ko + n].cpu().float().repeat_interleave(4, 1).transpose(0, 1)
            v = f.v[ko : ko + n].cpu().float().repeat_interleave(4, 1).transpose(0, 1)
            scores = q @ k.transpose(-1, -2) * f.layer.scaling
            key_positions = torch.arange(n)[None, :]
            query_positions = torch.arange(n - m, n)[:, None]
            mask = key_positions > query_positions
            if window >= 0:
                mask |= key_positions < query_positions - window
            expected = (
                scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
            ).transpose(0, 1)
            torch.testing.assert_close(
                actual[offset : offset + m], expected, rtol=0.03, atol=0.03
            )
            offset += m
            ko += n

    def test_paged_gqa(self):
        for mode, qlens, count in (
            (ForwardMode.EXTEND, [21, 35], None),
            (ForwardMode.EXTEND, [2, 3], None),
            (ForwardMode.EXTEND, [21, 3], None),  # Only one request has a prefix.
            (ForwardMode.DECODE, [1, 1], None),
            (ForwardMode.MIXED, [2, 3], 2),
            (ForwardMode.MIXED, [2, 1], 1),
            (ForwardMode.MIXED, [21, 1], 1),
            (ForwardMode.MIXED, [1, 1], 1),
            (ForwardMode.MIXED, [1, 1], 0),
        ):
            with self.subTest(mode=mode, qlens=qlens, count=count):
                self._check(mode, qlens, count)

    def test_mixed_sliding_window(self):
        """Prefill and decode must include the same left window boundary."""
        self._check(ForwardMode.MIXED, [2, 1], 1, window=8)

    def test_mixed_metadata_rejects_missing_boundary_and_multi_query_decode(self):
        """Missing scheduler identity and unsupported decode shapes must fail closed."""
        f = self._fixture(ForwardMode.MIXED, [2, 1], 1)
        for count in (None, -1, 3):
            with self.subTest(count=count):
                f.fb.mixed_prefill_reqs = count
                with self.assertRaisesRegex(ValueError, "mixed_prefill_reqs"):
                    f.backend.init_forward_metadata(f.fb)
        f.fb.mixed_prefill_reqs = 1
        f.fb.extend_seq_lens_cpu = [2, 2]
        with self.assertRaisesRegex(NotImplementedError, "one query token per decode"):
            f.backend.init_forward_metadata(f.fb)

    def test_unsupported_features_rejected_before_kv_write(self):
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
                ("return_lse", True, "return_lse"),
            )
            if mode.is_decode():
                cases = cases[-2:]
            for field, value, message in cases:
                with self.subTest(mode=mode, feature=field):
                    # Zero sinks still request unsupported semantics.
                    kwargs = {field: value} if field in ("sinks", "return_lse") else {}
                    with (
                        patch.object(f.layer, field, value, create=True),
                        patch.object(f.pool, "set_kv_buffer") as write_kv,
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
                        write_kv.assert_not_called()
                        prefill.assert_not_called()
                        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
