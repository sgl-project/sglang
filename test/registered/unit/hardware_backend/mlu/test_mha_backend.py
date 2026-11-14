"""Eager MLU paged GQA, cached prefixes, mixed batches and rejected features."""

import unittest
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlu.attention.mha_backend import MLUMHABackend
from sglang.srt.hardware_backend.mlu.memory_pool import MLUMHATokenToKVPool
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestMLUMHABackend(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        # These tests construct MHA pools without platform configuration.
        stack.enter_context(envs.SGLANG_USE_HND_KVCACHE.override(True))
        torch.manual_seed(42)

    def _create_fixture(self, mode, query_lens, mixed_prefill_reqs=None):
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
        req_to_token = torch.zeros(2, 128, device="mlu", dtype=torch.int32)
        for i, pages in enumerate(((1, 5), (3, 7, 10))):
            req_to_token[i, : 16 * len(pages)] = torch.cat(
                [torch.arange(p * 16, (p + 1) * 16, device="mlu") for p in pages]
            )
        backend = MLUMHABackend(
            NS(
                device="mlu",
                page_size=16,
                model_config=NS(context_len=128),
                req_to_token_pool=NS(req_to_token=req_to_token),
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
        kv_lens = [21, 35]
        k = torch.randn(sum(kv_lens), 2, 128, device="mlu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        loc = torch.cat(
            [
                req_to_token[i, kv_len - query_len : kv_len]
                for i, (kv_len, query_len) in enumerate(zip(kv_lens, query_lens))
            ]
        )
        kv_indices = torch.cat(
            [req_to_token[i, :length] for i, length in enumerate(kv_lens)]
        )
        pool.set_kv_buffer(layer, kv_indices, k, v)
        # Leave only historical KV in the fixture: forward must write current tokens.
        empty = torch.zeros(sum(query_lens), 2, 128, device="mlu", dtype=k.dtype)
        pool.set_kv_buffer(layer, loc, empty, empty)
        forward_batch = NS(
            batch_size=2,
            req_pool_indices=torch.arange(2, device="mlu"),
            seq_lens=torch.tensor(kv_lens, device="mlu"),
            seq_lens_cpu=torch.tensor(kv_lens),
            extend_seq_lens=torch.tensor(query_lens, device="mlu"),
            extend_seq_lens_cpu=query_lens,
            extend_prefix_lens_cpu=[
                kv_len - query_len for kv_len, query_len in zip(kv_lens, query_lens)
            ],
            mixed_prefill_reqs=mixed_prefill_reqs,
            forward_mode=mode,
            out_cache_loc=loc,
        )
        backend.init_forward_metadata(forward_batch)
        q = torch.randn(sum(query_lens), 8, 128, device="mlu", dtype=k.dtype)
        current_k, current_v = (
            torch.cat(
                [
                    chunk[-length:]
                    for chunk, length in zip(tensor.split(kv_lens), query_lens)
                ]
            )
            for tensor in (k, v)
        )
        return NS(
            backend=backend,
            layer=layer,
            forward_batch=forward_batch,
            q=q,
            k=k,
            v=v,
            current_k=current_k,
            current_v=current_v,
        )

    def _check_attention(self, mode, query_lens, mixed_prefill_reqs, window=-1):
        # Create the backend, batch, and input tensors.
        fixture = self._create_fixture(mode, query_lens, mixed_prefill_reqs)
        fixture.layer.sliding_window_size = window

        # Compute causal GQA outputs and LSE on CPU for each request.
        expected_outputs, expected_lses = [], []
        kv_lens = fixture.forward_batch.seq_lens_cpu.tolist()
        for q, k, v in zip(
            fixture.q.cpu().float().split(query_lens),
            fixture.k.cpu().float().split(kv_lens),
            fixture.v.cpu().float().split(kv_lens),
        ):
            query_len, kv_len = q.shape[0], k.shape[0]
            q = q.transpose(0, 1)
            k = k.repeat_interleave(4, 1).transpose(0, 1)
            v = v.repeat_interleave(4, 1).transpose(0, 1)
            scores = q @ k.transpose(-1, -2) * fixture.layer.scaling
            key_positions = torch.arange(kv_len)[None, :]
            query_positions = torch.arange(kv_len - query_len, kv_len)[:, None]
            mask = key_positions > query_positions
            if window >= 0:
                mask |= key_positions < query_positions - window
            scores = scores.masked_fill(mask, float("-inf"))
            expected_outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
            expected_lses.append(scores.logsumexp(-1).transpose(0, 1))
        expected = torch.cat(expected_outputs)
        expected_lse = torch.cat(expected_lses)

        # Run the public forward entry, then reuse the populated cache without K/V.
        forward = partial(
            fixture.backend.forward,
            q=fixture.q,
            layer=fixture.layer,
            forward_batch=fixture.forward_batch,
        )
        actual = forward(k=fixture.current_k, v=fixture.current_v)
        cached = forward(k=None, v=None, save_kv_cache=False)
        with_lse, lse = forward(k=None, v=None, save_kv_cache=False, return_lse=True)

        # Compare outputs, cache reuse, and LSE against the CPU reference.
        torch.testing.assert_close(
            actual.cpu().float(),
            expected.flatten(1),
            rtol=3e-2,
            atol=3e-2,
        )
        torch.testing.assert_close(cached, actual, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(with_lse, actual, rtol=3e-2, atol=3e-2)
        self.assertEqual(lse.shape, expected_lse.shape)
        self.assertEqual(lse.dtype, torch.float32)
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=1e-3, atol=1e-3)

    def test_paged_gqa(self):
        for mode, query_lens, count in (
            (ForwardMode.EXTEND, [21, 35], None),  # No cached prefixes.
            (ForwardMode.EXTEND, [2, 3], None),  # Both requests have cached prefixes.
            (ForwardMode.EXTEND, [21, 3], None),  # Only one request has a prefix.
            (ForwardMode.DECODE, [1, 1], None),  # One decode token per request.
            (ForwardMode.MIXED, [2, 3], 2),  # All requests are prefill.
            (ForwardMode.MIXED, [2, 1], 1),  # Cached-prefix prefill plus decode.
            (ForwardMode.MIXED, [21, 1], 1),  # Uncached prefill plus decode.
            (ForwardMode.MIXED, [1, 1], 1),  # One-token prefill plus decode.
            (ForwardMode.MIXED, [1, 1], 0),  # All requests are decode.
        ):
            with self.subTest(mode=mode, query_lens=query_lens, count=count):
                self._check_attention(mode, query_lens, count)

    def test_mixed_sliding_window(self):
        """Prefill and decode must include the same left window boundary."""
        self._check_attention(ForwardMode.MIXED, [2, 1], 1, window=8)

    def test_invalid_mixed_prefill_reqs_rejected(self):
        """Reject missing or out-of-range prefill request counts in mixed batches."""
        fixture = self._create_fixture(ForwardMode.MIXED, [2, 1], 1)
        for count in (None, -1, 3):
            with self.subTest(count=count):
                fixture.forward_batch.mixed_prefill_reqs = count
                with self.assertRaises(ValueError):
                    fixture.backend.init_forward_metadata(fixture.forward_batch)

    def test_unsupported_features_rejected(self):
        for mode, query_lens in (
            (ForwardMode.EXTEND, [2, 3]),
            (ForwardMode.DECODE, [1, 1]),
        ):
            fixture = self._create_fixture(mode, query_lens)
            cases = (
                ("logit_cap", 0.5),
                ("attn_type", AttentionType.ENCODER_ONLY),
                ("attn_type", AttentionType.DECODER_BIDIRECTIONAL),
                ("is_cross_attention", True),
                ("sinks", torch.zeros(8, device="mlu", dtype=fixture.q.dtype)),
            )
            for field, value in cases:
                with self.subTest(mode=mode, feature=field):
                    # Zero sinks still request unsupported semantics.
                    kwargs = {field: value} if field == "sinks" else {}
                    with (
                        patch.object(fixture.layer, field, value, create=True),
                        self.assertRaises(NotImplementedError),
                    ):
                        fixture.backend.forward(
                            fixture.q,
                            fixture.current_k,
                            fixture.current_v,
                            fixture.layer,
                            fixture.forward_batch,
                            save_kv_cache=True,
                            **kwargs,
                        )


if __name__ == "__main__":
    unittest.main()
