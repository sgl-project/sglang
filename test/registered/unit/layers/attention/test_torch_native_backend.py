"""Unit tests for TorchNativeAttnBackend sliding-window masks — no server, no model loading.

Encoder-only models (e.g. ModernBERT) use a bidirectional local window
``|q - k| <= sliding_window_size``; causal decoders use
``q - sliding_window_size <= k <= q``. Expected outputs come from a direct
softmax(QK^T)V reference with an explicit mask, not from the backend's own
mask helper.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.test_utils import CustomTestCase

HEAD_DIM = 8


def _allowed(q_pos: int, k_pos: int, window: int | None, causal: bool) -> bool:
    if causal and k_pos > q_pos:
        return False
    if window is None:
        return True
    return abs(q_pos - k_pos) <= window


def _reference_attention(q, k, v, q_positions, *, window, causal, scale):
    """q: [Lq, Hq, D], k/v: [Lkv, Hkv, D] -> [Lq, Hq, D]."""
    num_q_heads, num_kv_heads = q.shape[1], k.shape[1]
    k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    mask = torch.tensor(
        [
            [_allowed(qp, kp, window, causal) for kp in range(k.shape[0])]
            for qp in q_positions
        ]
    )
    scores = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v.double())
    return out.to(q.dtype)


def _make_backend() -> TorchNativeAttnBackend:
    # Only the SDPA helpers are exercised, so skip the ModelRunner-based init.
    return TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)


class _Requests:
    """A paged-free KV layout: request i owns cache slots [offset_i, offset_i + len_i)."""

    def __init__(self, seq_lens, num_q_heads, num_kv_heads, generator):
        self.seq_lens = list(seq_lens)
        total = sum(self.seq_lens)
        max_len = max(self.seq_lens)
        self.k_cache = torch.randn(total, num_kv_heads, HEAD_DIM, generator=generator)
        self.v_cache = torch.randn(total, num_kv_heads, HEAD_DIM, generator=generator)
        self.req_to_token = torch.zeros(len(self.seq_lens), max_len, dtype=torch.int64)
        self.offsets = []
        offset = 0
        for i, length in enumerate(self.seq_lens):
            self.req_to_token[i, :length] = torch.arange(offset, offset + length)
            self.offsets.append(offset)
            offset += length
        self.req_pool_indices = torch.arange(len(self.seq_lens))
        self.num_q_heads = num_q_heads

    def kv(self, i):
        sl = slice(self.offsets[i], self.offsets[i] + self.seq_lens[i])
        return self.k_cache[sl], self.v_cache[sl]


class TestSlidingWindowMask(CustomTestCase):
    def _expected(self, q_len, kv_len, window, query_offset, causal):
        return torch.tensor(
            [
                [_allowed(query_offset + q, k, window, causal) for k in range(kv_len)]
                for q in range(q_len)
            ]
        )

    def test_mask_matches_definition(self):
        for causal in (True, False):
            for q_len, kv_len, window, offset in [
                (6, 6, 0, 0),
                (6, 6, 2, 0),
                (6, 6, 5, 0),
                (6, 6, 10, 0),
                (1, 9, 3, 8),
                (3, 9, 3, 6),
            ]:
                with self.subTest(
                    causal=causal, q_len=q_len, kv_len=kv_len, window=window
                ):
                    mask = TorchNativeAttnBackend._make_sliding_window_mask(
                        q_len=q_len,
                        kv_len=kv_len,
                        sliding_window_size=window,
                        device=torch.device("cpu"),
                        query_offset=offset,
                        causal=causal,
                    )
                    self.assertTrue(
                        torch.equal(
                            mask, self._expected(q_len, kv_len, window, offset, causal)
                        )
                    )

    def test_default_is_causal(self):
        kwargs = dict(q_len=5, kv_len=5, sliding_window_size=2, device="cpu")
        self.assertTrue(
            torch.equal(
                TorchNativeAttnBackend._make_sliding_window_mask(**kwargs),
                TorchNativeAttnBackend._make_sliding_window_mask(**kwargs, causal=True),
            )
        )

    def test_bidirectional_window_is_symmetric_and_sees_future_tokens(self):
        mask = TorchNativeAttnBackend._make_sliding_window_mask(
            q_len=7, kv_len=7, sliding_window_size=2, device="cpu", causal=False
        )
        self.assertTrue(torch.equal(mask, mask.T))
        self.assertTrue(mask[3, 5])  # two tokens ahead: inside the window
        self.assertFalse(mask[3, 6])  # three tokens ahead: outside the window


class TestSdpaSlidingWindow(CustomTestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(0)
        self.backend = _make_backend()
        self.scale = HEAD_DIM**-0.5

    def _run_extend(self, reqs, prefix_lens, *, window, causal):
        extend_lens = [s - p for s, p in zip(reqs.seq_lens, prefix_lens)]
        query = torch.randn(
            sum(extend_lens), reqs.num_q_heads, HEAD_DIM, generator=self.generator
        )
        output = torch.empty_like(query)
        self.backend._run_sdpa_forward_extend(
            query,
            output,
            reqs.k_cache,
            reqs.v_cache,
            reqs.req_to_token,
            reqs.req_pool_indices,
            torch.tensor(reqs.seq_lens),
            torch.tensor(prefix_lens),
            torch.tensor(extend_lens),
            scaling=self.scale,
            enable_gqa=reqs.num_q_heads != reqs.k_cache.shape[1],
            causal=causal,
            sliding_window_size=window,
        )
        return query, output, extend_lens

    def _check_extend(self, seq_lens, prefix_lens, *, num_kv_heads, window, causal):
        reqs = _Requests(seq_lens, 4, num_kv_heads, self.generator)
        query, output, extend_lens = self._run_extend(
            reqs, prefix_lens, window=window, causal=causal
        )
        start = 0
        for i, (prefix, extend) in enumerate(zip(prefix_lens, extend_lens)):
            k, v = reqs.kv(i)
            expected = _reference_attention(
                query[start : start + extend],
                k,
                v,
                range(prefix, prefix + extend),
                window=window,
                causal=causal,
                scale=self.scale,
            )
            torch.testing.assert_close(
                output[start : start + extend], expected, atol=1e-5, rtol=1e-5
            )
            start += extend
        return reqs, query, output

    def test_extend_bidirectional_window(self):
        for num_kv_heads in (4, 2):
            for window in (0, 2, 3, 20):
                with self.subTest(num_kv_heads=num_kv_heads, window=window):
                    self._check_extend(
                        (5, 9, 1),
                        (0, 0, 0),
                        num_kv_heads=num_kv_heads,
                        window=window,
                        causal=False,
                    )

    def test_extend_bidirectional_window_with_prefix(self):
        self._check_extend((9, 7), (4, 3), num_kv_heads=2, window=2, causal=False)

    def test_extend_bidirectional_window_differs_from_full_attention(self):
        reqs, query, windowed = self._check_extend(
            (12,), (0,), num_kv_heads=4, window=2, causal=False
        )
        full = torch.empty_like(query)
        self.backend._run_sdpa_forward_extend(
            query,
            full,
            reqs.k_cache,
            reqs.v_cache,
            reqs.req_to_token,
            reqs.req_pool_indices,
            torch.tensor([12]),
            torch.tensor([0]),
            torch.tensor([12]),
            scaling=self.scale,
            causal=False,
            sliding_window_size=None,
        )
        self.assertFalse(torch.allclose(windowed, full, atol=1e-3))

    def test_extend_causal_window_unchanged(self):
        for window in (0, 2, 20):
            with self.subTest(window=window):
                self._check_extend(
                    (5, 9), (0, 4), num_kv_heads=2, window=window, causal=True
                )

    def test_decode_window_matches_causal_reference(self):
        # forward_decode passes causal=False. With the query at the last
        # position, the bidirectional window must equal the causal one.
        reqs = _Requests((3, 4, 5, 10), 4, 2, self.generator)
        query = torch.randn(4, 4, HEAD_DIM, generator=self.generator)
        output = torch.empty_like(query)
        self.backend._run_sdpa_forward_decode(
            query,
            output,
            reqs.k_cache,
            reqs.v_cache,
            reqs.req_to_token,
            reqs.req_pool_indices,
            torch.tensor(reqs.seq_lens),
            scaling=self.scale,
            enable_gqa=True,
            causal=False,
            sliding_window_size=3,
        )
        for i, seq_len in enumerate(reqs.seq_lens):
            k, v = reqs.kv(i)
            expected = _reference_attention(
                query[i : i + 1],
                k,
                v,
                [seq_len - 1],
                window=3,
                causal=True,
                scale=self.scale,
            )
            torch.testing.assert_close(
                output[i : i + 1], expected, atol=1e-5, rtol=1e-5
            )


class _KVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache, self.v_cache = k_cache, v_cache

    def get_key_buffer(self, layer_id):
        return self.k_cache

    def get_value_buffer(self, layer_id):
        return self.v_cache


class TestForwardExtendWindowRouting(CustomTestCase):
    """forward_extend must apply the layer's window for encoder-only layers too."""

    def _forward_extend(self, attn_type, window):
        generator = torch.Generator().manual_seed(1)
        reqs = _Requests((10,), 4, 4, generator)
        backend = _make_backend()
        backend.token_to_kv_pool = _KVPool(reqs.k_cache, reqs.v_cache)
        backend.req_to_token_pool = SimpleNamespace(req_to_token=reqs.req_to_token)
        backend.swa_out_cache_loc = None
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=4,
            tp_k_head_num=4,
            qk_head_dim=HEAD_DIM,
            v_head_dim=HEAD_DIM,
            scaling=HEAD_DIM**-0.5,
            is_cross_attention=False,
            attn_type=attn_type,
            sliding_window_size=window,
        )
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.arange(10),
            encoder_out_cache_loc=None,
            req_pool_indices=reqs.req_pool_indices,
            seq_lens=torch.tensor([10]),
            extend_prefix_lens=torch.tensor([0]),
            extend_seq_lens=torch.tensor([10]),
            encoder_lens=None,
        )
        q = torch.randn(10, 4 * HEAD_DIM, generator=generator)
        out = backend.forward_extend(
            q, None, None, layer, forward_batch, save_kv_cache=False
        )
        k, v = reqs.kv(0)
        return q.view(10, 4, HEAD_DIM), out.view(10, 4, HEAD_DIM), k, v

    def test_encoder_only_layer_uses_bidirectional_window(self):
        q, out, k, v = self._forward_extend(AttentionType.ENCODER_ONLY, window=2)
        expected = _reference_attention(
            q, k, v, range(10), window=2, causal=False, scale=HEAD_DIM**-0.5
        )
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_decoder_layer_uses_causal_window(self):
        q, out, k, v = self._forward_extend(AttentionType.DECODER, window=2)
        expected = _reference_attention(
            q, k, v, range(10), window=2, causal=True, scale=HEAD_DIM**-0.5
        )
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_encoder_only_layer_without_window_is_full_attention(self):
        q, out, k, v = self._forward_extend(AttentionType.ENCODER_ONLY, window=-1)
        expected = _reference_attention(
            q, k, v, range(10), window=None, causal=False, scale=HEAD_DIM**-0.5
        )
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
