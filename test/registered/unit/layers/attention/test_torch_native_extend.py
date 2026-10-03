"""CPU numerical coverage for prefix-aware torch-native extend attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import torch_native_backend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTorchNativeExtend(CustomTestCase):
    def _check_attention(
        self,
        *,
        prefix_lens,
        extend_lens,
        causal=True,
        window=None,
        encoder_lens=None,
        cross_attention=False,
        num_kv_heads=2,
        cache_dtype=torch.float32,
        public_forward=False,
    ):
        generator = torch.Generator().manual_seed(42)
        batch_size = len(prefix_lens)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        encoder_lens = encoder_lens or [0] * batch_size
        cache_lens = [s + e for s, e in zip(seq_lens, encoder_lens)]
        num_heads, qk_dim, v_dim = 4, 8, 6
        query = torch.randn(sum(extend_lens), num_heads, qk_dim, generator=generator)
        key = torch.randn(
            sum(cache_lens), num_kv_heads, qk_dim, generator=generator
        ).to(cache_dtype)
        value = torch.randn(
            sum(cache_lens), num_kv_heads, v_dim, generator=generator
        ).to(cache_dtype)
        slots = torch.randperm(sum(cache_lens), generator=generator)
        req_indices = torch.arange(batch_size - 1, -1, -1)
        req_to_token = torch.zeros(batch_size, max(cache_lens), dtype=torch.int64)
        start = 0
        for req_idx, length in zip(req_indices, cache_lens):
            req_to_token[req_idx, :length] = slots[start : start + length]
            start += length

        expected = []
        start_q = 0
        scale = 0.3
        for i, extend_len in enumerate(extend_lens):
            encoder_len = encoder_lens[i]
            kv_start = 0 if cross_attention else encoder_len
            kv_end = encoder_len if cross_attention else encoder_len + seq_lens[i]
            token_ids = req_to_token[req_indices[i], kv_start:kv_end]
            q = query[start_q : start_q + extend_len].transpose(0, 1)
            k = key[token_ids].float().transpose(0, 1)
            v = value[token_ids].float().transpose(0, 1)
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=0)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=0)
            logits = q @ k.transpose(-1, -2) * scale
            if causal:
                # Enumerate visible positions independently of the backend mask.
                visible = torch.tensor(
                    [
                        [
                            k_pos <= prefix_lens[i] + q_pos
                            and (
                                window is None
                                or k_pos >= prefix_lens[i] + q_pos - window
                            )
                            for k_pos in range(len(token_ids))
                        ]
                        for q_pos in range(extend_len)
                    ]
                )
                logits = logits.masked_fill(~visible, -torch.inf)
            expected.append((logits.softmax(dim=-1) @ v).transpose(0, 1))
            start_q += extend_len

        backend = object.__new__(TorchNativeAttnBackend)
        output = torch.empty(sum(extend_lens), num_heads, v_dim)
        batch = SimpleNamespace(
            req_pool_indices=req_indices,
            seq_lens=torch.tensor(seq_lens),
            extend_prefix_lens=torch.tensor(prefix_lens),
            extend_seq_lens=torch.tensor(extend_lens),
            encoder_lens=torch.tensor(encoder_lens),
            out_cache_loc=None,
            encoder_out_cache_loc=None,
        )
        with patch.object(
            torch_native_backend,
            "scaled_dot_product_attention",
            wraps=scaled_dot_product_attention,
        ) as sdpa:
            if public_forward:
                backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
                backend.token_to_kv_pool = SimpleNamespace(
                    get_key_buffer=lambda _: key, get_value_buffer=lambda _: value
                )
                layer = SimpleNamespace(
                    layer_id=0,
                    tp_q_head_num=num_heads,
                    tp_k_head_num=num_kv_heads,
                    qk_head_dim=qk_dim,
                    v_head_dim=v_dim,
                    scaling=scale,
                    is_cross_attention=cross_attention,
                    attn_type=AttentionType.DECODER,
                    sliding_window_size=window,
                )
                output = backend.forward_extend(
                    q=query.flatten(1),
                    k=None,
                    v=None,
                    layer=layer,
                    forward_batch=batch,
                    save_kv_cache=False,
                ).view_as(output)
            else:
                backend._run_sdpa_forward_extend(
                    query=query,
                    output=output,
                    k_cache=key,
                    v_cache=value,
                    req_to_token=req_to_token,
                    req_pool_indices=req_indices,
                    seq_lens=batch.seq_lens,
                    extend_prefix_lens=batch.extend_prefix_lens,
                    extend_seq_lens=batch.extend_seq_lens,
                    encoder_lens=batch.encoder_lens,
                    scaling=scale,
                    enable_gqa=num_heads != num_kv_heads,
                    causal=causal,
                    is_cross_attn=cross_attention,
                    sliding_window_size=window,
                )

        torch.testing.assert_close(output, torch.cat(expected), atol=1e-6, rtol=1e-5)
        # Cached prefix queries must not be recomputed and discarded by SDPA.
        self.assertEqual(
            [call.args[0].shape[-2] for call in sdpa.call_args_list],
            list(extend_lens),
        )

    def test_causal_prefix_and_ragged_batch(self):
        """A rectangular causal mask must align each query after its own prefix."""
        self._check_attention(prefix_lens=(0, 3, 7), extend_lens=(4, 2, 1))

    def test_sliding_window_prefix(self):
        """Window edges use absolute query positions even when the prefix is omitted."""
        for window in (0, 2, 20):
            with self.subTest(window=window):
                self._check_attention(
                    prefix_lens=(0, 5, 8), extend_lens=(3, 4, 1), window=window
                )

    def test_noncausal_prefix(self):
        """Noncausal queries retain access to future keys after removing padding."""
        self._check_attention(prefix_lens=(0, 4), extend_lens=(3, 2), causal=False)

    def test_encoder_decoder_self_attention(self):
        """Decoder self attention must skip encoder cache slots for each request."""
        self._check_attention(
            prefix_lens=(2, 5),
            extend_lens=(3, 1),
            encoder_lens=(7, 4),
            public_forward=True,
        )

    def test_cross_attention(self):
        """Decoder queries attend to encoder keys of an independent sequence length."""
        self._check_attention(
            prefix_lens=(3, 2),
            extend_lens=(2, 4),
            encoder_lens=(7, 3),
            causal=False,
            cross_attention=True,
            public_forward=True,
        )

    def test_cache_dtype_conversion(self):
        """Query and cache dtypes may differ; conversion must preserve the output."""
        self._check_attention(
            prefix_lens=(6,),
            extend_lens=(2,),
            num_kv_heads=4,
            cache_dtype=torch.float16,
        )


if __name__ == "__main__":
    unittest.main()
