import unittest

import sgl_kernel  # noqa: F401
import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")

torch.manual_seed(1234)


class TestDecodeAttention(CustomTestCase):
    def _scaled_dot_product_attention(self, Q, K, V, S, scaling, sliding_window):
        # sliding_window <= 0 means no sliding window
        # Q: [n_tokens_q, n_heads, q_mult, d_head]
        # K: [n_tokens_kv, n_heads, d_head]
        # V: [n_tokens_kv, n_heads, d_head]
        n_tokens_q, n_heads, q_mult, d_head = Q.shape
        n_tokens_kv = K.shape[0]
        assert K.shape == (n_tokens_kv, n_heads, d_head)
        assert V.shape == (n_tokens_kv, n_heads, d_head)
        K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
        V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
        S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens_q, -1)
        if n_tokens_q == n_tokens_kv:  # Prefill
            mask = torch.triu(
                Q.new_full((n_tokens_q, n_tokens_kv), -float("inf")), diagonal=1
            )
        else:  # Decode
            mask = Q.new_zeros((n_tokens_q, n_tokens_kv))
        if sliding_window is not None and sliding_window > 0:
            mask += torch.tril(
                mask.new_full((n_tokens_q, n_tokens_kv), -float("inf")),
                diagonal=n_tokens_kv - n_tokens_q - sliding_window,
            )
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
        QK *= scaling
        QK += mask[None, None, :, :]
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]
        attn = torch.einsum("hmqk,khmd->qhmd", W, V)
        return attn.reshape(n_tokens_q, -1)

    def _run_sdpa_forward_decode_sink(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_kv_heads: int,
        q_mult: int,
        scaling=None,
        sliding_window=None,
        attention_sinks=None,
        enable_gqa=False,
        causal=False,
    ):
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            per_req_query = query[:, start_q:end_q, :]
            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]

            per_req_query = per_req_query.permute(1, 0, 2).reshape(
                seq_len_q, num_kv_heads, q_mult, per_req_query.shape[-1]
            )
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_key = per_req_key.permute(1, 0, 2)
            per_req_value = per_req_value.permute(1, 0, 2)

            per_req_out = self._scaled_dot_product_attention(
                per_req_query,
                per_req_key,
                per_req_value,
                attention_sinks,
                scaling=scaling,
                sliding_window=sliding_window,
            ).reshape(seq_len_q, -1, per_req_value.shape[-1])
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens=None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
    ):
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            if encoder_lens is not None:
                start_kv = 0 if is_cross_attn else encoder_lens[seq_idx]
                end_kv = (
                    encoder_lens[seq_idx] if is_cross_attn else start_kv + seq_len_kv
                )
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def _test_grouped_decode_attention_once(
        self,
        B,
        H_Q,
        H_KV,
        D,
        D_V,
        sliding_window,
        sink,
        is_cross_attn,
        dtype,
        device,
        kvcache_dtype=torch.bfloat16,
    ):
        # This represents the number of tokens already in the sequence
        seq_len = 1024
        encoder_len = 0 if sink else 10
        total_tokens = B * (seq_len + encoder_len)
        sm_scale = 1.0 / (D**0.5)
        logit_cap = 0.0
        num_kv_splits = 8
        enable_gqa = H_Q != H_KV

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)
        sinks = torch.rand(H_Q, dtype=dtype, device=device) * 10

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)
        k_scale = 1.0
        v_scale = 1.0
        if kvcache_dtype == torch.float8_e4m3fn:
            k_scale = 0.5
            v_scale = 0.25
            k_buffer_fp8 = (k_buffer / k_scale).to(torch.float8_e4m3fn)
            v_buffer_fp8 = (v_buffer / v_scale).to(torch.float8_e4m3fn)
            k_buffer = (k_buffer_fp8.float() * k_scale).to(dtype)
            v_buffer = (v_buffer_fp8.float() * v_scale).to(dtype)

        key = torch.randn(B, H_KV, D, dtype=dtype)
        value = torch.randn(B, H_KV, D_V, dtype=dtype)
        loc = torch.randint(0, 10, (B,)).to(torch.int64)

        # set kv cache
        if not is_cross_attn:
            if kvcache_dtype == torch.float8_e4m3fn:
                k_buffer[loc] = (
                    (key / k_scale).to(torch.float8_e4m3fn).float() * k_scale
                ).to(dtype)
                v_buffer[loc] = (
                    (value / v_scale).to(torch.float8_e4m3fn).float() * v_scale
                ).to(dtype)
            else:
                k_buffer[loc] = key
                v_buffer[loc] = value

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        req_to_token = (
            torch.arange(total_tokens, device=device)
            .reshape(B, seq_len + encoder_len)
            .to(torch.int32)
        )
        b_req_idx = torch.arange(B, device=device).to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device=device).to(torch.int64)
        encoder_lens = torch.full((B,), encoder_len, device=device).to(torch.int64)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device=device,
        )

        # k_buffer, v_buffer, query, key and value supports non-contiguous tensors
        k_buffer = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        key = key.transpose(0, 1).contiguous().transpose(0, 1)
        value = value.transpose(0, 1).contiguous().transpose(0, 1)
        torch.ops.sgl_kernel.decode_attention_cpu(
            q,
            (k_buffer if kvcache_dtype != torch.float8_e4m3fn else k_buffer_fp8),
            (v_buffer if kvcache_dtype != torch.float8_e4m3fn else v_buffer_fp8),
            k_scale,
            v_scale,
            o,
            key if not is_cross_attn else None,
            value if not is_cross_attn else None,
            loc,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_seq_len,
            sm_scale,
            logit_cap,
            is_cross_attn,
            sliding_window if sliding_window is not None else 0,
            encoder_lens,
            sinks if sink else None,
        )

        if sink:
            self._run_sdpa_forward_decode_sink(
                q,
                o_grouped,
                k_buffer,
                v_buffer,
                req_to_token,
                b_req_idx,
                b_seq_len,
                num_kv_heads=H_KV,
                q_mult=H_Q // H_KV if enable_gqa else 1,
                scaling=sm_scale,
                sliding_window=sliding_window if sliding_window is not None else None,
                attention_sinks=sinks,
                enable_gqa=enable_gqa,
            )
        else:
            self._run_sdpa_forward_decode(
                q,
                o_grouped,
                k_buffer,
                v_buffer,
                req_to_token,
                b_req_idx,
                b_seq_len,
                scaling=sm_scale,
                enable_gqa=enable_gqa,
                encoder_lens=encoder_lens,
                is_cross_attn=is_cross_attn,
            )
        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.99)
        torch.testing.assert_close(o, o_grouped, atol=3e-2, rtol=1e-6)

    def _test_grouped_decode_attention(self, device="cuda"):
        configs = [
            (2, 16, 16, 64, 64),
            (2, 16, 1, 16, 16),
            (2, 32, 8, 33, 55),
            (2, 16, 1, 64, 64),
            (2, 64, 1, 13, 13),
            (2, 128, 1, 80, 80),
            (2, 128, 2, 512, 512),
            (1, 16, 1, 576, 512),
            (1, 16, 16, 576, 512),
            (1, 22, 1, 576, 512),
            (1, 40, 8, 128, 128),
        ]

        for B, H_Q, H_KV, D, D_V in configs:
            for dtype in [torch.bfloat16, torch.float16]:
                for sink in [True, False]:
                    if D != D_V and sink:
                        continue
                    for sliding_window in [None, 10]:
                        if sliding_window is not None and not sink:
                            continue
                        self._test_grouped_decode_attention_once(
                            B,
                            H_Q,
                            H_KV,
                            D,
                            D_V,
                            sliding_window,
                            sink,
                            False,
                            dtype=dtype,
                            device=device,
                        )
                self._test_grouped_decode_attention_once(
                    B, H_Q, H_KV, D, D_V, None, False, True, dtype=dtype, device=device
                )

        fp8_configs = [
            (2, 32, 8, 33, 55, None, False, False),
            (1, 16, 1, 576, 512, None, False, False),
            (2, 16, 16, 64, 64, 10, True, False),
            (2, 16, 1, 64, 64, None, False, True),
        ]
        for B, H_Q, H_KV, D, D_V, sliding_window, sink, is_cross_attn in fp8_configs:
            self._test_grouped_decode_attention_once(
                B,
                H_Q,
                H_KV,
                D,
                D_V,
                sliding_window,
                sink,
                is_cross_attn,
                dtype=torch.bfloat16,
                device=device,
                kvcache_dtype=torch.float8_e4m3fn,
            )

    def _test_decode_attention_swa_mapping_case(
        self, B, H_Q, H_KV, D, D_V, seq_len, window_size
    ):
        max_tokens_full = 500
        max_tokens_swa = 120
        max_context_len = seq_len + 32

        q = torch.randn(B, H_Q, D, dtype=torch.bfloat16)
        k_buffer_swa = torch.randn(max_tokens_swa, H_KV, D, dtype=torch.bfloat16)
        v_buffer_swa = torch.randn(max_tokens_swa, H_KV, D_V, dtype=torch.bfloat16)

        # Poison slot 0 with NaN: valid swa_id allocations start at >= 1.
        # Any unexpected access to expired or unmapped tokens (which clamp to 0)
        # will read NaN and fail the test.
        k_buffer_swa[0].fill_(float("nan"))
        v_buffer_swa[0].fill_(float("nan"))

        # Deliberately initialize mapping to an out-of-bounds sentinel (99999).
        # Any access to expired tokens outside the window will trigger out-of-bounds
        # if the kernel fails to respect kv_offset.
        full_to_swa_mapping = torch.full((max_tokens_full,), 99999, dtype=torch.int64)

        active_start = max(0, seq_len - window_size)
        req_to_token_virtual = torch.zeros(B, max_context_len, dtype=torch.int64)
        req_to_token_mapped = torch.zeros(B, max_context_len, dtype=torch.int64)

        for b in range(B):
            # Expired tokens outside the sliding window: assign virtual IDs with no valid SWA mapping
            for pos in range(active_start):
                v_id = pos + 1
                req_to_token_virtual[b, pos] = v_id
                req_to_token_mapped[b, pos] = 0

            # Active window tokens: map to valid SWA slots
            for pos in range(active_start, seq_len):
                v_id = 200 + b * window_size + (pos - active_start)
                swa_id = 1 + (b * window_size + (pos - active_start)) % (
                    max_tokens_swa - 1
                )
                full_to_swa_mapping[v_id] = swa_id
                req_to_token_virtual[b, pos] = v_id
                req_to_token_mapped[b, pos] = swa_id

        req_pool_indices = torch.arange(B, dtype=torch.int64)
        seq_lens = torch.full((B,), seq_len, dtype=torch.int64)
        num_kv_splits = 8

        out1 = torch.empty(B, H_Q, D_V, dtype=torch.bfloat16)
        attn_logits1 = torch.empty(B, H_Q, num_kv_splits, D_V + 1, dtype=torch.float32)
        loc = torch.zeros(B, dtype=torch.int64)

        torch.ops.sgl_kernel.decode_attention_cpu(
            q,
            k_buffer_swa,
            v_buffer_swa,
            1.0,
            1.0,
            out1,
            None,
            None,
            loc,
            attn_logits1,
            req_to_token_virtual,
            req_pool_indices,
            seq_lens,
            1.0 / (D**0.5),
            0.0,
            False,
            window_size,
            None,
            None,
            full_to_swa_mapping,
        )

        out2 = torch.empty(B, H_Q, D_V, dtype=torch.bfloat16)
        attn_logits2 = torch.empty(B, H_Q, num_kv_splits, D_V + 1, dtype=torch.float32)

        torch.ops.sgl_kernel.decode_attention_cpu(
            q,
            k_buffer_swa,
            v_buffer_swa,
            1.0,
            1.0,
            out2,
            None,
            None,
            loc,
            attn_logits2,
            req_to_token_mapped,
            req_pool_indices,
            seq_lens,
            1.0 / (D**0.5),
            0.0,
            False,
            window_size,
            None,
            None,
            None,
        )

        self.assertFalse(torch.isnan(out1).any(), "out1 contains NaN!")
        self.assertTrue(torch.allclose(out1, out2, atol=1e-3, rtol=1e-3))

    def test_decode_attention_swa_mapping(self):
        # 1. GQA branch (H_Q != H_KV) with sliding window (seq_len > window_size)
        self._test_decode_attention_swa_mapping_case(
            B=2, H_Q=8, H_KV=2, D=64, D_V=64, seq_len=70, window_size=20
        )
        # 2. MHA branch (H_Q == H_KV) with sliding window
        self._test_decode_attention_swa_mapping_case(
            B=2, H_Q=4, H_KV=4, D=64, D_V=64, seq_len=70, window_size=20
        )
        # 3. Case where seq_len <= window_size (no tokens clipped)
        self._test_decode_attention_swa_mapping_case(
            B=2, H_Q=4, H_KV=2, D=64, D_V=64, seq_len=15, window_size=20
        )

    def test_grouped_decode_attention(self):
        self._test_grouped_decode_attention("cpu")


if __name__ == "__main__":
    unittest.main()
