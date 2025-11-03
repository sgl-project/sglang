import unittest

import sgl_kernel
import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestExtendAttention(CustomTestCase):
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

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        encoder_lens=None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
    ):

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            if encoder_lens is not None:
                start_kv = 0 if is_cross_attn else encoder_lens[seq_idx]
                end_kv = (
                    encoder_lens[seq_idx] if is_cross_attn else start_kv + seq_len_kv
                )
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_extend_sink(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        num_kv_heads: int,
        q_mult: int,
        scaling=None,
        sliding_window=None,
        attention_sinks=None,
        enable_gqa=False,
        causal=False,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv
            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_query_redudant = per_req_query_redudant.permute(1, 0, 2).reshape(
                seq_len_kv, num_kv_heads, q_mult, per_req_query_redudant.shape[-1]
            )
            per_req_key = per_req_key.permute(1, 0, 2)
            per_req_value = per_req_value.permute(1, 0, 2)
            per_req_out_redudant = self._scaled_dot_product_attention(
                per_req_query_redudant,
                per_req_key,
                per_req_value,
                attention_sinks,
                scaling=scaling,
                sliding_window=sliding_window,
            ).reshape(seq_len_kv, -1, per_req_value.shape[-1])
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _test_extend_attention_once(
        self,
        B,
        N_CTX,
        H_Q,
        H_KV,
        D,
        DV,
        sliding_window=None,
        has_sink=False,
        mla=False,
        is_cross_attn=False,
    ):
        dtype = torch.bfloat16

        b_seq_len_prefix = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
        encoder_lens = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int64)
        if mla:
            b_seq_len_prefix.zero_()
            encoder_lens.zero_()
        if has_sink:
            encoder_lens.zero_()
        b_seq_len_extend = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = (
            torch.max(b_seq_len, 0)[0].item() + torch.max(encoder_lens, 0)[0].item()
        )

        b_req_idx = torch.arange(B, dtype=torch.int32)
        req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32)
        b_start_loc = torch.zeros((B,), dtype=torch.int32)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1] + encoder_lens[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        for i in range(B):
            req_to_tokens[i, : b_seq_len[i] + encoder_lens[i]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len[i] + encoder_lens[i]
            )

        total_token_num = torch.sum(b_seq_len).item() + torch.sum(encoder_lens).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()

        H_BUF = 1 if mla else H_KV
        k_buffer = torch.randn((total_token_num, H_BUF, D), dtype=dtype)
        v_buffer = torch.randn((total_token_num, H_BUF, DV), dtype=dtype)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype)
        v_extend = torch.empty((extend_token_num, H_KV, DV), dtype=dtype)
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype)
        sinks = torch.rand(H_Q, dtype=dtype)

        for i in range(B):
            extend_start_in_buffer = (
                b_start_loc[i] + b_seq_len_prefix[i] + encoder_lens[i]
            )
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i] + encoder_lens[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.randn(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype
            )

        # q_extend, k_extend, v_extend, k_buffer and v_buffer supports non-contiguous tensors
        q_extend = q_extend.transpose(0, 1).contiguous().transpose(0, 1)
        k_extend = k_extend.transpose(0, 1).contiguous().transpose(0, 1)
        v_extend = v_extend.transpose(0, 1).contiguous().transpose(0, 1)
        k_buffer = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        b_start_loc_extend = torch.zeros_like(b_seq_len)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

        sm_scale = 1.0 / (D**0.5)
        logit_cap = 0.0

        # handle index type
        b_req_idx = b_req_idx.to(torch.int64)
        b_seq_len = b_seq_len.to(torch.int64)

        enable_gqa = H_Q != H_KV
        o_ref = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
        if has_sink:
            self._run_sdpa_forward_extend_sink(
                q_extend,
                o_ref,
                k_buffer,
                v_buffer,
                req_to_tokens,
                b_req_idx,
                b_seq_len,
                b_seq_len_prefix,
                b_seq_len_extend,
                H_KV,
                H_Q // H_KV if enable_gqa else 1,
                scaling=sm_scale,
                sliding_window=sliding_window,
                attention_sinks=sinks,
            )
        else:
            self._run_sdpa_forward_extend(
                q_extend,
                o_ref,
                k_buffer,
                v_buffer,
                req_to_tokens,
                b_req_idx,
                b_seq_len,
                b_seq_len_prefix,
                b_seq_len_extend,
                scaling=sm_scale,
                enable_gqa=enable_gqa,
                causal=not is_cross_attn,
                is_cross_attn=is_cross_attn,
                encoder_lens=encoder_lens,
            )

        o_extend = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
            sm_scale,
            logit_cap,
            is_cross_attn,
            sliding_window if sliding_window is not None else 0,
            encoder_lens,
            sinks if has_sink else None,
        )

        torch.testing.assert_close(o_ref, o_extend, atol=1e-2, rtol=1e-2)

    def test_extend_attention(self):
        for has_sink in [True, False]:
            for sliding_window in [None, 10, 128]:
                if not has_sink and sliding_window is not None:
                    continue
                self._test_extend_attention_once(
                    1, 123, 16, 4, 64, 64, sliding_window, has_sink, False, False
                )
                self._test_extend_attention_once(
                    1, 20, 16, 1, 64, 64, sliding_window, has_sink, False, False
                )
                self._test_extend_attention_once(
                    1, 20, 1, 1, 64, 64, sliding_window, has_sink, False, False
                )
        for is_mla in [True, False]:
            for is_cross_attn in [True, False]:
                if is_mla and is_cross_attn:
                    continue
                self._test_extend_attention_once(
                    1, 123, 1, 1, 128, 96, None, False, is_mla, is_cross_attn
                )
                self._test_extend_attention_once(
                    1, 123, 16, 1, 128, 96, None, False, is_mla, is_cross_attn
                )
                self._test_extend_attention_once(
                    4, 1230, 16, 4, 128, 96, None, False, is_mla, is_cross_attn
                )


if __name__ == "__main__":
    unittest.main()
