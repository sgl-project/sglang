import itertools
import unittest

import sgl_kernel
import torch
from torch.nn.functional import scaled_dot_product_attention
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestMLA(CustomTestCase):
    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        key: torch.Tensor,
        loc: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        # set kv cache
        k_cache[loc] = key

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
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

    def _test_grouped_decode_attention_once(self, B, H_Q, H_KV, D, D_V, seq_len):
        dtype = torch.bfloat16

        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        logit_cap = 0.0
        num_kv_splits = 8
        enable_gqa = H_Q != H_KV

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype)

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype)
        v_buffer = k_buffer.narrow(2, 0, D_V)

        key = torch.randn(B, H_KV, D, dtype=dtype)
        value = key.narrow(2, 0, D_V)
        # make sure no duplicates in loc
        loc = torch.randperm(total_tokens)[:B].to(torch.int64)

        k_buffer2 = k_buffer.clone()
        v_buffer2 = k_buffer2.narrow(2, 0, D_V)

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype)
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype)

        req_to_token = torch.arange(total_tokens).reshape(B, seq_len).to(torch.int32)
        b_req_idx = torch.arange(B).to(torch.int64)
        b_seq_len = torch.full((B,), seq_len).to(torch.int64)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
        )

        torch.ops.sgl_kernel.decode_attention_cpu(
            q,
            k_buffer2,
            v_buffer2,
            o,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_seq_len,
            sm_scale,
            logit_cap,
        )

        self._run_sdpa_forward_decode(
            q,
            o_grouped,
            k_buffer,
            v_buffer,
            key,
            loc,
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        atol = rtol = precision[q.dtype]
        self.assertGreater(cos_sim.item(), 0.99)
        torch.testing.assert_close(o, o_grouped, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_buffer, k_buffer2, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_buffer, v_buffer2, atol=atol, rtol=rtol)

    def test_grouped_decode_attention(self):
        configs = [
            (1, 22, 1, 576, 512, 8 * 111),
            (4, 22, 1, 576, 512, 8 * 128),
            (40, 22, 1, 576, 512, 8 * 133),
        ]

        for B, H_Q, H_KV, D, D_V, seqlen in configs:
            self._test_grouped_decode_attention_once(B, H_Q, H_KV, D, D_V, seqlen)


if __name__ == "__main__":
    unittest.main()
