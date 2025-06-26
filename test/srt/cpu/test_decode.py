import unittest

import sgl_kernel
import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestDecodeAttention(CustomTestCase):
    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
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

    def _test_grouped_decode_attention_once(self, B, H_Q, H_KV, D, D_V, device):
        dtype = torch.bfloat16
        # This represents the number of tokens already in the sequence
        seq_len = 1024
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        logit_cap = 0.0
        num_kv_splits = 8
        enable_gqa = H_Q != H_KV

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)

        key = torch.randn(B, H_KV, D, dtype=dtype)
        value = torch.randn(B, H_KV, D_V, dtype=dtype)
        loc = torch.randint(0, 10, (B,)).to(torch.int64)

        # set kv cache
        k_buffer[loc] = key
        v_buffer[loc] = value

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        req_to_token = (
            torch.arange(total_tokens, device=device)
            .reshape(B, seq_len)
            .to(torch.int32)
        )
        b_req_idx = torch.arange(B, device=device).to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device=device).to(torch.int64)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device=device,
        )

        # k_buffer, v_buffer, key and value supports non-contiguous tensors
        k_buffer = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        key = key.transpose(0, 1).contiguous().transpose(0, 1)
        value = value.transpose(0, 1).contiguous().transpose(0, 1)
        torch.ops.sgl_kernel.decode_attention_cpu(
            q,
            k_buffer,
            v_buffer,
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
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
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
            self._test_grouped_decode_attention_once(
                B, H_Q, H_KV, D, D_V, device=device
            )

    def test_grouped_decode_attention(self):
        self._test_grouped_decode_attention("cpu")


if __name__ == "__main__":
    unittest.main()
