import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)


class TestTritonAttention(unittest.TestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):
        dtype = torch.bfloat16

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
        req_to_tokens = torch.empty(
            (B, max_len_in_batch), dtype=torch.int32, device="cuda"
        )
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        for i in range(B):
            req_to_tokens[i, : b_seq_len[i]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
            ).normal_(mean=0.1, std=0.2)

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        b_start_loc_extend = torch.zeros_like(b_seq_len)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        extend_attention_fwd(
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
        )

        redundant_attention(
            q_extend,
            o_redundant,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_seq_len_prefix,
            max_len_in_batch,
        )

        self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2))

    def test_extend_attention(self):

        # Define the varying parameter values
        attention_values = [128, 96, 80, 13]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(19, 12331, 12, 4, value)

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        num_heads = 4
        seq_lens = [8, 12]
        max_seq_len = max(seq_lens)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
        k = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
        v = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, device="cuda")

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
        b_seq_len = torch.tensor(seq_lens, device="cuda")

        context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )

        cu_seq_lens = [0] * (len(seq_lens) + 1)
        for i, seq_len in enumerate(seq_lens):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

        for i in range(len(seq_lens)):
            start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
            o_torch = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].permute(1, 0, 2),
                k[start:end].permute(1, 0, 2),
                v[start:end].permute(1, 0, 2),
                is_causal=is_causal,
            ).permute(1, 0, 2)

            cos_sim = torch.nn.functional.cosine_similarity(
                o[start:end].flatten(), o_torch.flatten(), dim=0
            )
            self.assertTrue(cos_sim.item() > 1 - (1e-5))
            self.assertTrue(torch.allclose(o[start:end], o_torch, atol=1e-2))

    def test_context_attention(self):
        head_dim = [128, 96, 80, 13]

        for dim in head_dim:
            for is_causal in [True, False]:
                self._test_context_attention_once(dim, is_causal)

    def _test_decode_attention_once(self, B, H_Q, H_KV, D):
        dtype = torch.bfloat16
        seq_len = 10  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

        req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
        b_req_idx = torch.arange(B, device="cuda")
        b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
        b_seq_len = torch.full((B,), seq_len, device="cuda")

        attn_logits = torch.empty(
            (H_Q, total_tokens),
            dtype=dtype,
            device="cuda",
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            attn_logits,
            seq_len,
            sm_scale,
        )

    def test_decode_attention(self):
        # Here we just to ensure there is no error
        # TODO: correctnesss test

        # Test configurations
        configs = [
            (2, 4, 4, 64),  # MHA
            (2, 4, 2, 64),  # GQA
            (2, 4, 4, 80),  # Non-standard head dim
            (2, 4, 4, 13),  # Prime number head dim
        ]

        for B, H_Q, H_KV, D in configs:
            self._test_decode_attention_once(B, H_Q, H_KV, D)

    def _test_grouped_decode_attention_once(self, B, H_Q, H_KV, D, D_V):
        dtype = torch.bfloat16
        seq_len = 10  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
        o_grouped = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

        req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
        b_req_idx = torch.arange(B, device="cuda")
        b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
        b_seq_len = torch.full((B,), seq_len, device="cuda")

        attn_logits = torch.empty(
            (H_Q, total_tokens),
            dtype=dtype,
            device="cuda",
        )

        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            attn_logits,
            seq_len,
            sm_scale,
        )

        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o_grouped,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            attn_logits,
            seq_len,
            sm_scale,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o, o_grouped, atol=3e-2))

    def test_grouped_decode_attention(self):
        configs = [
            (2, 16, 1, 64, 64),
            (2, 64, 1, 13, 13),
            (2, 128, 1, 80, 80),
            (2, 128, 2, 512, 512),
            (2, 128, 1, 576, 512),
        ]

        for B, H_Q, H_KV, D, D_V in configs:
            self._test_grouped_decode_attention_once(B, H_Q, H_KV, D, D_V)


if __name__ == "__main__":
    unittest.main()
