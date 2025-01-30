import random
import unittest

import torch

from sglang.srt.layers.attention.wave_ops.decode_attention import (
    decode_attention_wave,
)
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped as triton_decode_attention_fwd_grouped,
)


from sglang.srt.layers.attention.wave_ops.prefill_attention import (
    prefill_attention_wave,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)


class TestWaveAttention(unittest.TestCase):

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

    def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
        dtype = torch.float16
        seq_len = S  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        num_kv_splits = 8

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")

        # o will have the same shape as q
        o_triton = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

        req_to_token = torch.arange(
            total_tokens, device="cuda", dtype=torch.int32
        ).reshape(B, seq_len)
        b_req_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        b_seq_len = torch.full((B,), seq_len, device="cuda", dtype=torch.int32)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device="cuda",
        )

        triton_decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o_triton,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
        )

        k_buffer = k_buffer.view(B, seq_len, H_KV, D)
        v_buffer = v_buffer.view(B, seq_len, H_KV, D_V)
        attn_logits = torch.empty(
            (num_kv_splits, B, D_V, H_Q),
            dtype=torch.float32,
            device="cuda",
        )

        attn_logits_max = torch.empty(
            (num_kv_splits, B, H_Q),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_wave(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            attn_logits_max,
            num_kv_splits,
            sm_scale,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_triton.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o, o_triton, atol=3e-2))

    def test_grouped_decode_attention(self):
        # seq_lens = [5, 100, 128, 500]
        seq_lens = [128,]
        configs = [
            # (2, 16, 16, 64, 64),
            # (2, 16, 1, 64, 64), uncomment this
            # (2, 64, 1, 13, 13),
            (2, 128, 1, 80, 80),
            # (2, 128, 2, 512, 512),
            # (2, 128, 1, 576, 512),
        ]

        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        dtype = torch.float16
        num_heads = 4
        seq_lens = [64, 128]
        max_seq_len = max(seq_lens)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda")
        o_triton = torch.zeros(sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda")
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, dtype=torch.float32, device="cuda")

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
        b_seq_len = torch.tensor(seq_lens, device="cuda")

        context_attention_fwd(
            q, k, v, o_triton, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )
        prefill_attention_wave(q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal)
        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_triton.to(torch.float32).flatten(), dim=0
        )

        print(cos_sim.item())
        self.assertTrue(torch.allclose(o, o_triton.to(torch.float32), atol=3e-2))
        self.assertTrue(cos_sim.item() > 1 - (1e-5))

    def test_context_attention(self):
        # head_dim = [128, 96, 80, 13]
        # for is_causal in [False, True]:

        head_dim = [128]

        for dim in head_dim:
            for is_causal in [False]:
                self._test_context_attention_once(dim, is_causal)



if __name__ == "__main__":
    unittest.main()
