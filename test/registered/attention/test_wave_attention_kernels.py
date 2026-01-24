import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped as triton_decode_attention_fwd_grouped,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.attention.wave_ops.decode_attention import (
    decode_attention_intermediate_arrays_shapes,
    decode_attention_wave,
)
from sglang.srt.layers.attention.wave_ops.extend_attention import extend_attention_wave
from sglang.srt.layers.attention.wave_ops.prefill_attention import (
    prefill_attention_wave,
)
from sglang.test.ci.ci_register import register_amd_ci

# Wave attention kernel unit tests (AMD only - requires wave_lang)
register_amd_ci(est_time=60, suite="stage-a-test-1-amd")


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

    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):
        dtype = torch.float16
        extend_seq_len = 1024

        b_seq_len_prefix = torch.full(
            (B,), N_CTX // B, dtype=torch.int32, device="cuda"
        )
        b_seq_len_extend = torch.full(
            (B,), extend_seq_len, dtype=torch.int32, device="cuda"
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int32, device="cuda"
        )

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
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
        o_extend_mask = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        custom_mask = None
        mask_indptr = None

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

        is_causal = True

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            max_len_extend,
        )

        o_wave = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        extend_attention_wave(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_indptr,
            max_len_extend,
            o_wave,
            is_causal=is_causal,
        )

        self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2))
        self.assertTrue(torch.allclose(o_wave, o_redundant, rtol=1e-2))

    def test_extend_attention(self):

        # Define the varying parameter values
        attention_values = [128]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(32, 16384, 6, 1, value)

    def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
        dtype = torch.float16
        seq_len = S  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="cuda")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")

        # o will have the same shape as q
        o_triton = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

        req_to_token = torch.arange(total_tokens, device="cuda", dtype=torch.int32)
        b_req_idx = torch.zeros(B + 1, device="cuda", dtype=torch.int32)
        b_seq_len = torch.full((B,), seq_len, device="cuda", dtype=torch.int32)
        b_req_idx[1 : B + 1] = torch.cumsum(b_seq_len, dim=0)

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D_V + 1),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )

        logit_cap = 0.0
        triton_decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o_triton,
            b_req_idx,
            req_to_token,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
        )

        attn_logits_shape, attn_logits_max_shape = (
            decode_attention_intermediate_arrays_shapes(B, D_V, H_Q, max_kv_splits)
        )

        attn_logits = torch.empty(
            attn_logits_shape,
            dtype=torch.float32,
            device="cuda",
        )

        attn_logits_max = torch.empty(
            attn_logits_max_shape,
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_wave(
            q,
            k_buffer,
            v_buffer,
            o,
            b_req_idx,
            req_to_token,
            attn_logits,
            attn_logits_max,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_triton.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o, o_triton, atol=3e-2))

    def test_grouped_decode_attention(self):
        seq_lens = [5, 100, 128, 500]
        configs = [
            (2, 16, 16, 64, 64),
            (2, 16, 1, 64, 64),
            (2, 128, 1, 80, 80),
            (32, 128, 2, 512, 512),
            (2, 128, 2, 512, 512),
            (2, 128, 1, 576, 512),
        ]

        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        dtype = torch.float16
        num_heads = 4
        kv_heads = 1
        seq_lens = [128, 256]
        max_seq_len = max(seq_lens)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(sum(seq_lens), kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(sum(seq_lens), kv_heads, head_dim, dtype=dtype, device="cuda")
        o_triton = torch.zeros(
            sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda"
        )
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, dtype=dtype, device="cuda")

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
        b_seq_len = torch.tensor(seq_lens, device="cuda")

        context_attention_fwd(
            q, k, v, o_triton, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )
        prefill_attention_wave(
            q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_triton.flatten(), dim=0
        )

        print(cos_sim.item())
        self.assertTrue(torch.allclose(o, o_triton, atol=3e-2))
        self.assertTrue(cos_sim.item() > 1 - (1e-5))

    def test_context_attention(self):
        head_dim = [128, 96]

        for dim in head_dim:
            for is_causal in [False]:
                self._test_context_attention_once(dim, is_causal)


if __name__ == "__main__":
    unittest.main()
