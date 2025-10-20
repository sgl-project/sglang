import random
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    build_unified_kv_indices,
    extend_attention_fwd,
    extend_attention_fwd_unified,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.test.test_utils import CustomTestCase


def extend_attention_fwd_torch(
    q: torch.Tensor,  # [extend_tokens, H_Q, D]
    k: torch.Tensor,  # [extend_tokens, H_KV, D]
    v: torch.Tensor,  # [extend_tokens, H_KV, D]
    o: torch.Tensor,  # [extend_tokens, H_Q, D]
    k_cache: torch.Tensor,  # [total_tokens, H_KV, D]
    v_cache: torch.Tensor,  # [total_tokens, H_KV, D]
    qo_indptr: torch.Tensor,  # [B+1]
    kv_indptr: torch.Tensor,  # [B+1]
    kv_indices: torch.Tensor,  # [prefix_tokens]
    sliding_window_size: int,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q.shape
    _, H_KV, _ = k.shape

    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    for i in range(B):
        q_start = int(qo_indptr[i].item())
        q_end = int(qo_indptr[i + 1].item())
        kv_start = int(kv_indptr[i].item())
        kv_end = int(kv_indptr[i + 1].item())

        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_cache[prefix_indices]  # [prefix_len, H_KV, D]
        v_prefix = v_cache[prefix_indices]  # [prefix_len, H_KV, D]

        k_extend = k[q_start:q_end]  # [extend_len, H_KV, D]
        v_extend = v[q_start:q_end]  # [extend_len, H_KV, D]
        q_extend = q[q_start:q_end]  # [extend_len, H_Q,  D]

        k_full = torch.cat([k_prefix, k_extend], dim=0)  # [total_len, H_KV, D]
        v_full = torch.cat([v_prefix, v_extend], dim=0)  # [total_len, H_KV, D]

        if group_size != 1:
            k_full_hq = k_full.repeat_interleave(
                group_size, dim=1
            )  # [total_len, H_Q, D]
            v_full_hq = v_full.repeat_interleave(
                group_size, dim=1
            )  # [total_len, H_Q, D]
        else:
            k_full_hq = k_full
            v_full_hq = v_full

        prefix_len = k_prefix.size(0)
        extend_len = k_extend.size(0)
        total_len = prefix_len + extend_len

        # causal
        pos_keys = torch.arange(total_len, device=q.device)
        t = prefix_len + torch.arange(extend_len, device=q.device)  # [extend_len]
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)

        # sliding window
        if sliding_window_size is not None and sliding_window_size > 0:
            start = (t - (sliding_window_size)).clamp_min(0)  # [extend_len]
        else:
            start = torch.zeros_like(t)
        window_mask = pos_keys.unsqueeze(0) >= start.unsqueeze(1)

        final_mask = causal_mask & window_mask

        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_extend, k_full_hq) * scale
        )  # [extend_len, H_Q, total_len]
        attn_scores = attn_scores.masked_fill(~final_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full_hq)


class TestTritonAttention(CustomTestCase):

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
            True,
            mask_indptr,
            max_len_extend,
        )

        b_seq_mask_len = b_seq_len_extend * b_seq_len
        custom_mask = torch.ones(
            (b_seq_mask_len.sum().item(),), dtype=torch.bool, device="cuda"
        )
        mask_indptr = torch.zeros((B + 1,), dtype=torch.int64, device="cuda")
        mask_indptr[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
        for i in range(B):
            causal_mask = (
                torch.tril(
                    torch.ones(b_seq_len_extend[i], b_seq_len_extend[i]), diagonal=0
                )
                == 1
            )
            prefix_mask = torch.ones(
                b_seq_len_extend[i], b_seq_len_prefix[i], dtype=torch.bool
            )
            mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
            custom_mask[mask_indptr[i] : mask_indptr[i + 1]] = mask_flatten

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend_mask,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            True,
            mask_indptr,
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
        self.assertTrue(torch.allclose(o_extend_mask, o_redundant, rtol=1e-2))

    def test_extend_attention(self):

        # Define the varying parameter values
        attention_values = [128, 96, 80, 13]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(19, 12331, 12, 4, value)

    def _test_extend_attention_sliding_window_once(
        self, B, N_CTX, H_Q, H_KV, D, WINDOW_SIZE
    ):
        dtype = torch.bfloat16

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend

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

        o_extend_triton = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )
        o_extend_torch = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend_triton,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max_len_extend,
            sliding_window_size=WINDOW_SIZE,
        )

        extend_attention_fwd_torch(
            q_extend,
            k_extend,
            v_extend,
            o_extend_torch,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            WINDOW_SIZE,
        )

        self.assertTrue(
            torch.allclose(o_extend_triton, o_extend_torch, rtol=1e-3, atol=1e-3)
        )

    def test_extend_attention_sliding_window(self):
        window_sizes = [-1, 127]
        for window_size in window_sizes:
            self._test_extend_attention_sliding_window_once(
                19, 12331, 64, 8, 128, window_size
            )

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
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="cuda")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

        b_seq_len = torch.full((B,), seq_len, device="cuda")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device="cuda")

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
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

    def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
        dtype = torch.bfloat16
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
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

        b_seq_len = torch.full((B,), seq_len, device="cuda")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device="cuda")

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
        )

        attn_logits1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o_grouped,
            kv_indptr,
            kv_indices,
            attn_logits1,
            attn_lse1,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o, o_grouped, atol=3e-2))

    def test_grouped_decode_attention(self):
        seq_lens = [5, 100, 128, 500]
        configs = [
            (2, 16, 16, 64, 64),
            (2, 16, 1, 64, 64),
            (2, 64, 1, 13, 13),
            (2, 128, 1, 80, 80),
            (2, 128, 2, 512, 512),
            (2, 128, 1, 576, 512),
        ]

        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)

    def _test_extend_attention_unified_vs_regular_once(self, B, N_CTX, H_Q, H_KV, D):
        """Test that unified kernel produces same results as 2-stage kernel."""
        dtype = torch.bfloat16

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend

        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        # Setup prefix KV indices
        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int64, device="cuda"
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

        # Setup for extend attention
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        # Run 2-stage kernel
        o_regular = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_regular,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max_len_extend,
        )

        # Build unified KV indices
        extend_kv_indices = torch.arange(
            total_token_num - extend_token_num,
            total_token_num,
            dtype=torch.int64,
            device="cuda",
        )
        extend_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        extend_start_loc[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        unified_kv_indptr, unified_kv_indices, prefix_lens = build_unified_kv_indices(
            kv_indptr,
            kv_indices,
            extend_start_loc,
            b_seq_len_extend,
            extend_kv_indices,
            B,
        )

        # Run unified kernel
        o_unified = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        extend_attention_fwd_unified(
            q_extend,
            o_unified,
            k_buffer,
            v_buffer,
            qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            max_len_extend=max_len_extend,
            custom_mask=None,
            mask_indptr=None,
            sm_scale=None,
            logit_cap=0.0,
            is_causal=True,
        )

        # Compare results
        self.assertTrue(
            torch.allclose(o_regular, o_unified, rtol=0.15, atol=0.15),
            f"Unified kernel output differs from 2-stage kernel. "
            f"Max diff: {(o_regular - o_unified).abs().max()}",
        )

    def test_extend_attention_unified_vs_regular(self):
        """Test unified kernel matches 2-stage kernel across different configs."""
        configs = [
            (4, 512, 32, 8, 128),  # Standard config
            (2, 2048, 32, 8, 128),  # Long sequence (test 2048 specifically)
            (8, 256, 64, 8, 80),  # Non-standard head dim
        ]

        for B, N_CTX, H_Q, H_KV, D in configs:
            with self.subTest(B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D):
                self._test_extend_attention_unified_vs_regular_once(
                    B, N_CTX, H_Q, H_KV, D
                )

    def test_build_unified_kv_indices(self):
        """Test build_unified_kv_indices correctness."""
        B = 4
        dtype = torch.int64
        device = "cuda"

        # Setup test data
        prefix_lens = torch.tensor([10, 20, 15, 25], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([5, 3, 7, 4], dtype=torch.int32, device=device)

        # Build prefix indices
        prefix_kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        prefix_kv_indptr[1:] = torch.cumsum(prefix_lens, dim=0)
        prefix_kv_indices = torch.arange(
            prefix_lens.sum().item(), dtype=dtype, device=device
        )

        # Build extend indices
        extend_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        extend_start_loc[1:] = torch.cumsum(extend_lens[:-1], dim=0)
        extend_kv_indices = torch.arange(
            prefix_lens.sum().item(),
            prefix_lens.sum().item() + extend_lens.sum().item(),
            dtype=dtype,
            device=device,
        )

        # Build unified indices
        unified_kv_indptr, unified_kv_indices, returned_prefix_lens = (
            build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_lens,
                extend_kv_indices,
                B,
            )
        )

        # Verify unified_kv_indptr
        expected_lens = prefix_lens + extend_lens
        expected_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        expected_indptr[1:] = torch.cumsum(expected_lens, dim=0)
        self.assertTrue(torch.equal(unified_kv_indptr, expected_indptr))

        # Verify prefix_lens
        self.assertTrue(torch.equal(returned_prefix_lens, prefix_lens))

        # Verify unified_kv_indices structure
        for i in range(B):
            start_idx = int(unified_kv_indptr[i])
            end_idx = int(unified_kv_indptr[i + 1])
            prefix_len = int(prefix_lens[i])
            extend_len = int(extend_lens[i])

            # Check that prefix and extend are concatenated correctly
            unified_seq = unified_kv_indices[start_idx:end_idx]
            self.assertEqual(len(unified_seq), prefix_len + extend_len)


if __name__ == "__main__":
    unittest.main()
