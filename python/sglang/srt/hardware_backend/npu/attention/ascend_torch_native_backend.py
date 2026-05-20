from __future__ import annotations

import math

import torch
from torch.nn.functional import scaled_dot_product_attention


class AscendTorchNativeAttnBackend:
    def __init__(self):
        pass

    def scaled_dot_product_attention_with_softcapping(
        self,
        query,
        key,
        value,
        attn_mask=None,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        logit_cap=0.0,
        logit_capping_method="tanh",
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if logit_cap > 0:
            if logit_capping_method == "tanh":
                attn_weight = logit_cap * torch.tanh(attn_weight / logit_cap)

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def run_sdpa_forward_extend(
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
        encoder_lens: torch.Tensor = None,
        is_cross_attention: bool = False,
        scaling=None,
        enable_gqa=False,
        causal=False,
        logit_cap: float = 0.0,
        logit_capping_method: str = "tanh",
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            encoder_lens: [num_seqs]
            is_cross_attention: [bool]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv
            atten_start_kv = 0
            atten_end_kv = seq_lens[seq_idx]
            # support cross attention
            if encoder_lens is not None:
                if is_cross_attention:
                    atten_end_kv = encoder_lens[seq_idx]
                else:
                    atten_start_kv = encoder_lens[seq_idx]
                    atten_end_kv = encoder_lens[seq_idx] + extend_seq_len_q

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
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            if logit_cap > 0:
                per_req_out_redudant = (
                    self.scaled_dot_product_attention_with_softcapping(
                        per_req_query_redudant.unsqueeze(0),
                        per_req_key.unsqueeze(0),
                        per_req_value.unsqueeze(0),
                        enable_gqa=enable_gqa,
                        scale=scaling,
                        is_causal=causal,
                        logit_cap=logit_cap,
                        logit_capping_method=logit_capping_method,
                    )
                    .squeeze(0)
                    .movedim(query.dim() - 2, 0)
                )
            else:
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

    def run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor = None,
        is_cross_attention: bool = False,
        scaling=None,
        enable_gqa=False,
        causal=False,
        logit_cap: float = 0.0,
        logit_capping_method: str = "tanh",
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            encoder_lens: [num_seqs]
            is_cross_attention: [bool]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            atten_start_kv = 0
            atten_end_kv = seq_lens[seq_idx]
            # support cross attention
            if encoder_lens is not None:
                if is_cross_attention:
                    atten_end_kv = encoder_lens[seq_idx]
                else:
                    atten_start_kv = encoder_lens[seq_idx]
                    atten_end_kv = encoder_lens[seq_idx] + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            if logit_cap > 0:
                per_req_out = (
                    self.scaled_dot_product_attention_with_softcapping(
                        per_req_query.unsqueeze(0),
                        per_req_key.unsqueeze(0),
                        per_req_value.unsqueeze(0),
                        enable_gqa=enable_gqa,
                        scale=scaling,
                        is_causal=causal,
                        logit_cap=logit_cap,
                        logit_capping_method=logit_capping_method,
                    )
                    .squeeze(0)
                    .movedim(query.dim() - 2, 0)
                )
            else:
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

    def support_triton(self):
        return False
