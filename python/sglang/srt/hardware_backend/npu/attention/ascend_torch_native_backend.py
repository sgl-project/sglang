from __future__ import annotations

import math
from typing import Optional

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
        sliding_window_size: int = -1,
        full_to_swa_mapping: Optional[torch.Tensor] = None,
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
            sliding_window_size: int, -1 means no sliding window
            full_to_swa_mapping: mapping from full pool index to SWA pool index,
                required for SWA layers to translate req_to_token indices

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

            extend_seq_len_q = int(extend_seq_lens[seq_idx].item())
            prefill_seq_len_q = int(extend_prefix_lens[seq_idx].item())

            seq_len_kv = int(seq_lens[seq_idx].item())
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv
            atten_start_kv = 0
            atten_end_kv = seq_len_kv
            # support cross attention
            if encoder_lens is not None:
                if is_cross_attention:
                    atten_end_kv = int(encoder_lens[seq_idx].item())
                else:
                    atten_start_kv = int(encoder_lens[seq_idx].item())
                    atten_end_kv = atten_start_kv + extend_seq_len_q

            is_swa_self_attn = (
                sliding_window_size is not None
                and sliding_window_size > -1
                and encoder_lens is None
            )
            if is_swa_self_attn:
                # For extend, the sliding window must be anchored at the first
                # query token in this chunk rather than the final sequence
                # length. Otherwise a large extend chunk can no longer fit in
                # the cropped query suffix, which breaks the native fallback.
                atten_start_kv = max(
                    prefill_seq_len_q - sliding_window_size, atten_start_kv
                )

            per_req_query = query[:, start_q:end_q, :]

            # SWA crops the front of the KV window, so the redundant query
            # tensor must match the cropped window to keep Q.len == K.len for
            # the causal mask. In cross-attention (and non-SWA self-attention)
            # the original sizing — text seq len — must be preserved, since Q
            # (text) and KV (encoder) lengths legitimately differ there.
            if is_swa_self_attn:
                redundant_len = atten_end_kv - atten_start_kv
                query_start_idx = max(prefill_seq_len_q - atten_start_kv, 0)
            else:
                redundant_len = int(seq_lens[seq_idx].item())
                query_start_idx = prefill_seq_len_q

            per_req_query_redundant = torch.zeros(
                (per_req_query.shape[0], redundant_len, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redundant[:, query_start_idx:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            # For SWA layers, k_cache/v_cache are from the SWA pool but
            # req_to_token stores full pool indices. Translate before indexing.
            if full_to_swa_mapping is not None:
                per_req_tokens = full_to_swa_mapping[per_req_tokens]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            if logit_cap > 0:
                per_req_out_redundant = (
                    self.scaled_dot_product_attention_with_softcapping(
                        per_req_query_redundant.unsqueeze(0),
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
                per_req_out_redundant = (
                    scaled_dot_product_attention(
                        per_req_query_redundant.unsqueeze(0),
                        per_req_key.unsqueeze(0),
                        per_req_value.unsqueeze(0),
                        enable_gqa=enable_gqa,
                        scale=scaling,
                        is_causal=causal,
                    )
                    .squeeze(0)
                    .movedim(query.dim() - 2, 0)
                )
            output[start_q:end_q, :, :] = per_req_out_redundant[
                query_start_idx : query_start_idx + extend_seq_len_q, :, :
            ]
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
        sliding_window_size: int = -1,
        full_to_swa_mapping: Optional[torch.Tensor] = None,
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
            seq_len_kv = int(seq_lens[seq_idx].item())
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            atten_start_kv = 0
            atten_end_kv = seq_len_kv
            # support cross attention
            if encoder_lens is not None:
                if is_cross_attention:
                    atten_end_kv = int(encoder_lens[seq_idx].item())
                else:
                    atten_start_kv = int(encoder_lens[seq_idx].item())
                    atten_end_kv = atten_start_kv + seq_len_kv

            if (
                sliding_window_size is not None
                and sliding_window_size > -1
                and encoder_lens is None
            ):
                atten_start_kv = max(
                    atten_end_kv - (sliding_window_size + 1), atten_start_kv
                )

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            # For SWA layers, k_cache/v_cache are from the SWA pool but
            # req_to_token stores full pool indices. Translate before indexing.
            if full_to_swa_mapping is not None:
                per_req_tokens = full_to_swa_mapping[per_req_tokens]
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
