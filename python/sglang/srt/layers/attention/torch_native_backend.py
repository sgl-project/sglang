from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: List[int],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        scaling=None,
        enable_gqa=False,
        causal=False,
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
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        start_extend = 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len = extend_seq_lens_cpu[seq_idx]
            prefix_seq_len = extend_prefix_lens_cpu[seq_idx]
            is_prefill = prefix_seq_len == 0

            seq_len = seq_lens_cpu[seq_idx]
            end_extend = start_extend + extend_seq_len

            per_req_extend_query = query[:, start_extend:end_extend, :]
            per_req_extend_key = key[:, start_extend:end_extend, :]
            per_req_extend_value = value[:, start_extend:end_extend, :]

            if not is_prefill:
                per_req_query = torch.empty(
                    (
                        per_req_extend_query.shape[0],
                        seq_len,
                        per_req_extend_query.shape[2],
                    ),
                    dtype=per_req_extend_query.dtype,
                    device=per_req_extend_query.device,
                )

                per_req_query[:, prefix_seq_len:, :] = per_req_extend_query

                per_req_key = torch.empty(
                    (per_req_extend_key.shape[0], seq_len, per_req_extend_key.shape[2]),
                    dtype=per_req_extend_key.dtype,
                    device=per_req_extend_key.device,
                )
                per_req_value = torch.empty(
                    (
                        per_req_extend_value.shape[0],
                        seq_len,
                        per_req_extend_value.shape[2],
                    ),
                    dtype=per_req_extend_value.dtype,
                    device=per_req_extend_value.device,
                )

                # get the cached prefix kv
                # get key and value from cache. per_req_tokens contains the kv cache
                # index for each token in the sequence.
                req_pool_idx = req_pool_indices[seq_idx]
                curr_req_to_tokens = torch.index_select(
                    req_to_token, 0, req_pool_idx
                ).squeeze(0)
                per_req_prefix_tokens = curr_req_to_tokens[:prefix_seq_len]
                per_req_prefix_key = k_cache[per_req_prefix_tokens].movedim(
                    0, query.dim() - 2
                )
                per_req_prefix_value = v_cache[per_req_prefix_tokens].movedim(
                    0, query.dim() - 2
                )

                # concat prefix kv and extend kv
                per_req_key[:, prefix_seq_len:, :] = per_req_extend_key
                per_req_value[:, prefix_seq_len:, :] = per_req_extend_value
                per_req_key[:, :prefix_seq_len, :] = per_req_prefix_key
                per_req_value[:, :prefix_seq_len, :] = per_req_prefix_value
            else:
                per_req_query = per_req_extend_query
                per_req_key = per_req_extend_key
                per_req_value = per_req_extend_value

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
            output[start_extend:end_extend, :, :] = per_req_out[prefix_seq_len:, :, :]
            start_extend = end_extend
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
        seq_lens_cpu: List[int],
        scaling=None,
        enable_gqa=False,
        causal=False,
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
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens_cpu[seq_idx]  # get a scalar
            end_q = start_q + seq_len_q  # pure python scalar add
            end_kv = start_kv + seq_len_kv  # tensor add

            per_req_query = query[:, start_q:end_q, :]  # slice

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # select, get a scalar tensor
            req_pool_idx = req_pool_indices[seq_idx]
            curr_req_to_tokens = torch.index_select(
                req_to_token, 0, req_pool_idx
            ).squeeze(0)
            per_req_tokens = curr_req_to_tokens[:seq_len_kv]
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

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_ = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_ = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_extend(
            q_,
            k_,
            v_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens,
            forward_batch.extend_seq_lens_cpu,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=not layer.is_cross_attention,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o
