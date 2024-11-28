from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention import AttentionBackend
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

    def init_cuda_graph_state(self, max_bs: int):
        # TODO: Support CUDA graph
        raise ValueError(
            "Torch native attention does not support CUDA graph for now. Please --disable-cuda-graph"
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
    ):
        # TODO: Support CUDA graph
        raise ValueError(
            "Torch native attention does not support CUDA graph for now. Please --disable-cuda-graph"
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor] = None,
    ):
        # TODO: Support CUDA graph
        raise ValueError(
            "Torch native attention does not support CUDA graph for now. Please --disable-cuda-graph"
        )

    def get_cuda_graph_seq_len_fill_value(self):
        # TODO: Support CUDA graph
        raise ValueError(
            "Torch native attention does not support CUDA graph for now. Please --disable-cuda-graph"
        )

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

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = extend_seq_lens[seq_idx]
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
                    enable_gqa=True,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv
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
                    enable_gqa=True,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
        )
        return o

    def forward_decode(
        self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

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
        )

        return o
