from __future__ import annotations

"""
Support different attention backends.
Now there are three backends: FlashInfer, Triton and FlashAttention.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache


@dataclass
class DecodeMetadata:
    """Placeholder metadata class for decode operations."""


@dataclass
class PrefillMetadata:
    """Placeholder metadata class for prefill operations."""


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.is_multimodal = model_runner.model_config.is_multimodal

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        max_bs = model_runner.req_to_token_pool.size

        # Initialize metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # QQ NOTE: need to support extend case with kv cache
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use Flash Attention for prefill
        prefix_lens = forward_batch.extend_prefix_lens
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
        seqlens_in_batch = forward_batch.seq_lens
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        # cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        if not extend_no_prefix:
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            k, v, _ = self.prepare_kv_cache(forward_batch, layer)
        else:
            cu_seqlens_q = cu_seqlens_k
        max_seq_len_q = seqlens_in_batch.max().item()
        max_seq_len_k = max_seq_len_q

        o, _ = flash_attn_varlen_func(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k=k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim),
            v=v.contiguous().view(-1, layer.tp_v_head_num, layer.head_dim),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,
            softmax_scale=layer.scaling,
            causal=not layer.is_cross_attention,
            window_size=(
                (layer.sliding_window_size - 1, 0)
                if layer.sliding_window_size is not None
                else (-1, -1)
            ),
            softcap=layer.logit_cap,
            # qv=qv_unpad,
            # q_descale=q_descale,
            k_descale=layer.k_scale,
            v_descale=layer.v_scale,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Get KV cache
        key_cache, value_cache, cache_seqlens = self.prepare_kv_cache(
            forward_batch, layer
        )

        # Use Flash Attention for decode
        seqlens_in_batch = forward_batch.seq_lens
        # cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        # QQ NOTE: we fix to generate 1 token per req currently
        cu_seqlens_q = torch.arange(
            0, len(forward_batch.seq_lens) + 1, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        max_seq_len_q = 1
        max_seq_len_k = seqlens_in_batch.max().item()

        # torch.distributed.breakpoint()
        # breakpoint()
        o, _ = flash_attn_varlen_func(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k=key_cache.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim),
            v=value_cache.contiguous().view(-1, layer.tp_v_head_num, layer.head_dim),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,
            softmax_scale=layer.scaling,
            causal=not layer.is_cross_attention,
            window_size=(
                (layer.sliding_window_size - 1, 0)
                if layer.sliding_window_size is not None
                else (-1, -1)
            ),
            softcap=layer.logit_cap,
            # qv=qv_unpad,
            # q_descale=q_descale,
            k_descale=layer.k_scale,
            v_descale=layer.v_scale,
        )

        # torch.distributed.breakpoint()
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
        # return o.squeeze(1).view(-1, layer.tp_q_head_num * layer.head_dim)

        # o, _ = flash_attn_with_kvcache(
        #     q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1),
        #     k_cache=key_cache,
        #     v_cache=value_cache,
        #     cache_seqlens=cache_seqlens,
        #     softmax_scale=layer.scaling,
        #     causal=True,
        #     window_size=((layer.sliding_window_size - 1, 0) if layer.sliding_window_size is not None else (-1, -1)),
        #     softcap=layer.logit_cap,
        #     return_softmax_lse=True,
        #     num_splits=1,  # kv split for chunked process and speed tuning
        #     # descale_q=descale_q,
        #     k_descale=layer.k_scale,
        #     v_descale=layer.v_scale,
        # )
        # o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def prepare_kv_cache(self, forward_batch: ForwardBatch, layer: RadixAttention):
        # torch.distributed.breakpoint()
        # breakpoint()
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache_pool, value_cache_pool = kv_cache[0], kv_cache[1]

        # QQ NOTE: need a faster way
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        req_tokens = []
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_kv = seq_lens[seq_idx]
            req_pool_idx = req_pool_indices[seq_idx]
            req_tokens.append(req_to_token[req_pool_idx, :seq_len_kv])
        req_tokens = torch.concat(req_tokens)
        key_cache = key_cache_pool[req_tokens]
        value_cache = value_cache_pool[req_tokens]
        cache_seqlens = seq_lens
        return key_cache, value_cache, cache_seqlens

    def init_cuda_graph_state(self, max_bs: int):
        """Initialize CUDA graph state for the attention backend."""
        pass  # Flash Attention Backend currently doesn't support CUDA graph

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        pass  # Flash Attention Backend currently doesn't support CUDA graph

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        num_kv_heads: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        pass  # Flash Attention Backend currently doesn't support CUDA graph

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 0
