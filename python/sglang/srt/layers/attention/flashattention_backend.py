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
class FlashAttentionMetadata:
    """Metadata for decode operations to avoid redundant computations."""

    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    max_seq_len_k: int = 0
    window_size: tuple = (-1, -1)
    page_table: torch.Tensor = None
    cache_seqlens_int32: torch.Tensor = None
    max_seq_len_q: int = 0


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        # print("FlashAttentionBackend init Biao!!!")
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
        self.forward_metadata: FlashAttentionMetadata = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata to cache repetitive calculations."""
        # Create metadata based on forward mode
        metadata = FlashAttentionMetadata()
        extend_seq_lens = forward_batch.extend_seq_lens
        # Get sequence information
        seqlens_in_batch = forward_batch.seq_lens
        # Precompute int32 version of sequence lengths
        metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
        batch_size = len(seqlens_in_batch)
        device = seqlens_in_batch.device
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        # Precompute maximum sequence length
        metadata.max_seq_len_k = seqlens_in_batch.max().item()
        # Precompute page table
        metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, : metadata.max_seq_len_k
        ]
        if forward_batch.forward_mode == ForwardMode.DECODE:
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
        else:
            extend_no_prefix = not any(forward_batch.extend_prefix_lens)
            # Precompute cumulative sequence lengths
            if not extend_no_prefix:
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
            metadata.max_seq_len_q = seqlens_in_batch.max().item()
        self.forward_metadata = metadata

    def forward_extend(
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

        # Initialize metadata if needed
        if self.forward_metadata is None or not isinstance(
            self.forward_metadata, FlashAttentionMetadata
        ):
            self.init_forward_metadata(forward_batch)

        # Use precomputed metadata
        metadata = self.forward_metadata

        # # Use Flash Attention for prefill
        # Calculate window size (can be moved to metadata if layer properties don't change)
        window_size = (
            (layer.sliding_window_size - 1, 0)
            if layer.sliding_window_size is not None
            else (-1, -1)
        )
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        o = flash_attn_with_kvcache(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache=key_cache.unsqueeze(1),
            v_cache=value_cache.unsqueeze(1),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=window_size,
            softcap=layer.logit_cap,
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
    ) -> torch.Tensor:
        """Forward pass with FlashAttention using precomputed metadata."""
        # import time
        # start_time = time.time()

        # Save KV cache if needed
        if k is not None and v is not None and save_kv_cache:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        # Get KV cache
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache, value_cache = kv_cache[0], kv_cache[1]

        # Initialize metadata if needed
        if self.forward_metadata is None or not isinstance(
            self.forward_metadata, FlashAttentionMetadata
        ):
            self.init_forward_metadata(forward_batch)

        # Use precomputed metadata
        metadata = self.forward_metadata

        # Pre-reshape query tensor
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        # Calculate window size (can be moved to metadata if layer properties don't change)
        window_size = (
            (layer.sliding_window_size - 1, 0)
            if layer.sliding_window_size is not None
            else (-1, -1)
        )

        # end_time = time.time()
        # print(f"Time taken to prepare tensors: {end_time - start_time} seconds")
        # start_time = time.time()

        # Run attention with precomputed values
        o = flash_attn_with_kvcache(
            q=q_reshaped,
            k_cache=key_cache.unsqueeze(1),
            v_cache=value_cache.unsqueeze(1),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=layer.k_scale,
            v_descale=layer.v_scale,
        )

        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Time taken to run flash_attn_with_kvcache: {end_time - start_time} seconds")

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

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
