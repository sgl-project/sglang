from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.python.sglang.srt.model_executor.model_runner import ModelRunner

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

from flash_attn_interface import flash_attn_varlen_func


class FlashAttentionBackend(ABC):
    """The FA3 attention backend"""

    def __init__(
        self,
        model_runner: ModelRunner,
    ):
        super().__init__()

        self.forward_metadata = None
        self.device = model_runner.device

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init the metadata for a forward pass for replying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        raise NotImplementedError()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Run forward for decode mode using FlashAttention.

        Args:
            q: Query tensor [num_tokens, num_heads, head_size]
            k: Key tensor [num_tokens, num_kv_heads, head_size]
            v: Value tensor [num_tokens, num_kv_heads, head_size]
            layer: RadixAttention layer
            forward_batch: Batch information for the forward pass
            save_kv_cache: Whether to save the KV cache

        Returns:
            Output tensor [num_tokens, num_heads * head_size]
        """
        # Save new KV to cache if requested
        if save_kv_cache:
            # Update the key/value cache for the new tokens
            layer.key_cache[forward_batch.slot_mapping] = k
            layer.value_cache[forward_batch.slot_mapping] = v

        # Run flash attention with variable length sequences
        output = flash_attn_varlen_func(
            q=q,
            k=layer.key_cache,
            v=layer.value_cache,
            cu_seqlens_q=forward_batch.query_start_loc,
            max_seqlen_q=forward_batch.max_query_len,
            seqused_k=forward_batch.seq_lens,
            max_seqlen_k=forward_batch.max_seq_len,
            softmax_scale=layer.scale,
            causal=True,
            # Optional parameters
            alibi_slopes=layer.alibi_slopes if hasattr(layer, "alibi_slopes") else None,
            window_size=(-1, -1),  # No sliding window by default
            block_table=forward_batch.block_table,
        )

        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Run a forward for extend mode using FlashAttention.

        Args:
            q: Query tensor [num_tokens, num_heads, head_size]
            k: Key tensor [num_tokens, num_kv_heads, head_size]
            v: Value tensor [num_tokens, num_kv_heads, head_size]
            layer: RadixAttention layer
            forward_batch: Batch information for the forward pass
            save_kv_cache: Whether to save the KV cache

        Returns:
            Output tensor [num_tokens, num_heads * head_size]
        """
        # Get batch metadata
        num_tokens = q.shape[0]

        # Run flash attention with variable length sequences
        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=forward_batch.query_start_loc,
            max_seqlen_q=forward_batch.max_query_len,
            seqused_k=forward_batch.seq_lens,
            max_seqlen_k=forward_batch.max_seq_len,
            softmax_scale=layer.scale,
            causal=True,
            # Optional parameters
            alibi_slopes=layer.alibi_slopes if hasattr(layer, "alibi_slopes") else None,
            window_size=(-1, -1),  # No sliding window by default
        )

        # Save KV cache if requested
        if (
            save_kv_cache
            and hasattr(layer, "key_cache")
            and hasattr(layer, "value_cache")
        ):
            # Update the key/value cache
            # Note: This assumes the cache tensors are already properly initialized
            layer.key_cache[forward_batch.slot_mapping] = k
            layer.value_cache[forward_batch.slot_mapping] = v

        return output
