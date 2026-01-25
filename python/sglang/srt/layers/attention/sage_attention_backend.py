from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sageattention import sageattn_varlen

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


@dataclass
class SageAttentionMetadata:
    # For extend (prefill) operations
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Whether this is a causal attention
    is_causal: bool = True


class SageAttnBackend(AttentionBackend):
    """
    SageAttention backend for SGLang.

    This backend uses SageAttention's 8-bit quantized attention for improved
    throughput. SageAttention quantizes Q and K to INT8 on-the-fly during
    computation, providing speedup while maintaining accuracy.

    Both extend (prefill) and decode operations use SageAttention. For decode,
    we gather the KV cache into contiguous memory to use SageAttention's API.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.model_runner = model_runner
        self.device = model_runner.device

        # Create a Triton backend for CUDA graph support and fallback
        # Also used when prefix caching is active (SageAttention's causal mask
        # only works when qo_len == kv_len)
        self.triton_backend = TritonAttnBackend(model_runner)

        # Store reference to KV pool for gathering KV cache
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Forward metadata
        self.forward_metadata: Optional[SageAttentionMetadata] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize metadata for a forward pass."""
        # Always initialize triton backend metadata (for CUDA graph support)
        self.triton_backend.init_forward_metadata(forward_batch)

        if forward_batch.forward_mode.is_decode_or_idle():
            # For decode, prepare SageAttention metadata
            metadata = SageAttentionMetadata()

            # Decode processes 1 token per sequence
            bs = forward_batch.batch_size

            # Q: 1 token per sequence
            metadata.max_seqlen_q = 1
            metadata.cu_seqlens_q = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )

            # K, V: full sequence length (all cached tokens)
            # Use seq_lens_cpu for max (scalar), seq_lens (GPU) for cumsum
            metadata.max_seqlen_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )

            metadata.is_causal = True

            self.forward_metadata = metadata
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            # For extend (prefill), prepare SageAttention metadata
            metadata = SageAttentionMetadata()

            # Build cumulative sequence lengths for variable-length attention
            if forward_batch.extend_seq_lens is not None:
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seqlen_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seqlen_q = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32),
                    (1, 0),
                )

            # K, V sequence lengths (full sequence including prefix)
            metadata.max_seqlen_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )

            metadata.is_causal = True

            self.forward_metadata = metadata
        else:
            self.forward_metadata = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        # Delegate to triton backend for CUDA graph support
        self.triton_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Init the metadata for capturing a cuda graph."""
        # Delegate to triton backend
        self.triton_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init the metadata for replaying a cuda graph."""
        # Delegate to triton backend
        self.triton_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens."""
        return self.triton_backend.get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """
        Forward pass for decode using SageAttention.

        This uses SageAttention's 8-bit quantized attention for decode operations.
        We gather the KV cache into contiguous memory to use SageAttention's API.

        Note: We use is_causal=False for decode because SageAttention's causal mask
        only works when qo_len == kv_len. For decode, Q has 1 token and K has seq_len
        tokens, so we must disable the causal mask. This is correct because all K tokens
        are past tokens that should be attended to.
        """
        if self.forward_metadata is None:
            return self.triton_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        # Save KV cache first
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # Get full K, V tensors by gathering from KV cache
        metadata = self.forward_metadata

        # For decode, we need to gather the full KV cache (all tokens including current)
        k_full, v_full = self._gather_kv_for_decode(
            layer, forward_batch, k, v, metadata
        )

        # Reshape Q for SageAttention: (batch_size, num_heads, head_dim)
        # Decode has 1 token per sequence
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        # For decode, is_causal must be False because SageAttention's causal mask
        # only works when qo_len == kv_len. For decode, Q=1 token, K=seq_len tokens.
        # All K tokens are past tokens that should be fully attended to.
        is_causal = False

        # Apply SageAttention with variable-length support
        output = sageattn_varlen(
            q_reshaped.contiguous(),
            k_full.contiguous(),
            v_full.contiguous(),
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            is_causal=is_causal,
            sm_scale=layer.scaling,
        )

        # Reshape output back to (batch_size, num_heads * v_head_dim)
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """
        Forward pass for extend (prefill) using SageAttention.

        This uses SageAttention's 8-bit quantized attention for the prefill
        operation where we have all Q, K, V tensors available.

        Note: SageAttention's causal mask only works when qo_len == kv_len.
        If there's a prefix (cached KV), we fall back to Triton because
        qo_len (extend_len) != kv_len (prefix_len + extend_len).
        """
        if self.forward_metadata is None:
            return self.triton_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        # Check if there's a prefix - if so, fall back to Triton because
        # SageAttention's causal mask only works when qo_len == kv_len
        has_prefix = (
            forward_batch.extend_prefix_lens_cpu is not None
            and any(p > 0 for p in forward_batch.extend_prefix_lens_cpu)
        )
        if has_prefix:
            return self.triton_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        # Save KV cache first
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # Get full K, V tensors by gathering from KV cache
        # This includes both prefix (cached) and current tokens
        metadata = self.forward_metadata

        # For extend, we need to gather the full K, V including cached prefix
        k_full, v_full = self._gather_kv_for_extend(
            layer, forward_batch, k, v, metadata
        )

        # Reshape Q for SageAttention: (total_q_tokens, num_heads, head_dim)
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        # Determine causality - only use causal when qo_len == kv_len (no prefix)
        is_causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            is_causal = False

        # Apply SageAttention with variable-length support
        output = sageattn_varlen(
            q_reshaped.contiguous(),
            k_full.contiguous(),
            v_full.contiguous(),
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            is_causal=is_causal,
            sm_scale=layer.scaling,
        )

        # Reshape output back to (total_tokens, num_heads * v_head_dim)
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _gather_kv_for_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        metadata: SageAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather full K, V tensors for extend operation.

        This combines cached KV (prefix) with current KV tokens.
        """
        bs = forward_batch.batch_size
        total_kv_tokens = metadata.cu_seqlens_k[-1].item()

        # Get K, V buffers from cache
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Allocate output tensors
        k_full = torch.empty(
            (total_kv_tokens, layer.tp_k_head_num, layer.qk_head_dim),
            dtype=k_current.dtype,
            device=self.device,
        )
        v_full = torch.empty(
            (total_kv_tokens, layer.tp_v_head_num, layer.v_head_dim),
            dtype=v_current.dtype,
            device=self.device,
        )

        # Gather K, V for each sequence
        # This includes prefix tokens (from cache) and current tokens
        k_current_reshaped = k_current.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_current_reshaped = v_current.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        k_idx = 0
        current_idx = 0
        for i in range(bs):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            prefix_len = forward_batch.extend_prefix_lens_cpu[i] if forward_batch.extend_prefix_lens_cpu is not None else 0
            extend_len = forward_batch.extend_seq_lens_cpu[i] if forward_batch.extend_seq_lens_cpu is not None else seq_len

            # Copy prefix from cache
            if prefix_len > 0:
                req_pool_idx = forward_batch.req_pool_indices[i].item()
                cache_indices = self.req_to_token[req_pool_idx, :prefix_len]

                k_full[k_idx : k_idx + prefix_len] = k_buffer[cache_indices].view(
                    prefix_len, layer.tp_k_head_num, layer.qk_head_dim
                )
                v_full[k_idx : k_idx + prefix_len] = v_buffer[cache_indices].view(
                    prefix_len, layer.tp_v_head_num, layer.v_head_dim
                )
                k_idx += prefix_len

            # Copy current tokens
            k_full[k_idx : k_idx + extend_len] = k_current_reshaped[
                current_idx : current_idx + extend_len
            ]
            v_full[k_idx : k_idx + extend_len] = v_current_reshaped[
                current_idx : current_idx + extend_len
            ]
            k_idx += extend_len
            current_idx += extend_len

        return k_full, v_full

    def _gather_kv_for_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        metadata: SageAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather full K, V tensors for decode operation.

        For decode, we gather all KV tokens from the cache. Note that k_current and
        v_current have already been saved to the cache before this function is called,
        so we gather seq_len tokens from the cache (which includes the token we just saved).
        """
        bs = forward_batch.batch_size
        total_kv_tokens = metadata.cu_seqlens_k[-1].item()

        # Get K, V buffers from cache
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Allocate output tensors
        k_full = torch.empty(
            (total_kv_tokens, layer.tp_k_head_num, layer.qk_head_dim),
            dtype=k_current.dtype,
            device=self.device,
        )
        v_full = torch.empty(
            (total_kv_tokens, layer.tp_v_head_num, layer.v_head_dim),
            dtype=v_current.dtype,
            device=self.device,
        )

        # Gather K, V for each sequence
        # After save_kv_cache, the cache contains seq_len tokens (including the one we just saved)
        k_idx = 0
        for i in range(bs):
            seq_len = forward_batch.seq_lens_cpu[i].item()

            # Gather all tokens from cache (includes the token we just saved)
            req_pool_idx = forward_batch.req_pool_indices[i].item()
            cache_indices = self.req_to_token[req_pool_idx, :seq_len]

            k_full[k_idx : k_idx + seq_len] = k_buffer[cache_indices].view(
                seq_len, layer.tp_k_head_num, layer.qk_head_dim
            )
            v_full[k_idx : k_idx + seq_len] = v_buffer[cache_indices].view(
                seq_len, layer.tp_v_head_num, layer.v_head_dim
            )
            k_idx += seq_len

        return k_full, v_full