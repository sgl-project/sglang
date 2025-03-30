from __future__ import annotations

from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

"""
Support different attention backends.
Now there are three backends: FlashInfer, Triton and FlashAttention.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from flash_attn_interface import flash_attn_with_kvcache


@dataclass
class FlashAttentionMetadata:
    """Metadata for decode operations to avoid redundant computations."""

    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    max_seq_len_q: int = 0
    max_seq_len_k: int = 0
    window_size: tuple = (-1, -1)
    page_table: torch.Tensor = None
    cache_seqlens_int32: torch.Tensor = None


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # Initialize metadata
        self.forward_metadata: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.page_size
        self.use_mla = (
            model_runner.model_config.attention_arch == AttentionArch.MLA
        ) and (not global_server_args_dict["disable_mla"])

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata to cache repetitive calculations."""
        # Create metadata based on forward mode
        metadata = FlashAttentionMetadata()

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

        # Precompute strided indices
        # [0, page_size, 2 * page_size, ...]
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        if forward_batch.forward_mode == ForwardMode.DECODE:
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
        else:
            # Precompute cumulative sequence lengths
            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
            else:
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
                metadata.max_seq_len_q = metadata.max_seq_len_k
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

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Use precomputed metadata
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None
            else (-1, -1)
        )

        page_table = metadata.page_table

        # # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            o = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
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
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]
            o = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=metadata.max_seq_len_q,
                softmax_scale=layer.scaling,
                causal=True,
                softcap=layer.logit_cap,
                k_descale=layer.k_scale,
                v_descale=layer.v_scale,
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

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
        # Save KV cache if needed
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Use precomputed metadata
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None
            else (-1, -1)
        )

        page_table = metadata.page_table

        if not self.use_mla:
            # Do multi-head attention

            # Get KV cache
            kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            # Pre-reshape query tensor
            q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

            # Run attention with precomputed values
            o = flash_attn_with_kvcache(
                q=q_reshaped,
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
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
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

            o = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=1,
                softmax_scale=layer.scaling,
                causal=True,
                softcap=layer.logit_cap,
                k_descale=layer.k_scale,
                v_descale=layer.v_scale,
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        # Initialize fixed size tensors for decode operations
        self.decode_cuda_graph_metadata = {
            # Page table for token mapping (batch_size, max_context_len)
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

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
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()
        # Get sequence information
        metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
        batch_size = len(seq_lens)
        device = seq_lens.device
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        # Precompute maximum sequence length
        metadata.max_seq_len_k = seq_lens.max().item()
        # Precompute page table
        metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
            req_pool_indices, :
        ]
        if forward_mode == ForwardMode.DECODE:
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
        else:
            raise ValueError("Do not support Prefill Mode cuda graph")
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

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
        # """Initialize forward metadata for replaying CUDA graph."""
        metadata = self.decode_cuda_graph_metadata[bs]

        # For CPU operations
        max_len = seq_lens_cpu[:bs].max().item()
        metadata.max_seq_len_k = max_len

        # For GPU operations
        seq_lens_in_batch = seq_lens[:bs]
        metadata.cache_seqlens_int32 = seq_lens_in_batch.to(torch.int32)
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seq_lens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )

        max_seq_pages = (metadata.max_seq_len_k + self.page_size - 1) // self.page_size
        page_indices = self.req_to_token[
            :, self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages]
        ]
        page_indices = page_indices[req_pool_indices[:bs]] // self.page_size
        metadata.page_table[:, :max_seq_pages].copy_(page_indices)
        metadata.page_table[:, max_seq_pages:].fill_(0)
        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 0
