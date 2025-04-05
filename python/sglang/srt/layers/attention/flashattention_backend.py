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

from sgl_kernel.flash_attn import flash_attn_with_kvcache


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata."""

    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 0
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None
    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        topk=0,
        speculative_num_steps=0,
        step_id=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.page_size
        self.use_mla = (
            model_runner.model_config.attention_arch == AttentionArch.MLA
        ) and (not global_server_args_dict["disable_mla"])
        self.skip_prefill = skip_prefill

        # TODO: Support Topk > 1 for FlashAttentionBackend Spec Decoding
        assert (
            topk <= 1
        ), "topk must be 1 (if spec decoding) or 0 (if no spec decoding) for FlashAttentionBackend"

        self.topk = 1
        self.step_id = step_id
        self.speculative_num_steps = speculative_num_steps

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata to cache repetitive calculations."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = len(seqlens_in_batch)
        device = seqlens_in_batch.device
        if forward_batch.forward_mode.is_decode():
            # Skip Prefill or Draft Decode
            # Note: Draft Decode will be ran on the Draft Worker
            if forward_batch.spec_info is not None:
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                seq_lens_with_decode = seqlens_in_batch + (self.step_id + 1)
                metadata.cache_seqlens_int32 = seq_lens_with_decode.to(torch.int32)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                    self.step_id + 1
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
                cache_loc = forward_batch.out_cache_loc.view(
                    self.speculative_num_steps, -1
                ).T

                for idx, single_seq_len in enumerate(seq_lens_with_decode):
                    real_bsz_start_idx = idx
                    real_bsz_end_idx = idx + 1
                    metadata.page_table[
                        real_bsz_start_idx:real_bsz_end_idx,
                        (single_seq_len - (self.step_id + 1)) : single_seq_len,
                    ] = cache_loc[
                        real_bsz_start_idx:real_bsz_end_idx, : (self.step_id + 1)
                    ]
            else:  # Normal Decode without Spec Decoding
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
        elif forward_batch.forward_mode.is_target_verify():
            # Note: Target Verify will be ran on the Target Worker
            draft_token_num = forward_batch.spec_info.draft_token_num
            metadata.cache_seqlens_int32 = (
                forward_batch.seq_lens + draft_token_num
            ).to(torch.int32)
            metadata.max_seq_len_q = draft_token_num
            metadata.max_seq_len_k = (
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
            metadata.cu_seqlens_q = torch.arange(
                0,
                batch_size * draft_token_num + 1,
                draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

        elif forward_batch.forward_mode.is_extend_or_draft_extend():
            # Normal or Draft Extend (Both of them will be ran on the Target Worker)
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            # Precompute maximum sequence length
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            # Precompute page table
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
            # Precompute cumulative sequence lengths
            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
            else:
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
                metadata.max_seq_len_q = metadata.max_seq_len_k

        # Precompute strided indices
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )
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

        # Use Flash Attention for prefill
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
        self.decode_cuda_graph_metadata = {
            # Page table for token mapping (batch_size, max_context_len)
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "page_table_draft_decode": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 128, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 128, dtype=torch.int32, device=self.device
            ),
        }

        self.target_verify_metadata = {
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.zeros(
                max_bs + 128, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 128, dtype=torch.int32, device=self.device
            ),
            "max_seqlen_q": 0,
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
        device = seq_lens.device
        if forward_mode.is_decode():
            if spec_info is not None:
                # Draft Decode
                metadata.cu_seqlens_q = torch.arange(
                    0, bs + 1, dtype=torch.int32, device=device
                )
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]

                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]

                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.max_seq_len_k = seq_lens.max().item() + (self.step_id + 1)
                metadata.page_table = self.decode_cuda_graph_metadata[
                    "page_table_draft_decode"
                ][req_pool_indices, :]
            else:
                # Normal Decode
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
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
            self.decode_cuda_graph_metadata[bs] = metadata
        elif forward_mode.is_target_verify():
            draft_token_num = spec_info.draft_token_num

            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + draft_token_num).to(torch.int32)
            )

            metadata.max_seq_len_q = draft_token_num
            metadata.max_seq_len_k = seq_lens.max().item() + draft_token_num

            metadata.cu_seqlens_q = self.target_verify_metadata["cu_seqlens_q"][
                torch.arange(
                    0,
                    bs * draft_token_num + 1,
                    draft_token_num,
                    dtype=torch.int32,
                    device=device,
                )
            ]
            cu_k = self.target_verify_metadata["cu_seqlens_k"][: (bs + 1)]
            cu_k.copy_(
                torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
            )
            metadata.cu_seqlens_k = cu_k
            metadata.page_table = self.target_verify_metadata["page_table"][
                req_pool_indices, :
            ]

            self.target_verify_metadata[bs] = metadata

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
        out_cache_loc: torch.Tensor = None,
    ):
        # """Initialize forward metadata for replaying CUDA graph."""
        device = seq_lens.device
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        if forward_mode.is_decode():
            metadata = self.decode_cuda_graph_metadata[bs]

            if spec_info is not None:
                # Draft Decode
                max_len = seq_lens_cpu.max().item()
                metadata.max_seq_len_k = max_len + (self.step_id + 1)

                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + (self.step_id + 1)).to(torch.int32)
                )

                metadata.max_seq_len_k = seq_lens_cpu.max().item() + (self.step_id + 1)

                metadata.cu_seqlens_k.copy_(
                    torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                )

                page_table = self.req_to_token[
                    req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)
            else:
                # Normal Decode
                max_len = seq_lens_cpu.max().item()
                metadata.max_seq_len_k = max_len

                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )

                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    :,
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                page_indices = page_indices[req_pool_indices] // self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
                metadata.page_table[:, max_seq_pages:].fill_(0)

        elif forward_mode.is_target_verify():
            metadata = self.target_verify_metadata[bs]
            draft_token_num = spec_info.draft_token_num

            metadata.cu_seqlens_q.copy_(
                torch.arange(
                    0,
                    bs * draft_token_num + 1,
                    draft_token_num,
                    dtype=torch.int32,
                    device=device,
                )
            )
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + draft_token_num).to(torch.int32)
            )

            metadata.max_seq_len_k = seq_lens_cpu.max().item() + draft_token_num
            metadata.cu_seqlens_k.copy_(
                torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
            )
            page_table = self.req_to_token[req_pool_indices, : metadata.max_seq_len_k]
            metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 0


class FlashAttentionMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    step_id=i,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )
