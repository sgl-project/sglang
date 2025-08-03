from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import (
    TRITON_PAD_NUM_PAGE_PER_BLOCK,
    create_flashmla_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

# Constants
DEFAULT_WORKSPACE_SIZE_MB = 128  # Memory workspace size in MB

# Block constraint from flashinfer requirements
# From flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0
# This imposes that the total number of blocks must be divisible by
# (128 / block_size). We capture the 128 constant here so we can
# compute the LCM with other padding constraints.
TRTLLM_BLOCK_CONSTRAINT = 128


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""

    workspace: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(model_runner, skip_prefill, kv_indptr_buf, q_indptr_decode_buf)

        config = model_runner.model_config

        # Model parameters
        self.num_q_heads = config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = config.get_num_kv_heads(get_attention_tp_size())
        self.num_local_heads = config.num_attention_heads // get_attention_tp_size()

        # MLA-specific dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Runtime parameters
        self.scaling = config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Workspace allocation
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        self.workspace_buffer = torch.empty(
            self.workspace_size, dtype=torch.int8, device=self.device
        )

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.cuda_graph_kv_indices = None
        self.forward_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.

        Args:
            max_seq_len: Maximum sequence length in tokens

        Returns:
            Number of blocks padded to satisfy all constraints
        """
        blocks = triton.cdiv(max_seq_len, self.page_size)

        # Apply dual constraints (take LCM to satisfy both):
        # 1. TRT-LLM: block_num % (128 / page_size) == 0
        # 2. Triton: page table builder uses 64-index bursts, needs multiple of 64
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        constraint_lcm = math.lcm(trtllm_constraint, TRITON_PAD_NUM_PAGE_PER_BLOCK)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.

        Args:
            batch_size: Batch size
            max_blocks: Maximum number of blocks per sequence
            req_pool_indices: Request pool indices
            seq_lens: Sequence lengths
            device: Target device

        Returns:
            Block KV indices tensor
        """
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        create_flashmla_kv_indices_triton[(batch_size,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            NUM_PAGE_PER_BLOCK=TRITON_PAD_NUM_PAGE_PER_BLOCK,
            PAGED_SIZE=self.page_size,
        )

        return block_kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        self.cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_workspace = torch.empty(
            self.workspace_size, dtype=torch.int8, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        """Initialize metadata for CUDA graph capture."""
        # Delegate to parent for non-decode modes or when speculative execution is used.
        if not (forward_mode.is_decode_or_idle() and spec_info is None):
            return super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        # Custom fast-path for decode/idle without speculative execution.
        max_seqlen_pad = self._calc_padded_blocks(seq_lens.max().item())
        block_kv_indices = self.cuda_graph_kv_indices[:bs, :max_seqlen_pad]

        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_seqlen_pad,
            NUM_PAGE_PER_BLOCK=TRITON_PAD_NUM_PAGE_PER_BLOCK,
            PAGED_SIZE=self.page_size,
        )

        metadata = TRTLLMMLADecodeMetadata(self.cuda_graph_workspace, block_kv_indices)
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
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Replay CUDA graph with new inputs."""
        # Delegate to parent for non-decode modes or when speculative execution is used.
        if not (forward_mode.is_decode_or_idle() and spec_info is None):
            return super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        metadata = self.decode_cuda_graph_metadata[bs]

        # Update block indices for new sequences.
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens[:bs],
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            NUM_PAGE_PER_BLOCK=TRITON_PAD_NUM_PAGE_PER_BLOCK,
            PAGED_SIZE=self.page_size,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        # Delegate to parent for non-decode modes or when speculative execution is used.
        if not (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is None
        ):
            return super().init_forward_metadata(forward_batch)

        bs = forward_batch.batch_size

        # Get maximum sequence length.
        if getattr(forward_batch, "seq_lens_cpu", None) is not None:
            max_seq = forward_batch.seq_lens_cpu.max().item()
        else:
            max_seq = forward_batch.seq_lens.max().item()

        max_seqlen_pad = self._calc_padded_blocks(max_seq)
        block_kv_indices = self._create_block_kv_indices(
            bs,
            max_seqlen_pad,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens.device,
        )

        self.forward_metadata = TRTLLMMLADecodeMetadata(
            self.workspace_buffer, block_kv_indices
        )
        forward_batch.decode_trtllm_mla_metadata = self.forward_metadata

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""
        # Save KV cache if requested
        if k is not None and save_kv_cache:
            cache_loc = forward_batch.out_cache_loc
            if k_rope is not None:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer, cache_loc, k, k_rope
                )
            elif v is not None:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        # Prepare query tensor inline
        if q_rope is not None:
            # q contains NOPE part (v_head_dim)
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = torch.cat([q_nope, q_rope_reshaped], dim=-1)
        else:
            # q already has both parts
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1
        if query.dim() == 3:
            query = query.unsqueeze(1)

        # Prepare KV cache inline
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        pages = k_cache.view(-1, self.page_size, self.kv_cache_dim)
        # TRT-LLM expects single KV data with extra dimension
        kv_cache = pages.unsqueeze(1)

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_metadata
        )

        # Scale computation for TRTLLM MLA kernel:
        # - BMM1 scale = q_scale * k_scale * softmax_scale
        # - For FP16 path we keep q_scale = 1.0, softmax_scale = 1/sqrt(head_dim) which is pre-computed as layer.scaling
        # - k_scale is read from model checkpoint if available
        # TODO: Change once fp8 path is supported
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )

        bmm1_scale = q_scale * k_scale * layer.scaling

        # Call TRT-LLM kernel
        raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=metadata.workspace,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            max_seq_len=int(metadata.block_kv_indices.shape[1] * self.page_size),
            bmm1_scale=bmm1_scale,
        )

        # Extract value projection part and reshape
        raw_out_v = raw_out[..., : layer.v_head_dim].contiguous()
        output = raw_out_v.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        return output
