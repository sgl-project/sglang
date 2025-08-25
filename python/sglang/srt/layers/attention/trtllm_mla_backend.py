from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
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

global_zero_init_workspace_buffer = None


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
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
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

        self.decode_cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        self.decode_cuda_graph_workspace = torch.empty(
            self.workspace_size, dtype=torch.int8, device=self.device
        )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

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

        # Delegate to parent for non-decode modes.
        if not forward_mode.is_decode_or_idle():
            return super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        # Custom fast-path for decode/idle.
        max_seqlen_pad = self._calc_padded_blocks(seq_lens.max().item())
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_seqlen_pad]

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

        metadata = TRTLLMMLADecodeMetadata(
            self.decode_cuda_graph_workspace, block_kv_indices
        )
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
        # Delegate to parent for non-decode modes.
        if not forward_mode.is_decode_or_idle():
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
        # Delegate to parent for non-decode modes.
        if not forward_batch.forward_mode.is_decode_or_idle():
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

    def quantize_and_rope_for_fp8(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        forward_batch: ForwardBatch,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and apply RoPE for FP8 attention path.

        This function handles the FP8 quantization and RoPE application for MLA attention.
        It takes separate query/key nope and rope components, applies RoPE to the rope parts,
        quantizes all components to FP8, and merges the query components into a single tensor.

        Args:
            q_nope: Query no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            q_rope: Query RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            k_nope: Key no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            k_rope: Key RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            forward_batch: Forward batch containing position information
            cos_sin_cache: Precomputed cosine/sine cache for RoPE
                - expected dtype: matches q_/k_ input dtype (torch.bfloat16)
            is_neox: Whether to use NeoX-style RoPE (interleaved) or GPT-style (half rotation)

        Returns:
            tuple: (merged_q_out, k_nope_out, k_rope_out) quantized to FP8
                - merged_q_out: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim], dtype=torch.float8_e4m3fn
                - k_nope_out:   [seq_len, num_heads, kv_lora_rank], dtype=torch.float8_e4m3fn
                - k_rope_out:   [seq_len, num_heads, qk_rope_head_dim], dtype=torch.float8_e4m3fn
        """
        attn_dtype = torch.float8_e4m3fn
        q_len, num_heads = q_rope.shape[0], q_rope.shape[1]

        # Allocate output tensors with FP8 dtype
        # Query output will contain merged nope + rope components
        q_out = q_rope.new_empty(
            q_len,
            num_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=attn_dtype,
        )

        # Key outputs maintain original shapes but with FP8 dtype
        k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)
        k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)

        # Apply RoPE and quantize all components in a single fused kernel call
        # This kernel handles:
        # 1. RoPE application to q_rope and k_rope using cos_sin_cache and positions
        # 2. Quantization of all components to FP8 format
        # 3. Output placement into pre-allocated tensors
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=forward_batch.positions,
            is_neox=is_neox,
            quantize_dtype=attn_dtype,
            # Output tensor slicing: q_out contains [nope_part, rope_part]
            q_rope_out=q_out[..., self.kv_lora_rank :],  # RoPE part goes to end
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., : self.kv_lora_rank],  # Nope part goes to beginning
            k_nope_out=k_nope_out,
            # Quantization scales (set to 1.0 for no additional scaling)
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )

        return q_out, k_nope_out, k_rope_out

    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope
        k: torch.Tensor,  # k_nope
        v: torch.Tensor,  # not used in this backend
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""
        merge_query = q_rope is not None
        if self.data_type == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = self.quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch,
                cos_sin_cache,
                is_neox,
            )
            merge_query = False

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = torch.cat([q_nope, q_rope_reshaped], dim=-1)
        else:
            # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1
        if query.dim() == 3:
            query = query.unsqueeze(1)

        # Prepare KV cache inline
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_metadata
        )

        # Scale computation for TRTLLM MLA kernel BMM1 operation:
        # The final BMM1 scale is computed as: q_scale * k_scale * softmax_scale
        # Scale components:
        # - q_scale: Query scaling factor (set to 1.0 for both FP16/FP8 paths)
        # - k_scale: Key scaling factor from model checkpoint (defaults to 1.0 if not available)
        # - softmax_scale: Attention softmax scaling = 1/sqrt(head_dim), pre-computed as layer.scaling
        # This unified approach works for both FP16 and FP8 quantized attention paths.
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


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
