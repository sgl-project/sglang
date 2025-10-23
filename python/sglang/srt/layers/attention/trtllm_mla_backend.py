from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.utils import (
    create_flashmla_kv_indices_triton,
    get_num_page_per_block_flashmla,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cuda, is_flashinfer_available
from sglang.srt.utils.common import cached_triton_kernel

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import concat_mla_absorb_q

# Constants
DEFAULT_WORKSPACE_SIZE_MB = 128  # Memory workspace size in MB

# Block constraint from flashinfer requirements
# From flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0
# This imposes that the total number of blocks must be divisible by
# (128 / block_size). We capture the 128 constant here so we can
# compute the LCM with other padding constraints.
TRTLLM_BLOCK_CONSTRAINT = 128


@cached_triton_kernel(lambda _, kwargs: (kwargs["BLOCK_SIZE"]))
@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,  # Input query tensor [total_seq_len, num_heads, head_dim]
    padded_q_ptr,  # Output padded query tensor [batch_size, max_seq_len, num_heads, head_dim]
    seq_lens_q_ptr,  # Sequence lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing."""
    # Use 3D program IDs: (batch_seq, head_block, dim_block)
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return

    # Load cumulative sum to get start position in input tensor
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset
    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    # Calculate output offset
    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@cached_triton_kernel(lambda _, kwargs: (kwargs["BLOCK_SIZE"]))
@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  # Input raw output tensor (batch_size, token_per_batch, tp_q_head_num, v_head_dim)
    output_ptr,  # Output tensor (-1, tp_q_head_num, v_head_dim)
    accept_length_ptr,  # Accept lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]
    batch_size,
    token_per_batch,
    tp_q_head_num,
    v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing."""
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    accept_len = tl.load(accept_length_ptr + batch_id)

    if seq_pos >= accept_len:
        return

    # Load cumulative sum to get start position in output tensor
    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset: (batch_id, seq_pos, head_id, dim_id)
    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


global_zero_init_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations."""

    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""

    block_kv_indices: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    max_seq_len_q: Optional[int] = None
    sum_seq_lens_q: Optional[int] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    seq_lens_q: Optional[torch.Tensor] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

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
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

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
        # 2. Triton: number of pages per block
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

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
        num_tokens_per_bs = max_num_tokens // max_bs

        # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
        self.padded_q_buffer = torch.zeros(
            (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),
            dtype=self.data_type,
            device=self.device,
        )

        # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
        self.unpad_output_buffer = torch.zeros(
            (max_num_tokens, self.num_q_heads, 512),
            dtype=self.data_type,
            device=self.device,
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
        spec_info: Optional[SpecInput],
    ):
        """Initialize metadata for CUDA graph capture."""

        # Delegate to parent for non-decode modes.
        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        ):
            return super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens

        # Custom fast-path for decode/idle.
        # Capture with full width so future longer sequences are safe during replay
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]

        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks_per_seq,
            PAGED_SIZE=self.page_size,
        )

        # Record the true maximum sequence length for this capture batch so that
        # the kernel launch path (which requires an int not a tensor) can reuse
        # it safely during both capture and replay.
        max_seq_len_val = int(seq_lens.max().item())

        metadata = TRTLLMMLADecodeMetadata(
            block_kv_indices,
            max_seq_len_val,
        )
        if forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs + 1
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

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
        """Replay CUDA graph with new inputs."""
        # Delegate to parent for non-decode modes.
        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        ):
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

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            del seq_lens_sum  # not handle "num_draft_tokens" but we do not need it

        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_draft_extend(include_v2=True):
            accept_length = spec_info.accept_length[:bs]
            if spec_info.accept_length_cpu:
                metadata.max_seq_len_q = max(spec_info.accept_length_cpu[:bs])
                metadata.sum_seq_lens_q = sum(spec_info.accept_length_cpu[:bs])
            else:
                metadata.max_seq_len_q = 1
                metadata.sum_seq_lens_q = bs
            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(accept_length, dim=0, dtype=torch.int32)
            )
            metadata.seq_lens_q.copy_(accept_length)

        # Update block indices for new sequences.
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens[:bs],
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=self.page_size,
        )

        # Update stored max_seq_len so subsequent kernel calls use the correct value
        # Prefer CPU tensor to avoid GPU synchronization when available.
        if seq_lens_cpu is not None:
            metadata.max_seq_len = int(seq_lens_cpu.max().item())
        else:
            metadata.max_seq_len = int(seq_lens.max().item())

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        # Delegate to parent for non-decode modes.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            if self.disable_chunked_prefix_cache:
                super().init_forward_metadata(forward_batch)

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            cum_seq_lens_q = torch.cat(
                (
                    torch.tensor([0], device=forward_batch.seq_lens.device),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            bs = forward_batch.batch_size

            # Get maximum sequence length.
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = forward_batch.seq_lens.max().item()

            seq_lens = forward_batch.seq_lens

            if forward_batch.forward_mode.is_target_verify():
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens

            max_seqlen_pad = self._calc_padded_blocks(max_seq)
            block_kv_indices = self._create_block_kv_indices(
                bs,
                max_seqlen_pad,
                forward_batch.req_pool_indices,
                seq_lens,
                seq_lens.device,
            )

            max_seq_len_val = int(max_seq)
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata(
                block_kv_indices, max_seq_len_val
            )
            if forward_batch.forward_mode.is_draft_extend(include_v2=True):
                max_seq = forward_batch.seq_lens_cpu.max().item()

                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        super().init_mha_chunk_metadata(forward_batch, disable_flashinfer_ragged=True)

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

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel."""
        batch_size = cu_seqlens_q.shape[0] - 1
        max_seq_len_q = padded_q.shape[1]
        num_heads = padded_q.shape[2]
        head_dim = padded_q.shape[3]

        # Launch Triton kernel with 3D grid for parallelized head and dim processing
        BLOCK_SIZE = 64
        num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
        num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
        grid = (batch_size * max_seq_len_q, num_head_blocks, num_dim_blocks)

        pad_draft_extend_query_kernel[grid](
            q_ptr=q,
            padded_q_ptr=padded_q,
            seq_lens_q_ptr=seq_lens_q,
            cumsum_ptr=cu_seqlens_q,
            batch_size=batch_size,
            max_seq_len=max_seq_len_q,
            num_heads=num_heads,
            head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return padded_q

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel."""
        # raw_out: (batch_size, token_per_batch, layer.tp_q_head_num, layer.v_head_dim)
        batch_size = seq_lens_q.shape[0]
        token_per_batch = raw_out.shape[1]  # max_seq_len
        tp_q_head_num = raw_out.shape[2]  # num_heads
        v_head_dim = raw_out.shape[3]  # head_dim
        total_tokens = sum_seq_lens_q

        # Check if we're in CUDA graph mode (buffers are pre-allocated)
        if self.unpad_output_buffer is not None:
            # Use pre-allocated buffer for CUDA graph compatibility
            output = self.unpad_output_buffer[:total_tokens, :, :].to(
                dtype=raw_out.dtype
            )
        else:
            # Dynamic allocation for non-CUDA graph mode
            output = torch.empty(
                (total_tokens, tp_q_head_num, v_head_dim),
                dtype=raw_out.dtype,
                device=raw_out.device,
            )

        # Launch Triton kernel with 3D grid for parallelized head and dim processing
        BLOCK_SIZE = 64
        num_head_blocks = triton.cdiv(tp_q_head_num, BLOCK_SIZE)
        num_dim_blocks = triton.cdiv(v_head_dim, BLOCK_SIZE)
        grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

        unpad_draft_extend_output_kernel[grid](
            raw_out_ptr=raw_out,
            output_ptr=output,
            accept_length_ptr=seq_lens_q,
            cumsum_ptr=cu_seqlens_q,
            batch_size=batch_size,
            token_per_batch=token_per_batch,
            tp_q_head_num=tp_q_head_num,
            v_head_dim=v_head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output[:total_tokens, :, :]

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
            query = _concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
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
            or self.forward_decode_metadata
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
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            max_seq_len=metadata.max_seq_len_k,
            bmm1_scale=bmm1_scale,
        )

        # Reshape output directly without slicing
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
    ) -> torch.Tensor:
        # TODO refactor to avoid code duplication
        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
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

        # TODO refactor to avoid code duplication
        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = _concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function
            q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)

        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )

            # Ensure query has shape [bs, num_draft_tokens, num_q_heads, head_dim]
            bs = forward_batch.batch_size

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q_scale = 1.0
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            q = q.to(self.data_type)

            bmm1_scale = q_scale * k_scale * layer.scaling
            if forward_batch.forward_mode.is_target_verify():
                seq_lens = (
                    forward_batch.seq_lens.to(torch.int32)
                    + forward_batch.spec_info.draft_token_num
                )
                max_seq_len = (
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num
                )
            else:
                seq_lens = forward_batch.seq_lens.to(torch.int32)
                max_seq_len = metadata.max_seq_len_k
                # Check if we're in CUDA graph mode (buffers are pre-allocated)
                if self.padded_q_buffer is not None:
                    # Use pre-allocated buffer for CUDA graph compatibility
                    padded_q = self.padded_q_buffer[
                        :bs, : metadata.max_seq_len_q, :, :
                    ].to(dtype=q.dtype)
                else:
                    # Dynamic allocation for non-CUDA graph mode
                    padded_q = torch.zeros(
                        bs,
                        metadata.max_seq_len_q,
                        layer.tp_q_head_num,
                        layer.head_dim,
                        dtype=q.dtype,
                        device=q.device,
                    )
                q = self.pad_draft_extend_query(
                    q, padded_q, metadata.seq_lens_q, metadata.cu_seqlens_q
                )

            # TODO may use `mla_rope_quantize_fp8` fusion
            q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            assert kv_cache.dtype == self.data_type

            raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=metadata.block_kv_indices,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
            )

            # Reshape output directly without slicing

            if forward_batch.forward_mode.is_draft_extend(include_v2=True):
                raw_out = self.unpad_draft_extend_output(
                    raw_out,
                    metadata.cu_seqlens_q,
                    metadata.seq_lens_q,
                    metadata.sum_seq_lens_q,
                )
            output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        if forward_batch.attn_attend_prefix_cache:
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            output_shape = (q.shape[0], layer.tp_q_head_num, layer.v_head_dim)
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=self.workspace_buffer,
                seq_lens=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
                o_sf_scale=-1.0,
                batch_size=forward_batch.batch_size,
                window_left=-1,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                enable_pdl=False,
                is_causal=False,
                return_lse=True,
                out=torch.zeros(*output_shape, dtype=q.dtype, device=q.device),
            )

        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace_buffer,
            seq_lens=self.forward_prefill_metadata.seq_lens,
            max_q_len=self.forward_prefill_metadata.max_seq_len,
            max_kv_len=self.forward_prefill_metadata.max_seq_len,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=forward_batch.batch_size,
            window_left=-1,
            cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
            cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
            enable_pdl=False,
            is_causal=True,
            return_lse=forward_batch.mha_return_lse,
        )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )


def _concat_mla_absorb_q_general(q_nope, q_rope):
    if _is_cuda and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:
        return concat_mla_absorb_q(q_nope, q_rope)
    else:
        return torch.cat([q_nope, q_rope], dim=-1)
