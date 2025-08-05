from __future__ import annotations

from python.sglang.srt.layers.radix_attention import RadixAttention

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.utils import (
    TRITON_PAD_NUM_PAGE_PER_BLOCK,
    create_flashinfer_kv_indices_triton,
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
class TRTLLMMHADecodeMetadata:
    """Metadata for TRTLLM MHA decode operations."""

    workspace: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None


@dataclass
class TRTLLMMHAMetadata:
    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for `query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None


class TRTLLMHAAttnBackend(FlashInferAttnBackend):
    """TRTLLM MHA attention kernel from flashinfer."""

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

        # MHA-specific dimensions
        self.max_context_len = model_runner.model_config.context_len
        self.sliding_window_size = model_runner.sliding_window_size
        self.hidden_size = config.hidden_size

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
        self.forward_metadata: Union[TRTLLMMHADecodeMetadata, TRTLLMMHAMetadata] = None

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

        create_flashinfer_kv_indices_triton[(batch_size,)](
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
        """Initialize CUDA graph state for TRTLLM MHA."""
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        self.cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_workspace = torch.empty(
            self.workspace_size, dtype=torch.int8, device=self.device
        )

    # todo: kv_indptr
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

        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.shape[1],
            PAGE_SIZE=self.page_size,
        )

        metadata = TRTLLMMHADecodeMetadata(self.cuda_graph_workspace, block_kv_indices)
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
        # todo: kv_indptr
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.shape[1],
            PAGE_SIZE=self.page_size,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        metadata = TRTLLMMHAMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Normal Decode
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
        else:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

        # # Delegate to parent for non-decode modes or when speculative execution is used.
        # if not (
        #     forward_batch.forward_mode.is_decode_or_idle()
        #     and forward_batch.spec_info is None
        # ):
        #     return super().init_forward_metadata(forward_batch)

        # bs = forward_batch.batch_size

        # # Get maximum sequence length.
        # if getattr(forward_batch, "seq_lens_cpu", None) is not None:
        #     max_seq = forward_batch.seq_lens_cpu.max().item()
        # else:
        #     max_seq = forward_batch.seq_lens.max().item()

        # max_seqlen_pad = self._calc_padded_blocks(max_seq)
        # # todo: kv_indptr
        # block_kv_indices = self._create_block_kv_indices(
        #     bs,
        #     max_seqlen_pad,
        #     forward_batch.req_pool_indices,
        #     forward_batch.seq_lens,
        #     forward_batch.seq_lens.device,
        # )

        # self.forward_metadata = TRTLLMMHADecodeMetadata(
        #     self.workspace_buffer, block_kv_indices
        # )
        # forward_batch.decode_trtllm_mha_metadata = self.forward_metadata

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MHA kernel."""
        # Save KV cache if requested
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

        # Prepare query tensor inline
        assert (
            q.dim() == 3 and q.shape[1] == 1 and q.shape[2] == layer.head_dim
        ), "q must be of shape [bs, 1, head_dim]"

        # Prepare KV cache inline (num_blocks, 2, num_kv_heads, page_size, head_dim)
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        assert (
            kv_cache.dim() == 5
            and kv_cache.shape[1] == 2
            and kv_cache.shape[2] == self.num_kv_heads
            and kv_cache.shape[3] == self.page_size
            and kv_cache.shape[4] == layer.head_dim
        ), "kv_cache must be of shape [num_blocks, 2, num_kv_heads, page_size, head_dim]"

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mha_metadata", None)
            or self.forward_metadata
        )

        # Scale computation for TRTLLM MHA kernel:
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
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=metadata.workspace,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            max_seq_len=int(metadata.block_kv_indices.shape[1] * self.page_size),
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
        )

        return raw_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, self.page_size, self.num_kv_heads, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, self.num_kv_heads, layer.head_dim)
        kv_cache = torch.cat([k_cache, v_cache], dim=1)

        # TODO: bmm1_scale and bmm2_scale might require modification
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0

        o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.forward_metadata.page_table,
            seq_lens=self.forward_metadata.cache_seqlens_int32,
            max_q_len=self.forward_metadata.max_seq_len_q,
            max_seq_len=self.forward_metadata.max_seq_len_k,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=forward_batch.batch_size,
            cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
            cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
            window_left=self.sliding_window_size,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
