from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
import math  # Needed for scale correction
import os

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


@dataclass
class TRTLLMMLADecodeMetadata:
    workspace: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernels from flashinfer."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        # Model parameters
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[TRTLLMMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.page_size = model_runner.page_size  # Use page size from model runner
        
        # Allocate larger workspace for TRTLLM (128MB as in the test)
        self.workspace_size = 128 * 1024 * 1024
        self.workspace_buffer = torch.empty(
            self.workspace_size, dtype=torch.int8, device=self.device
        )
        
        # CUDA graph metadata storage
        self.decode_cuda_graph_metadata = {}
        self.cuda_graph_kv_indices = None

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """Return number of blocks padded so that it satisfies TRTLLM constraint."""
        blocks = triton.cdiv(max_seq_len, self.page_size)
        min_blocks = 128 // self.page_size  # kernel requirement
        if blocks % min_blocks != 0:
            blocks = triton.cdiv(blocks, min_blocks) * min_blocks
        return blocks

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""
        # Calculate padded block size that satisfies TRTLLM constraint
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        
        self.cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=self.device,
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
        if forward_mode.is_decode_or_idle():
            if spec_info is None:
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
                    self.page_size,
                )
                metadata = TRTLLMMLADecodeMetadata(
                    self.cuda_graph_workspace,
                    block_kv_indices,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
                self.forward_metadata = metadata
            else:
                super().init_forward_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )
        else:
            super().init_forward_metadata_capture_cuda_graph(
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
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Replay CUDA graph with new inputs."""
        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                # Reuse cached metadata
                metadata = self.decode_cuda_graph_metadata[bs]
                
                # Update block indices for new sequences
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    None,
                    metadata.block_kv_indices,
                    self.req_to_token.stride(0),
                    metadata.block_kv_indices.shape[1],
                    self.page_size,
                )
                
                self.forward_metadata = metadata
            else:
                # Speculative decoding: use parent class implementation
                super().init_forward_metadata_replay_cuda_graph(
                    bs, req_pool_indices, seq_lens, seq_lens_sum,
                    encoder_lens, forward_mode, spec_info, seq_lens_cpu
                )
        else:
            # Prefill: use parent class implementation
            super().init_forward_metadata_replay_cuda_graph(
                bs, req_pool_indices, seq_lens, seq_lens_sum,
                encoder_lens, forward_mode, spec_info, seq_lens_cpu
            )

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                # seq_lens_cpu may be None when cuda-graphs are disabled
                if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                    max_seq = forward_batch.seq_lens_cpu.max().item()
                else:
                    max_seq = forward_batch.seq_lens.max().item()

                max_seqlen_pad = self._calc_padded_blocks(max_seq)

                block_kv_indices = torch.full(
                    (bs, max_seqlen_pad),
                    -1,
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )

                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_seqlen_pad,
                    self.page_size,
                )

                self.forward_metadata = TRTLLMMLADecodeMetadata(
                    self.workspace_buffer,
                    block_kv_indices,
                )
                # Expose to the ForwardBatch so that other components can access it
                forward_batch.decode_trtllm_mla_metadata = self.forward_metadata
            else:
                super().init_forward_metadata(forward_batch)
        else:
            super().init_forward_metadata(forward_batch)

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
    ):
        """Run forward for decode using TRTLLM MLA kernel."""
        cache_loc = forward_batch.out_cache_loc

        # Save KV cache if requested
        if k is not None and save_kv_cache:
            if k_rope is not None:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer, cache_loc, k, k_rope
                )
            else:
                if v is not None:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v
                    )

        # Build query tensor
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, self.qk_nope_head_dim)
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            query = torch.cat([q_nope, q_rope], dim=-1)
        else:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Scale factor for TRT-LLM MLA kernel.
        # The kernel computes softmax with scale: 1 / (sqrt(head_dim_qk) * scale)
        # where head_dim_qk = 576 (kv_lora_rank + qk_rope_head_dim).
        # To get the same result as FlashInfer (which uses layer.scaling = 1/sqrt(192)),
        # we need: 1 / (sqrt(576) * scale) = 1 / sqrt(192)
        # Therefore: scale = sqrt(576) / sqrt(192) = sqrt(3)
        scale = math.sqrt(self.kv_lora_rank + self.qk_rope_head_dim) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)

        # KV cache tensor: reshape to (num_pages, page_size, dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        # Build KV cache slices expected by TRT-LLM: slice 0 → CKV+K, slice 1 → KPE.
        pages = k_cache.view(-1, self.page_size, self.kv_cache_dim)             # (P, blk, 576)

        # According to the FlashInfer test, both slices should contain the full 576-dim tensor.
        # Use torch.stack so each slice is an independent contiguous view.
        # NOTE: this duplicates the storage but matches the reference behaviour.
        kv_cache = torch.stack([pages, pages], dim=1)                           # (P, 2, blk, 576)

        # Metadata (prefer attribute on forward_batch for compatibility)
        metadata = getattr(forward_batch, "decode_trtllm_mla_metadata", None)
        if metadata is None:
            metadata = self.forward_metadata

        # ---------- Debug output (enable with env var) ----------
        if os.getenv("SGLANG_DEBUG_TRTLLM_MLA", "0") == "1":
            print(
                f"[TRTLLM-MLA] Debug shapes before kernel call:\n"
                f"  query: {query.shape} dtype={query.dtype}\n"
                f"  kv_cache: {kv_cache.shape} dtype={kv_cache.dtype}\n"
                f"  block_tables: {metadata.block_kv_indices.shape} dtype={metadata.block_kv_indices.dtype}\n"
                f"  seq_lens: {forward_batch.seq_lens.shape} dtype={forward_batch.seq_lens.dtype}\n"
                f"  page_size: {self.page_size}\n"
                f"  max_seq_len: {metadata.block_kv_indices.shape[1] * self.page_size}\n"
                f"  scale: {scale}\n"
                f"  qk_nope_head_dim: {self.qk_nope_head_dim}\n"
                f"  kv_lora_rank: {self.kv_lora_rank}\n"
                f"  qk_rope_head_dim: {self.qk_rope_head_dim}"
            )
        # --------------------------------------------------------

        raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=metadata.workspace,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            block_size=self.page_size,
            # Avoid .item() (host sync) during CUDA graph capture.
            # max_seq_len equals padded_blocks * page_size.
            max_seq_len=int(metadata.block_kv_indices.shape[1] * self.page_size),
            scale=scale,
            bmm1_scale=1.0,
            bmm2_scale=1.0,
        )
        # TRTLLM kernel may return both V and ROPE dims (kv_lora_rank + qk_rope_head_dim).
        # We only need the value projection part (v_head_dim).
        raw_out_v = raw_out[..., : layer.v_head_dim].contiguous()

        output = raw_out_v.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        if output.shape[0] > forward_batch.batch_size:
            output = output[: forward_batch.batch_size]
        return output 