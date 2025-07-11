from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

if is_flashinfer_available():
    import flashinfer


# TRTLLM MLA supports variable page sizes


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
        
        # Validate dimensions for TRTLLM MLA (based on test requirements)
        if self.qk_nope_head_dim != 128:
            raise ValueError(f"TRTLLM MLA requires qk_nope_head_dim=128, got {self.qk_nope_head_dim}")
        if self.kv_lora_rank != 512:
            raise ValueError(f"TRTLLM MLA requires kv_lora_rank=512, got {self.kv_lora_rank}")
        if self.qk_rope_head_dim != 64:
            raise ValueError(f"TRTLLM MLA requires qk_rope_head_dim=64, got {self.qk_rope_head_dim}")

        # Allocate larger workspace for TRTLLM (128MB as in the test)
        self.workspace_size = 128 * 1024 * 1024
        self.workspace_buffer = torch.empty(
            self.workspace_size, dtype=torch.int8, device=self.device
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info
        
        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                # Calculate max sequence length padded to page boundary
                max_seqlen_pad = triton.cdiv(
                    forward_batch.seq_lens_cpu.max().item(), self.page_size
                )
                
                # Create block indices
                block_kv_indices = torch.full(
                    (bs, max_seqlen_pad),
                    -1,
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )
                
                # Fill block indices using the existing triton kernel
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
                
                forward_batch.decode_trtllm_mla_metadata = TRTLLMMLADecodeMetadata(
                    self.workspace_buffer,
                    block_kv_indices,
                )
                self.forward_metadata = forward_batch.decode_trtllm_mla_metadata
            else:
                # Speculative decoding: use parent class implementation
                super().init_forward_metadata(forward_batch)
        else:
            # Prefill: use parent class implementation
            super().init_forward_metadata(forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        """Run forward for decode using TRTLLM kernel."""
        cache_loc = forward_batch.out_cache_loc

        if k is not None and save_kv_cache:
            if k_rope is not None:
                # MLA style KV cache storage
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )
            else:
                # Standard KV cache storage path. Skip if value tensor is absent (e.g., MLA decode tests).
                if v is not None:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Prepare query tensor - concatenate q_nope and q_rope
        if q_rope is not None:
            # q and q_rope are separate
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(-1, layer.tp_q_head_num, layer.qk_rope_head_dim)
            query = torch.cat([q_nope, q_rope], dim=-1)
        else:
            # q already contains both nope and rope parts
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Get KV cache
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # Reshape KV cache to 4-D (num_kv_heads, num_blocks, page_size, kv_dim)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(0)  # 1 KV head

        # Call TRTLLM MLA decode kernel
        raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=forward_batch.decode_trtllm_mla_metadata.workspace,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=forward_batch.decode_trtllm_mla_metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            block_size=self.page_size,
            max_seq_len=forward_batch.seq_lens.max().item(),
            scale=layer.scaling,
            out=None,
            bmm1_scale=1.0,  # Only needed for FP8
            bmm2_scale=1.0,  # Only needed for FP8
        )
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        if output.shape[0] > forward_batch.batch_size:
            output = output[: forward_batch.batch_size]
        return output 