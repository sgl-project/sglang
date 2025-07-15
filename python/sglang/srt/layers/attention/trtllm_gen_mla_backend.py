from __future__ import annotations

"""
Support attention backend for TRTLLM-Gen MLA kernels from flashinfer.
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
class TRTLLMGENMLADecodeMetadata:
    workspace: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None


class TRTLLMGENMLABackend(FlashInferMLAAttnBackend):
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
        self.forward_metadata: Union[TRTLLMGENMLADecodeMetadata] = None
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

        # The Triton helper that builds `block_kv_indices` emits `NUM_PAGE_PER_BLOCK`
        # (= 64) indices at a time, independent of `page_size`.  To avoid it writing
        # past the end of the row we **must** make every row at least 64 long.
        # (Side-effect: max_seq_len is effectively rounded up to 64 × page_size = 2 K
        #  tokens for page_size 32, which is still below the 2048-ctx used here.)
        min_blocks = 64
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
                metadata = TRTLLMGENMLADecodeMetadata(
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
        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            print(f"[TRTLLM-MLA] init_forward_metadata_replay_cuda_graph: bs={bs}, forward_mode={forward_mode}, spec_info={spec_info is not None}")
        
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
                # Pad invalid blocks to keep TRTLLM happy
                # self._pad_invalid_blocks(metadata.block_kv_indices)

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

        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            print(f"[TRTLLM-MLA] init_forward_metadata: bs={bs}, forward_mode={forward_batch.forward_mode}, spec_info={spec_info is not None}")
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

                # Ensure padded blocks are valid to avoid TRTLLM skipping shorter seqs
                # self._pad_invalid_blocks(block_kv_indices)

                self.forward_metadata = TRTLLMGENMLADecodeMetadata(
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
            # q contains the NOPE part (v_head_dim)
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped_q[:, :, : layer.v_head_dim]
            q_rope = reshaped_q[:, :, layer.v_head_dim :]

        # Concatenate to build the full query as expected by TRTLLM kernel
        query = torch.cat([q_nope, q_rope], dim=-1)


        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            print(f"[TRTLLM-MLA]: model_runner.model_config.scaling.scaling is {self.model_runner.model_config.scaling}")

        # Get the model scaling factor
        # TRTLLM kernel applies the 1/sqrt(192) factor internally, so we
        # should pass `1.0` here (see Flash-Infer equivalence tests).
        sm_scale = 1 
        # (scale  * ((512 + 64) ** 0.5)) / ((128 + 64) ** 0.5)
        # ( sqrt(3)) / sqrt(192)

        # KV cache tensor: reshape to (num_pages, page_size, dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        # Build KV cache slices expected by TRT-LLM: slice 0 → CKV+K, slice 1 → KPE.
        pages = k_cache.view(-1, self.page_size, self.kv_cache_dim)             # (P, blk, 576)

        # Retrieve metadata so we can check block tables.
        metadata = getattr(forward_batch, "decode_trtllm_mla_metadata", None)
        if metadata is None:
            metadata = self.forward_metadata

        # ---------------------------------------------------------------------
        # According to the FlashInfer test, both slices should contain the full 576-dim tensor.
        # Use torch.stack so each slice is an independent contiguous view **after** we have
        # patched the pages in-place.
        kv_cache = torch.stack([pages, pages], dim=1)                           # (P, 2, blk, 576)

        # Metadata already obtained above; no change needed.

        # ------------- DEBUG KV CACHE CONSTRUCTION -------------
        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            print(
                f"[TRTLLM-MLA]  k_cache_raw: {k_cache.shape}  {k_cache.dtype}\n"
                f"[TRTLLM-MLA]  pages     : {pages.shape}  {pages.dtype}\n"
                f"[TRTLLM-MLA]  kv_cache  : {kv_cache.shape}  {kv_cache.dtype}\n"
                f"[TRTLLM-MLA]  block_kv_indices: {metadata.block_kv_indices.shape}  {metadata.block_kv_indices.dtype}\n"
                f"[TRTLLM-MLA]  workspace: {metadata.workspace.shape if metadata.workspace is not None else 'None'}\n"
                f"[TRTLLM-MLA]  max_seq_len: {metadata.block_kv_indices.shape[1] * self.page_size}\n"
                f"[TRTLLM-MLA]  k_cache[0,0,:3]: {k_cache[0,0,:3] if k_cache.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  pages[0,0,:3]: {pages[0,0,:3] if pages.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  kv_cache[0,0,0,:3]: {kv_cache[0,0,0,:3] if kv_cache.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  kv_cache[0,1,0,:3]: {kv_cache[0,1,0,:3] if kv_cache.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  block_kv_indices[0,:5]: {metadata.block_kv_indices[0,:5] if metadata.block_kv_indices.numel() > 0 else 'empty'}"
            )

        # ------------- DEBUG (align with FlashInfer-MLA) -------------
        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            q_nope_str = f"{q_nope.shape}  {q_rope.dtype}" if q_nope is not None else "None"
            q_rope_str = f"{q_rope.shape}  {q_rope.dtype}" if q_rope is not None else "None"
            q_nope_sample = f"{q_nope[0,:3,:3] if q_rope is not None and q_nope.numel() > 0 else 'empty'}"
            q_rope_sample = f"{q_rope[0,:3,:3] if q_rope is not None and q_rope.numel() > 0 else 'empty'}"
            
            print(
                f"[TRTLLM-MLA]  q_nope  : {q_nope_str}\n"
                f"[TRTLLM-MLA]  q_rope  : {q_rope_str}\n"
                f"[TRTLLM-MLA]  query   : {query.shape}  {query.dtype}\n"
                f"[TRTLLM-MLA]  kv_cache: {kv_cache.shape}  {kv_cache.dtype}\n"
                f"[TRTLLM-MLA]  scale   : {sm_scale}\n"
                f"[TRTLLM-MLA]  cache_loc: {cache_loc.shape}  {cache_loc.dtype}\n"
                f"[TRTLLM-MLA]  seq_lens: {forward_batch.seq_lens.shape}  {forward_batch.seq_lens.dtype}\n"
                f"[TRTLLM-MLA]  batch_size: {forward_batch.batch_size}\n"
                f"[TRTLLM-MLA]  page_size: {self.page_size}\n"
                f"[TRTLLM-MLA]  qk_nope_head_dim: {self.qk_nope_head_dim}\n"
                f"[TRTLLM-MLA]  qk_rope_head_dim: {self.qk_rope_head_dim}\n"
                f"[TRTLLM-MLA]  kv_lora_rank: {self.kv_lora_rank}\n"
                f"[TRTLLM-MLA]  v_head_dim: {layer.v_head_dim}\n"
                f"[TRTLLM-MLA]  kv_cache_dim: {self.kv_cache_dim}\n"
                f"[TRTLLM-MLA]  tp_q_head_num: {layer.tp_q_head_num}\n"
                f"[TRTLLM-MLA]  tp_k_head_num: {layer.tp_k_head_num}\n"
                f"[TRTLLM-MLA]  layer_id: {layer.layer_id}\n"
                f"[TRTLLM-MLA]  q_nope[0,:3,:3]: {q_nope_sample}\n"
                f"[TRTLLM-MLA]  q_rope[0,:3,:3]: {q_rope_sample}\n"
                f"[TRTLLM-MLA]  query[0,:3,:3]: {query[0,:3,:3] if query.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  seq_lens values: {forward_batch.seq_lens[:min(5, len(forward_batch.seq_lens))]}\n"
                f"[TRTLLM-MLA]  cache_loc values: {cache_loc[:min(5, len(cache_loc))]}"
            )

       
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
            max_seq_len=int(metadata.block_kv_indices.shape[1] * self.page_size), # max_seq_len equals padded_blocks * page_size.
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=sm_scale,
            o_scale=1.0,
        )
        # TRTLLM kernel may return both V and ROPE dims (kv_lora_rank + qk_rope_head_dim).
        # We only need the value projection part (v_head_dim).
        raw_out_v = raw_out[..., : layer.v_head_dim].contiguous()

        # ------------- DEBUG KERNEL OUTPUT -------------
        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            print(
                f"[TRTLLM-MLA]  raw_out   : {raw_out.shape}  {raw_out.dtype}\n"
                f"[TRTLLM-MLA]  raw_out_v : {raw_out_v.shape}  {raw_out_v.dtype}\n"
                f"[TRTLLM-MLA]  raw_out[0,:3,:3]: {raw_out[0,:3,:3] if raw_out.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  raw_out_v[0,:3,:3]: {raw_out_v[0,:3,:3] if raw_out_v.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  raw_out stats: min={raw_out.min():.6f}, max={raw_out.max():.6f}, mean={raw_out.mean():.6f}\n"
                f"[TRTLLM-MLA]  raw_out_v stats: min={raw_out_v.min():.6f}, max={raw_out_v.max():.6f}, mean={raw_out_v.mean():.6f}"
            )

        output = raw_out_v.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        if output.shape[0] > forward_batch.batch_size:
            output = output[: forward_batch.batch_size]
            
        # ------------- DEBUG FINAL OUTPUT -------------
        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            print(
                f"[TRTLLM-MLA]  output    : {output.shape}  {output.dtype}\n"
                f"[TRTLLM-MLA]  output[0,:10]: {output[0,:10] if output.numel() > 0 else 'empty'}\n"
                f"[TRTLLM-MLA]  output[0,10:20]: {output[0,10:20] if output.numel() > 10 else 'empty'}\n"
                f"[TRTLLM-MLA]  output[0,20:30]: {output[0,20:30] if output.numel() > 20 else 'empty'}\n"
                f"[TRTLLM-MLA]  output stats: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}\n"
                f"[TRTLLM-MLA]  output std: {output.std():.6f}\n"
                f"[TRTLLM-MLA]  output_reshaped: {output.shape[0]} -> {forward_batch.batch_size}\n"
                f"[TRTLLM-MLA] ===== END TRTLLM-MLA DEBUG ====="
            )
        
        return output 

    @staticmethod
    def _pad_invalid_blocks(block_kv_indices: torch.Tensor):
        """Replace -1 paddings with the last valid page id for each row.
 
        TRT-LLM treats a leading -1 as "empty sequence" and will skip the
        whole sequence.  For shorter sequences we therefore replicate the
        last real page id into the padded region instead of leaving -1.
        """
        for row in range(block_kv_indices.size(0)):
            row_view = block_kv_indices[row]
            # Find last non-negative entry (every sequence has at least one)
            valid_mask = row_view >= 0
            if not valid_mask.any():
                # Defensive – shouldn’t happen in decode mode
                continue
            last_valid = row_view[valid_mask][-1]
            row_view[~valid_mask] = last_valid
 
        # Extra visibility when debugging
        if os.getenv("SGLANG_DEBUG_MLA", "0") == "1":
            preview_rows = min(6, block_kv_indices.size(0))
            print("[TRTLLM-MLA]  block_kv_indices preview (after padding):")
            for i in range(preview_rows):
                print(f"  seq {i}:", block_kv_indices[i].tolist())