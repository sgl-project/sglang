from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import topk_transform_512
from sglang.srt.environ import envs
from sglang.srt.layers.attention.compressed.metadata import (
    PagedCoreMetadata,
    PagedIndexerMetadata,
)
from sglang.srt.layers.attention.indexer_topk_capturer import (
    get_global_indexer_capturer,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.compressor import CompressorBackend
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import C4Indexer

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

# if is_hip():
if is_fp8_fnuz():
    FP8_DTYPE = torch.float8_e4m3fnuz
    # FP8_MAX = torch.finfo(FP8_DTYPE).max
    FP8_MAX = 224.0
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max

_arange_cache: Dict[str, torch.Tensor] = {}


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Vectorized implementation that avoids .item() and Python loops,
    making it compatible with CUDA graph capture."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_num_pages = page_table.shape[1]
    SCALE_OFFSET = block_size * head_dim
    total_dim = block_size * (head_dim + 4)

    kvcache_flat = kvcache_fp8.view(-1, total_dim)

    pages_clamped = page_table.clamp(min=0)
    kvcache_gathered = kvcache_flat[pages_clamped]  # (B, max_num_pages, total_dim)

    kv_values_raw = kvcache_gathered[..., :SCALE_OFFSET].contiguous()
    kv_values_fp8 = kv_values_raw.view(dtype=FP8_DTYPE)
    kv_values = kv_values_fp8.to(torch.float32)
    kv_values = kv_values.reshape(batch_size, max_num_pages * block_size, head_dim)

    kv_scales_raw = kvcache_gathered[..., SCALE_OFFSET:].contiguous()
    kv_scales = kv_scales_raw.view(dtype=torch.float32)
    kv_scales = kv_scales.reshape(batch_size, max_num_pages * block_size)

    q_float = q_fp8[:, 0].to(torch.float32)  # (B, num_heads, head_dim)
    # (B, padded_seq_len, head_dim) @ (B, head_dim, num_heads) -> (B, padded_seq_len, num_heads)
    scores = torch.bmm(kv_values, q_float.transpose(1, 2))
    scores = F.relu(scores)
    scores = scores * weight.unsqueeze(1)  # (B, padded_seq_len, num_heads)
    scores = scores.sum(dim=2)  # (B, padded_seq_len)
    scores = scores * kv_scales  # (B, padded_seq_len)

    padded_seq_len = max_num_pages * block_size
    cache = _arange_cache
    arange_key = f"arange_{padded_seq_len}_{scores.device}"
    if arange_key not in cache:
        cache[arange_key] = torch.arange(padded_seq_len, device=scores.device)
    positions = cache[arange_key].unsqueeze(0)
    valid_mask = positions < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid_mask, 0.0)

    # Pad to max_seq_len if needed (padded_seq_len may be < max_seq_len)
    if padded_seq_len < max_seq_len:
        scores = F.pad(scores, (0, max_seq_len - padded_seq_len), value=0.0)
    else:
        scores = scores[:, :max_seq_len]

    return scores


# def fp8_paged_mqa_logits_torch(
#     q_fp8: torch.Tensor,
#     kvcache_fp8: torch.Tensor,
#     weight: torch.Tensor,
#     seq_lens: torch.Tensor,
#     page_table: torch.Tensor,
#     deep_gemm_metadata: Any,
#     max_seq_len: int,
#     clean_logits: bool = True,
# ) -> torch.Tensor:
#     """
#     Vectorized PyTorch implementation of fp8_paged_mqa_logits.
#     Processes all batches in parallel without Python for loops.
#     """
#     _ = deep_gemm_metadata
#     batch_size, _, num_heads, head_dim = q_fp8.shape
#     block_size = kvcache_fp8.shape[1]
#     device = q_fp8.device

#     assert head_dim == 128, "TODO"
#     assert block_size == 64, "TODO"
#     assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
#     assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
#     assert weight.shape == (batch_size, num_heads)
#     assert seq_lens.shape == (batch_size,)
#     assert page_table.shape[0] == batch_size
#     assert clean_logits == False

#     # Prepare q: (batch_size, num_heads, head_dim)
#     q = q_fp8[:, 0].to(torch.float32)  # (batch_size, num_heads, head_dim)

#     # Calculate number of pages per batch element
#     num_pages_per_batch = (seq_lens + block_size - 1) // block_size  # (batch_size,)
#     max_num_pages = int(
#         num_pages_per_batch.max().item()
#     )  # Single sync, outside main computation

#     # Padded seq len for each batch
#     padded_seq_lens = num_pages_per_batch * block_size  # (batch_size,)
#     max_padded_seq_len = max_num_pages * block_size

#     # Reshape kvcache for gathering
#     # Original: (num_blocks_total, block_size, 1, head_dim + 4)
#     # Reshape to: (num_blocks_total, block_size * (head_dim + 4))
#     kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))

#     # Gather pages for all batches: page_table[:, :max_num_pages]
#     # Shape: (batch_size, max_num_pages)
#     pages = page_table[:, :max_num_pages]

#     # Gather kvcache for all batches
#     # Shape: (batch_size, max_num_pages, block_size * (head_dim + 4))
#     gathered_kvcache = kvcache_flat[pages]

#     # Split into values and scales
#     SCALE_OFFSET = block_size * head_dim
#     # Shape: (batch_size, max_num_pages, block_size * head_dim)
#     kvcache_value_flat = gathered_kvcache[..., :SCALE_OFFSET]
#     # Shape: (batch_size, max_num_pages, block_size * 4) -> scales are 4 bytes per position
#     kvcache_scale_flat = gathered_kvcache[..., SCALE_OFFSET:]

#     # Convert FP8 values to float32
#     kvcache_value_fp8 = kvcache_value_flat.view(dtype=FP8_DTYPE)
#     kvcache_value = kvcache_value_fp8.to(torch.float32)
#     # Reshape to (batch_size, max_padded_seq_len, head_dim)
#     kvcache_value = kvcache_value.view(batch_size, max_padded_seq_len, head_dim)

#     # Convert scales to float32
#     kvcache_scale = kvcache_scale_flat.view(dtype=torch.float32)
#     # Reshape to (batch_size, max_padded_seq_len)
#     kvcache_scale = kvcache_scale.reshape(batch_size, max_padded_seq_len)

#     # Compute attention scores: kvcache_value @ q^T
#     # kvcache_value: (batch_size, max_padded_seq_len, head_dim)
#     # q: (batch_size, num_heads, head_dim)
#     # score: (batch_size, max_padded_seq_len, num_heads)
#     score = torch.bmm(kvcache_value, q.transpose(1, 2))

#     # Apply ReLU
#     score = F.relu(score)

#     # Multiply by weight (q_scale): (batch_size, num_heads)
#     # score: (batch_size, max_padded_seq_len, num_heads)
#     score = score * weight.unsqueeze(1)

#     # Sum over heads: (batch_size, max_padded_seq_len)
#     score = score.sum(dim=2)

#     # Multiply by kvcache_scale: (batch_size, max_padded_seq_len)
#     score = score * kvcache_scale

#     # Create output logits with proper masking
#     logits = torch.full(
#         (batch_size, max_seq_len),
#         float("-inf"),  # or 0.0 depending on requirements
#         dtype=torch.float32,
#         device=device,
#     )

#     # Create position indices for masking
#     positions = torch.arange(max_seq_len, device=device).unsqueeze(
#         0
#     )  # (1, max_seq_len)
#     valid_mask = positions < seq_lens.unsqueeze(1)  # (batch_size, max_seq_len)

#     # Copy valid scores to logits
#     # We need to handle the case where max_padded_seq_len might differ from max_seq_len
#     copy_len = min(max_padded_seq_len, max_seq_len)
#     logits[:, :copy_len] = torch.where(
#         valid_mask[:, :copy_len], score[:, :copy_len], logits[:, :copy_len]
#     )

#     return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """
    Vectorized PyTorch fallback for topk_transform_512.
    All helper tensors (arange, zeros) are cached to avoid device-tensor
    creation during HIP/CUDA graph capture.
    """

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    # ---- cached helper tensors (allocated once, reused on replay) ----
    cache = _arange_cache
    key_seq = f"arange_{max_seq_len}_{device}"
    key_topk = f"arange_{TOPK}_{device}"
    key_bs = f"arange_{batch_size}_{device}"
    if key_seq not in cache:
        cache[key_seq] = torch.arange(max_seq_len, device=device)
    if key_topk not in cache:
        cache[key_topk] = torch.arange(TOPK, device=device, dtype=torch.int32)
    if key_bs not in cache:
        cache[key_bs] = torch.arange(batch_size, device=device)

    positions = cache[key_seq].unsqueeze(0).expand(batch_size, -1)
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores.masked_fill_(~valid_mask, float("-inf"))

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        raw_indices = F.pad(raw_indices, (0, TOPK - actual_k), value=0)

    batch_indices = cache[key_bs].unsqueeze(1).expand(-1, TOPK)
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = cache[key_topk].unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    sequential_indices = cache[key_topk].unsqueeze(0).expand(batch_size, -1)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

    seq_indices_or_neg1 = sequential_indices.clone()
    seq_indices_or_neg1.masked_fill_(~sequential_valid, -1)

    needs_seq_mask = needs_sequential.unsqueeze(1).expand(-1, TOPK)
    raw_indices = torch.where(needs_seq_mask, seq_indices_or_neg1, raw_indices)
    valid_topk = torch.where(needs_seq_mask, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices.masked_fill_(~valid_topk, -1)

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = raw_indices.clone()
        raw_indices.masked_fill_(~valid_topk, -1)
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,  # [B, H]
    q_scale_ptr,  # [B, H, 1]
    out_ptr,  # [B, H, 1]
    numel,  # B * H
    out_scale,  # scalar
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    # Compute in fp32 for better numerical stability, then cast back.
    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Triton version of:
        weight.unsqueeze(-1) * out_scale * q_scale

    Args:
        weight:  [B, H], contiguous
        q_scale: [B, H, 1], contiguous
        out_scale: Python float / scalar

    Returns:
        out: [B, H, 1]
    """
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DeepseekV4Metadata
        self.debug_use_external_c4_sparse_indices: bool = False

        # this method should be type method
        # see srt/layers/attention/compressed/compressor.py

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        x_for_compressor: Optional[torch.Tensor] = None,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        # main stream
        self.forward_indexer_compressor(
            x=(
                x_for_compressor
                if (is_nsa_enable_prefill_cp() and x_for_compressor is not None)
                else x
            ),
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        # alt stream 0: compute q
        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

        # alt stream 1: compute weights
        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            stream_weights.wait_event(q_scale_ready)
            weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)

        current_stream.wait_stream(stream_q)
        current_stream.wait_stream(stream_weights)

        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        x_for_compressor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        q_fp8, q_scale = act_quant(q)
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        self.forward_indexer_compressor(
            x=(
                x_for_compressor
                if (is_nsa_enable_prefill_cp() and x_for_compressor is not None)
                else x
            ),
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        return q_fp8, weights, c4_indexer_kv_cache

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        x_for_compressor: Optional[torch.Tensor] = None,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackend)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
            DSV4AttnMetadataRadix,
        )

        assert isinstance(core_metadata, (PagedCoreMetadata, DSV4AttnMetadataRadix))
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        _x_comp = (
            x_for_compressor
            if (is_nsa_enable_prefill_cp() and x_for_compressor is not None)
            else x
        )
        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                x_for_compressor=_x_comp,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                x_for_compressor=_x_comp,
            )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132

        # DeepGEMM#280 does not change test_attention.py for fp8_paged_mqa_logits, thus
        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        # CUDA path: use deep_gemm
        if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        # elif is_hip():
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            fn = fp8_paged_mqa_logits_torch
        else:
            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    fp8_paged_mqa_logits_chunked as fn,
                )
            else:
                from deep_gemm import fp8_paged_mqa_logits as fn

        logits = fn(
            q_fp8,
            c4_indexer_kv_cache,
            weights,
            indexer_metadata.c4_seq_lens,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_seq_len,
            False,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return  # skip updating page indices

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer.is_enabled()

        raw_indices = None
        if capture_enabled or forward_batch.hisparse_coordinator is not None:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        else:
            topk_transform_512(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )

        if forward_batch.hisparse_coordinator is not None:
            if forward_batch.forward_mode.is_decode():
                # todo hisparse: to coordinate with kernel signature
                core_metadata.c4_sparse_page_indices = (
                    forward_batch.hisparse_coordinator.get_front_topk_tokens(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        raw_indices,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = token_to_kv_pool.c4_kv_pool.translate_loc_from_compressed_to_hisparse_device(
                    core_metadata.c4_sparse_page_indices
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)
