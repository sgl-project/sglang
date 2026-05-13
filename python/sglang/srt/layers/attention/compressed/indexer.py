from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import topk_transform_512, topk_transform_512_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.compressed.metadata import (
    PagedCoreMetadata,
    PagedIndexerMetadata,
    _is_sm120,
)
from sglang.srt.layers.attention.indexer_topk_capturer import (
    get_global_indexer_capturer,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.utils import is_hip, is_xpu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.compressor import CompressorBackend
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import C4Indexer

__is_xpu = is_xpu()

if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
    FP8_MAX = torch.finfo(FP8_DTYPE).max
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max


# Bound peak memory of the vectorized fallback.  For each row we materialize
#   chunk_size * page_table.shape[1] * block_size * head_dim bytes of fp32 KV
# plus the gathered raw bytes, plus chunk_size * padded_seq_len * num_heads
# fp32 scores.  256 MiB is plenty for typical (B, P) shapes while keeping
# launch overhead low on XPU.
_FP8_PAGED_MQA_LOGITS_CHUNK_BYTES = 256 * 1024 * 1024


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
    """Vectorized pure-PyTorch fallback for fp8_paged_mqa_logits.

    The original reference implementation looped in Python over ``batch_size``
    and called ``int(seq_lens[i].item())`` per iteration, forcing a device
    sync per query token per layer.  For prefill on DeepSeek V4 this is the
    dominant cost on XPU since ``batch_size == num_prefill_tokens`` and the
    indexer runs once per layer (e.g. 768 tokens * 43 layers = 33k host
    syncs and 33k tiny F.linear launches).

    This rewrite gathers the full KV slab once per chunk, computes scores in
    a single batched matmul, and masks invalid positions with the device-side
    ``seq_lens`` so no host sync is needed.  Out-of-range page-table entries
    are clamped to a valid index so the gather can never read past the pool;
    the corresponding scores are zeroed out by the position mask.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    device = q_fp8.device
    head_dim_with_sf = head_dim + 4
    SCALE_OFFSET = block_size * head_dim

    # Cap pages used per row to what fits into ``max_seq_len``; this is also
    # the largest valid index any row will ever access.
    max_pages_eff = (max_seq_len + block_size - 1) // block_size
    P = min(page_table.shape[1], max_pages_eff)
    padded_seq_len = P * block_size
    # Pad result to ``max_seq_len`` so callers see the documented shape.
    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)

    # Flatten the paged byte slab once: (num_pages_total, block_size*hd_with_sf)
    kv_flat = kvcache_fp8.reshape(-1, block_size * head_dim_with_sf)
    num_pages_total = kv_flat.shape[0]

    # Per-row valid mask is computed against absolute token positions.
    pos = torch.arange(padded_seq_len, device=device)

    # Pick a per-chunk batch size that keeps the gathered fp32 KV slab within
    # the configured byte budget.  4 bytes/elem * P * block_size * head_dim.
    bytes_per_row = max(1, P * block_size * head_dim * 4)
    chunk_size = max(1, _FP8_PAGED_MQA_LOGITS_CHUNK_BYTES // bytes_per_row)

    # Pre-clamp page table so the gather is always in-bounds.  Out-of-range
    # entries (rows past their seq_len, or padding) are mapped to page 0; the
    # corresponding score positions are zeroed by the seq-len mask below.
    pt = page_table[:, :P]
    if num_pages_total > 0:
        pt = pt.clamp_(min=0, max=num_pages_total - 1)

    for s in range(0, batch_size, chunk_size):
        e = min(s + chunk_size, batch_size)
        cb = e - s

        # (cb, P, block_size*hd_with_sf) bytes
        kv = kv_flat[pt[s:e]]
        # Split off value/scale halves and make them contiguous so view-as-dtype
        # is legal after slicing the trailing dim.
        kv_value_b = kv[..., :SCALE_OFFSET].contiguous()
        kv_scale_b = kv[..., SCALE_OFFSET:].contiguous()

        # bytes -> fp8 -> fp32, shape (cb, padded_seq_len, head_dim)
        kv_value = (
            kv_value_b.view(dtype=FP8_DTYPE)
            .view(cb, padded_seq_len, head_dim)
            .to(torch.float32)
        )
        # bytes -> fp32 scale per token, shape (cb, padded_seq_len)
        kv_scale = kv_scale_b.view(dtype=torch.float32).view(cb, padded_seq_len)

        # q: (cb, num_heads, head_dim) fp32
        q = q_fp8[s:e, 0].to(torch.float32)

        # score: (cb, padded_seq_len, num_heads)
        score = torch.einsum("bsd,bhd->bsh", kv_value, q)
        score = torch.relu(score)
        score = score * weight[s:e].unsqueeze(1)
        score = score.sum(dim=2)
        score = score * kv_scale

        # Mask positions outside [0, seq_len_i) to 0.
        valid = pos.unsqueeze(0) < seq_lens[s:e].unsqueeze(1)
        score = torch.where(valid, score, score.new_zeros(()))

        # Write back; truncate or pad to max_seq_len.
        write_len = min(padded_seq_len, max_seq_len)
        logits[s:e, :write_len] = score[:, :write_len]
        if write_len < max_seq_len:
            logits[s:e, write_len:] = 0

    return logits


_NEG_INF_I32_CACHE: dict = {}
_NEG_ONE_I32_CACHE: dict = {}


def _neg_inf_scalar(device: torch.device) -> torch.Tensor:
    """Cached scalar -inf fp32 tensor per device."""
    key = str(device)
    t = _NEG_INF_I32_CACHE.get(key)
    if t is None:
        t = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
        _NEG_INF_I32_CACHE[key] = t
    return t


def _neg_one_i32_scalar(device: torch.device) -> torch.Tensor:
    """Cached scalar -1 int32 tensor per device."""
    key = str(device)
    t = _NEG_ONE_I32_CACHE.get(key)
    if t is None:
        t = torch.tensor(-1, device=device, dtype=torch.int32)
        _NEG_ONE_I32_CACHE[key] = t
    return t


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    # Per-device cached scalar constants — constructing torch.tensor(...) on
    # the device path here would otherwise force an H2D copy + sync per call
    # (this function runs per-layer per prefill chunk).
    neg_inf = _neg_inf_scalar(device)
    neg_one_i32 = _neg_one_i32_scalar(device)

    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)  # (1, S)
    valid_mask = positions < seq_lens.unsqueeze(1)                     # (B, S)

    # NOTE: avoid `masked_scores[~valid_mask] = float("-inf")` — boolean-
    # indexed scatter writes are a known L0 sync hot spot on Intel XPU.
    masked_scores = torch.where(valid_mask, scores, neg_inf)

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    # Gather along dim=1 with a single int64 gather kernel — significantly
    # cheaper than the 2-D advanced index `scores[batch_idx.flatten(),
    # raw.flatten()].view(...)` pattern, which materializes a flat int64
    # index tensor of size B*TOPK for every call.
    gather_idx = raw_indices.clamp(min=0).to(torch.long)
    gathered_scores = torch.gather(scores, dim=1, index=gather_idx)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    # Always run the sequential override (cheap when no row hits it).
    # Previously this was guarded by `if needs_sequential.any():` which
    # forced a D2H sync per call on XPU.
    needs_sequential = (seq_lens <= TOPK).unsqueeze(1)  # (B, 1) bool
    sequential_indices = torch.arange(
        TOPK, device=device, dtype=torch.int32
    ).unsqueeze(0)                                       # (1, TOPK)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1).to(
        sequential_indices.dtype
    )                                                    # (B, TOPK)

    raw_indices = torch.where(
        needs_sequential,
        torch.where(sequential_valid, sequential_indices, neg_one_i32),
        raw_indices,
    )
    valid_topk = torch.where(needs_sequential, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)

    page_indices = torch.where(valid_topk, page_indices, neg_one_i32)

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(valid_topk, raw_indices, neg_one_i32)
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
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

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
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

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        q_fp8, q_scale = act_quant(q)
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        self.forward_indexer_compressor(
            x=x,
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
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: this runs from MQALayer._forward_prepare,
        # before attn_backend.forward() would trigger the upgrade.
        self._maybe_upgrade_forward_metadata()
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

        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
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
            )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132

        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get() or _is_sm120 or _is_xpu:
            fn = fp8_paged_mqa_logits_torch
        else:
            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    fp8_paged_mqa_logits_chunked as fn,
                )
            else:
                from deep_gemm import fp8_paged_mqa_logits as fn

        _c4sl = indexer_metadata.c4_seq_lens
        if _c4sl.dim() == 1:
            _c4sl = _c4sl.unsqueeze(-1)
        logits = fn(
            q_fp8,
            c4_indexer_kv_cache,
            weights,
            _c4sl,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_c4_seq_len,
            False,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer.is_enabled()

        hisparse_coordinator = forward_batch.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : core_metadata.c4_sparse_page_indices.size(0)
            ]

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform_512_v2(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
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
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
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
