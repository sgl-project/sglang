from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.dsv4 import (
    fused_q_indexer_rope_hadamard_fp4_quant,
    fused_q_indexer_rope_hadamard_quant,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.utils import add_prefix, is_cuda, is_hip
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


IndexerQuery: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


_arange_cache = {}


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
    """Vectorized implementation compatible with CUDA graph capture."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128
    assert block_size == 64
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
    kvcache_gathered = kvcache_flat[pages_clamped]

    kv_values_raw = kvcache_gathered[..., :SCALE_OFFSET].contiguous()
    kv_values_fp8 = kv_values_raw.view(dtype=FP8_DTYPE)
    kv_values = kv_values_fp8.to(torch.float32)
    kv_values = kv_values.reshape(batch_size, max_num_pages * block_size, head_dim)

    kv_scales_raw = kvcache_gathered[..., SCALE_OFFSET:].contiguous()
    kv_scales = kv_scales_raw.view(dtype=torch.float32)
    kv_scales = kv_scales.reshape(batch_size, max_num_pages * block_size)

    q_float = q_fp8[:, 0].to(torch.float32)
    scores = torch.bmm(kv_values, q_float.transpose(1, 2))
    scores = F.relu(scores)
    scores = scores * weight.unsqueeze(1)
    scores = scores.sum(dim=2)
    scores = scores * kv_scales

    padded_seq_len = max_num_pages * block_size
    cache = _arange_cache
    arange_key = f"arange_{padded_seq_len}_{scores.device}"
    if arange_key not in cache:
        cache[arange_key] = torch.arange(padded_seq_len, device=scores.device)
    positions = cache[arange_key].unsqueeze(0)
    valid_mask = positions < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid_mask, 0.0)

    if padded_seq_len < max_seq_len:
        scores = F.pad(scores, (0, max_seq_len - padded_seq_len), value=0.0)
    else:
        scores = scores[:, :max_seq_len]

    return scores


def _aiter_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Wrapper adapting aiter's deepgemm_fp8_paged_mqa_logits to SGLang's interface."""
    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
    )

    batch_size = q_fp8.shape[0]
    next_n = q_fp8.shape[1]
    total_tokens = batch_size * next_n
    _sl = seq_lens.squeeze(-1) if seq_lens.dim() == 2 else seq_lens
    kv_block_size = kvcache_fp8.shape[1]
    logits = torch.empty(
        total_tokens,
        max_seq_len,
        dtype=torch.float32,
        device=q_fp8.device,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kvcache_fp8,
        weight,
        logits,
        _sl.to(torch.int32),
        page_table.to(torch.int32),
        max_seq_len,
        KVBlockSize=kv_block_size,
        Preshuffle=True,
    )
    return logits


def fp8_paged_mqa_logits_torch_sm120(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """CUDA-graph-compatible FP8 paged MQA logits for SM120 (vectorized, no .item())."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    assert head_dim == 128, "Vectorized torch impl hardcodes DSV4 indexer head_dim=128"
    assert (
        block_size == 64
    ), "Vectorized torch impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_padded_seq = max_pages * block_size

    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    SCALE_OFFSET = block_size * head_dim

    page_ids = page_table[:, :max_pages]
    kvcache_gathered = kvcache_flat[page_ids]

    kv_value_raw = kvcache_gathered[..., :SCALE_OFFSET]
    kv_scale_raw = kvcache_gathered[..., SCALE_OFFSET:]

    kv_value = kv_value_raw.contiguous().view(dtype=FP8_DTYPE).to(torch.float32)
    kv_value = kv_value.view(batch_size, max_padded_seq, head_dim)

    kv_scale = kv_scale_raw.contiguous().view(dtype=torch.float32)
    kv_scale = kv_scale.view(batch_size, max_padded_seq)

    q = q_fp8[:, 0].to(torch.float32)

    score = torch.bmm(kv_value, q.transpose(1, 2))

    score = F.relu(score)
    score = score * weight.unsqueeze(1)
    score = score.sum(dim=2)

    score = score * kv_scale

    out_width = min(max_padded_seq, max_seq_len)
    logits = score.new_full((batch_size, max_seq_len), float("-inf"))
    logits[:, :out_width] = score[:, :out_width]

    positions = torch.arange(max_seq_len, device=device)
    invalid_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    logits.masked_fill_(invalid_mask, float("-inf"))

    return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Vectorized PyTorch fallback for topk_transform_512.
    All helper tensors (arange, zeros) are cached to avoid device-tensor
    creation during HIP/CUDA graph capture."""

    TOPK = out_page_indices.shape[1]
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

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


def _dcp_indexer_topk_enabled() -> bool:
    """Opt-in distributed (DCP) sharding of the C4 indexer top-k selection.

    The HIP ``deepseek_v4_topk_transform_kernel`` launches a single block per
    batch row, so at small decode batch the O(c4_seq_len) radix-select runs on a
    single CU and dominates context-length scaling (profiled: ~7.7ms/step of the
    +8ms/step at 1M). When enabled (and dcp_size>1, decode, long context), the
    candidate axis is sharded across the DCP group and merged, giving ~dcp-fold
    parallelism. Default off -> baseline byte-for-byte unchanged."""
    return os.environ.get("SGLANG_DSV4_DCP_INDEXER_TOPK", "0") not in (
        "0",
        "",
        "false",
        "False",
    )


# Minimum indexer (c4) sequence length before the distributed top-k path is
# worth the per-layer all-gather latency; below this we keep the local kernel.
_DCP_TOPK_MIN_C4_LEN = int(os.environ.get("SGLANG_DSV4_DCP_INDEXER_TOPK_MIN", "16384"))


def topk_transform_512_dcp(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    dcp_group,
    dcp_size: int,
    dcp_rank: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Distributed (DCP) equivalent of ``topk_transform_512``.

    Shards the candidate (score) axis across the DCP group: rank ``r`` selects
    the local top-k over its contiguous ``1/dcp`` slice of each score row, the
    group all-gathers ``(score, global_token_index)`` pairs, and every rank
    re-selects the global top-k from the ``dcp*k`` union. Each globally-top-k
    token is also top-k within its own slice, so the union always contains the
    exact global top-k -> the selected page set is identical to the non-sharded
    kernel (output order may differ; downstream sparse attention is invariant to
    page order). All shapes are static and helper tensors cached, so the path is
    CUDA-graph capturable (mirrors the existing DCP attention all-gather)."""

    TOPK = out_page_indices.shape[1]
    B = scores.shape[0]
    N = scores.shape[1]
    device = scores.device
    neg_inf = float("-inf")

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    chunk = (N + dcp_size - 1) // dcp_size
    padded = chunk * dcp_size
    base = dcp_rank * chunk

    cache = _arange_cache
    key_chunk = f"arange_{chunk}_{device}"
    if key_chunk not in cache:
        cache[key_chunk] = torch.arange(chunk, device=device)
    # Global token index of each position in this rank's slice: [1, chunk].
    local_pos = (cache[key_chunk] + base).unsqueeze(0)

    if padded != N:
        scores_p = F.pad(scores, (0, padded - N), value=neg_inf)
    else:
        scores_p = scores.contiguous()
    local = scores_p.view(B, dcp_size, chunk)[:, dcp_rank, :]  # [B, chunk]

    if seq_lens.dim() > 1:
        _sl = seq_lens.view(B, -1)[:, 0:1]
    else:
        _sl = seq_lens.view(B, 1)
    valid = local_pos < _sl  # [B, chunk]
    local = torch.where(valid, local, torch.full_like(local, neg_inf))

    k = min(TOPK, chunk)
    local_scores, local_idx = torch.topk(local, k, dim=1, largest=True, sorted=False)
    global_raw = local_idx.to(torch.int32) + base  # [B, k] global token idx
    if k < TOPK:
        pad = TOPK - k
        local_scores = F.pad(local_scores, (0, pad), value=neg_inf)
        global_raw = F.pad(global_raw, (0, pad), value=0)

    # Gather every rank's local top-k -> [B, dcp*TOPK], rank-ordered.
    g_scores = dcp_group.all_gather(local_scores.contiguous(), dim=1)
    g_raw = dcp_group.all_gather(global_raw.contiguous(), dim=1)

    m_scores, m_pos = torch.topk(g_scores, TOPK, dim=1, largest=True, sorted=False)
    final_raw = torch.gather(g_raw, 1, m_pos)  # [B, TOPK] global token idx (int32)
    final_valid = m_scores != neg_inf

    raw_i = final_raw.clamp(min=0)
    page_idx = raw_i >> page_bits
    offset = raw_i & page_mask
    physical = torch.gather(page_tables, 1, page_idx.to(torch.int64))
    page_indices = ((physical << page_bits) | offset).to(torch.int32)
    page_indices = torch.where(
        final_valid, page_indices, torch.full_like(page_indices, -1)
    )
    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        rr = torch.where(final_valid, final_raw, torch.full_like(final_raw, -1))
        out_raw_indices.copy_(rr.to(out_raw_indices.dtype))


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


class C4IndexerBackendMixin:
    def __init__(self):
        super().__init__()
        self.debug_use_external_c4_sparse_indices: bool = False

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

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

        # The weight projection is small and fast; compute it on its own
        # stream, then have the Q stream wait on it before launching the big
        # fused Q kernel (which folds rope, hadamard, quantization, and
        # weight scaling into one pass).
        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            weights_ready = stream_weights.record_event()

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            stream_q.wait_event(weights_ready)
            q, weights = c4_indexer.compute_q(q_lora, positions, weights)

        current_stream.wait_stream(stream_q)
        return q, weights

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_compressor: bool = False,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        weights = c4_indexer.compute_weights(x, skip_scale=True)
        q, weights = c4_indexer.compute_q(q_lora, positions, weights)
        if not skip_compressor:
            self.forward_indexer_compressor(
                x=x,
                forward_batch=forward_batch,
                layer_id=c4_indexer.layer_id,
                compressor=c4_indexer.compressor,
            )
        return q, weights

    def _can_use_nonpaged_indexer(
        self,
        *,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
    ) -> bool:
        if not envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.get():
            return False
        # This path calls CUDA DeepGEMM and assumes the CUDA FP8+FP32 packed
        # indexer cache layout. Explicitly reject HIP, NPU, and other devices.
        if not is_cuda() or is_hip():
            return False
        # The gather plan is built from eager, child-local ForwardBatch metadata.
        # Rewritten, TBO-split, and graph-backed batches must use the paged path.
        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or forward_batch._original_forward_mode is not None
            or forward_batch.tbo_parent_token_range is not None
            or forward_batch.batch_size != 1
            or indexer_metadata.use_prefill_cuda_graph
        ):
            return False
        if (
            c4_indexer.use_fp4_indexer
            or envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
            or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ):
            return False
        if (
            get_parallel().attn_cp_size != 1
            or self.hisparse_coordinator is not None
            or is_in_tc_piecewise_cuda_graph()
            or is_in_breakable_cuda_graph()
        ):
            return False
        return not torch.cuda.is_current_stream_capturing()

    def _get_nonpaged_indexer_plan(
        self,
        *,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
        page_table: torch.Tensor,
        c4_seq_lens: torch.Tensor,
        query_rows: int,
    ) -> Optional[NonPagedIndexerPlan]:
        if query_rows < envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.get():
            return None
        if not self._can_use_nonpaged_indexer(
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=indexer_metadata,
        ):
            return None
        if indexer_metadata.nonpaged_plan is not None:
            return indexer_metadata.nonpaged_plan

        if (
            forward_batch.seq_lens is None
            or forward_batch.seq_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_seq_lens is None
            or forward_batch.extend_start_loc is None
            or forward_batch.extend_num_tokens is None
        ):
            return None

        def to_cpu_int_list(values) -> Optional[List[int]]:
            if isinstance(values, torch.Tensor):
                if values.device.type != "cpu":
                    return None
                values = values.tolist()
            return [int(value) for value in values]

        extend_lens_cpu = to_cpu_int_list(forward_batch.extend_seq_lens_cpu)
        seq_lens_cpu = to_cpu_int_list(forward_batch.seq_lens_cpu)
        if (
            extend_lens_cpu is None
            or seq_lens_cpu is None
            or len(extend_lens_cpu) != 1
            or len(seq_lens_cpu) != 1
            or extend_lens_cpu[0] <= 0
        ):
            return None

        actual_queries = extend_lens_cpu[0]
        if (
            actual_queries != query_rows
            or int(forward_batch.extend_num_tokens) != query_rows
            or forward_batch.seq_lens.numel() != 1
            or forward_batch.extend_seq_lens.numel() != 1
            or forward_batch.extend_start_loc.numel() != 1
            or page_table.dim() != 2
            or page_table.shape[0] < query_rows
            or c4_seq_lens.numel() < query_rows
        ):
            return None

        final_c4_len = seq_lens_cpu[0] // 4
        if final_c4_len <= 0:
            return None

        request_page_table = page_table[:1].contiguous()
        ke = c4_seq_lens[:query_rows].reshape(-1).to(torch.int32).contiguous()
        gather_seq_lens = ke[-1:]
        ks = torch.zeros_like(ke)
        c4_page_size = indexer_metadata.c4_page_size
        max_seqlen_k = (final_c4_len + c4_page_size - 1) // c4_page_size * c4_page_size
        plan = NonPagedIndexerPlan(
            page_table=request_page_table,
            gather_seq_lens=gather_seq_lens,
            ks=ks,
            ke=ke,
            seq_len_sum=final_c4_len,
            max_seq_len=final_c4_len,
            max_seqlen_k=max_seqlen_k,
            query_rows=query_rows,
        )
        indexer_metadata.nonpaged_plan = plan
        return plan

    @staticmethod
    def _forward_nonpaged_indexer(
        *,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        c4_indexer: C4Indexer,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        plan: NonPagedIndexerPlan,
    ) -> torch.Tensor:
        import deep_gemm

        k_u8, scale_u8 = token_to_kv_pool.get_index_k_scale_buffer(
            layer_id=c4_indexer.layer_id,
            seq_len_tensor=plan.gather_seq_lens,
            page_indices=plan.page_table,
            seq_len_sum=plan.seq_len_sum,
            max_seq_len=plan.max_seq_len,
        )
        k_fp8 = k_u8.view(FP8_DTYPE)
        k_scale = scale_u8.view(torch.float32).squeeze(-1)
        return deep_gemm.fp8_mqa_logits(
            q_indexer[: plan.query_rows],
            (k_fp8, k_scale),
            weights[: plan.query_rows],
            plan.ks,
            plan.ke,
            clean_logits=False,
            max_seqlen_k=plan.max_seqlen_k,
        )

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
        skip_compressor: bool = False,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = self.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackendMixin)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        positions = core_metadata.positions
        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        if enable_multi_stream:
            q_indexer, weights = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_indexer, weights = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                skip_compressor=skip_compressor,
            )

        use_fp4_indexer = c4_indexer.use_fp4_indexer

        if use_fp4_indexer:
            q_fp4, q_sf = q_indexer
            assert len(q_fp4.shape) == 3
            assert len(q_sf.shape) == 2
            q = (q_fp4.unsqueeze(1), q_sf.unsqueeze(1))
        else:
            assert len(q_indexer.shape) == 3
            q = q_indexer.unsqueeze(1)

        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if use_fp4_indexer:
            weights = weights.float()
            if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
                raise RuntimeError("DeepSeek V4 FP4 indexer requires DeepGEMM indexer.")
            from deep_gemm import fp8_fp4_paged_mqa_logits as fn
        elif envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.dsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_OPT_USE_AITER_INDEXER.get():
            fn = _aiter_fp8_paged_mqa_logits
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            if is_sm120_supported():
                fn = fp8_paged_mqa_logits_torch_sm120
            else:
                fn = fp8_paged_mqa_logits_torch
        else:
            from deep_gemm import fp8_paged_mqa_logits as fn

        query_rows = q_indexer[0].shape[0] if use_fp4_indexer else q_indexer.shape[0]

        def match_num_queries(tensor: torch.Tensor, value: int) -> torch.Tensor:
            if tensor.shape[0] == query_rows:
                return tensor
            if tensor.shape[0] > query_rows:
                return tensor[:query_rows]
            pad = (0, 0) * (tensor.dim() - 1) + (0, query_rows - tensor.shape[0])
            return F.pad(tensor, pad, value=value)

        c4_seq_lens = match_num_queries(indexer_metadata.c4_seq_lens, value=1)
        _c4sl = c4_seq_lens
        page_table = match_num_queries(indexer_metadata.page_table, value=0)
        c4_sparse_page_indices = match_num_queries(
            core_metadata.c4_sparse_page_indices, value=-1
        )
        _use_tilelang = (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.get() and not use_fp4_indexer
        )
        _use_aiter = envs.SGLANG_OPT_USE_AITER_INDEXER.get() and not use_fp4_indexer
        if _c4sl.dim() == 1 and not _use_tilelang and not _use_aiter:
            _c4sl = _c4sl.unsqueeze(-1)
        nonpaged_plan = self._get_nonpaged_indexer_plan(
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=indexer_metadata,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
            query_rows=query_rows,
        )
        if nonpaged_plan is not None:
            assert isinstance(q_indexer, torch.Tensor)
            logits = self._forward_nonpaged_indexer(
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                plan=nonpaged_plan,
            )
        else:
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id,
            )
            assert c4_indexer_kv_cache.dim() == 2
            head_dim_with_sf = 68 if use_fp4_indexer else 132
            c4_indexer_kv_cache = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0], 64, 1, head_dim_with_sf
            )
            logits = fn(
                q,
                c4_indexer_kv_cache,
                weights,
                _c4sl,
                page_table,
                indexer_metadata.deep_gemm_metadata,
                indexer_metadata.max_c4_seq_len,
                False,
            )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = self.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : c4_sparse_page_indices.size(0)
            ]
        elif core_metadata.c4_sparse_raw_indices is not None:
            raw_indices = core_metadata.c4_sparse_raw_indices

        use_dcp_topk = (
            _dcp_indexer_topk_enabled()
            and getattr(self, "dcp_size", 1) > 1
            and forward_batch.forward_mode.is_decode()
            and indexer_metadata.max_c4_seq_len > _DCP_TOPK_MIN_C4_LEN
        )
        if use_dcp_topk:
            topk_transform_512_dcp(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                self._get_dcp_group(),
                int(self.dcp_size),
                int(self.dcp_rank),
                raw_indices,
            )
        elif envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            topk_transform_512_pytorch_vectorized(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform_512_v2(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            topk_transform_512(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
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
                # flash_mla C4 attention requires int32 page indices.
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    ).to(torch.int32)
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        freqs_cis: torch.Tensor,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        rotary_emb=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.n_local_heads = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
            rotary_emb=rotary_emb,
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        from sglang.srt.runtime_context import get_server_args

        self.use_fp4_indexer = get_server_args().enable_deepseek_v4_fp4_indexer
        self.alt_streams = alt_streams

    def compute_q(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_fp4_indexer:
            return fused_q_indexer_rope_hadamard_fp4_quant(
                q.contiguous(), weight, self.weight_scale, self.freqs_cis, positions
            )
        return fused_q_indexer_rope_hadamard_quant(
            q, weight, self.weight_scale, self.freqs_cis, positions
        )

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        out, _ = self.weights_proj(x)
        if not skip_scale:
            out = out * self.weight_scale
        return out

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
        skip_compressor: bool = False,
    ) -> None:
        return attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
            skip_compressor=skip_compressor,
        )
