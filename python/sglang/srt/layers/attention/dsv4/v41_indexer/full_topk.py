"""The full top-k of the ratio-1/2 index layers outside the candidate scheme:
DeepGEMM on SM100, torch elsewhere."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.deep_select import (
    is_deep_select_supported,
    topk_page_transform,
)
from sglang.kernels.ops.attention.dsv4.index_logits import (
    deep_gemm_fp4_paged_mqa_logits,
)
from sglang.kernels.ops.attention.dsv4.sm80_utils import is_dsv41_sm80_experimental
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_torch
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_from_metadata

from .scoring import (
    decode_scores,
    dense_prefill_topk,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    get_index_k_cache,
    prefill_requests,
    quantize_index_q,
    select_decode,
    write_prefill,
)
from .types import (
    CapturedPrefillInputs,
    DecodeInputs,
    PrefillInputs,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


def _all_causal_bound(inputs, req_to_token: torch.Tensor) -> int | None:
    """A host/shape-only upper bound; never synchronize live device lengths.

    These backends are exclusive to V4.1. The shared gate checks the actual
    query device, not a global platform flag or the current CUDA device.
    """
    if (
        not is_dsv41_sm80_experimental("deepseek_v41", inputs.positions.device)
        or inputs.compress_ratio not in (1, 2)
        or inputs.indexer.index_topk != 512
        or inputs.req_rows is None
        or inputs.req_rows.shape != inputs.positions.shape
    ):
        return None
    ratio = inputs.compress_ratio
    if isinstance(inputs, PrefillInputs):
        lengths, rows = inputs.seq_lens_cpu, inputs.rows_per_request
        if (
            lengths is None
            or rows is None
            or len(lengths) != len(rows)
            or len(lengths) != inputs.req_pool_indices.shape[0]
            or sum(rows) != inputs.positions.shape[0]
            or any(s < 0 or n < 0 or n > s for s, n in zip(lengths, rows))
        ):
            return None
        bound = max(lengths, default=0) // ratio
    else:
        # This is a replay-capacity bound, NOT a live length read. Wide graph
        # capacities deliberately fall back even if today's rows are short.
        bound = inputs.paged_metadata.max_compressed_seq_len
    if bound < 0 or bound > req_to_token.shape[1] // ratio:
        return None
    return bound


def _write_all_causal(
    inputs, req_to_token: torch.Tensor, bound: int, candidate_lens=None
) -> None:
    """Every visible key is in top-512, independent of queries and scores."""
    inputs.reset_outputs()
    num_rows = inputs.positions.shape[0]
    j = torch.arange(bound, device=inputs.positions.device)
    valid = j[None, :] < ((inputs.positions + 1) // inputs.compress_ratio)[:, None]
    if candidate_lens is not None:
        valid = valid & (j[None, :] < candidate_lens[:, None])
    if isinstance(inputs, PrefillInputs):
        inputs.out_raw_indices[:num_rows, :bound] = torch.where(valid, j[None, :], -1)
    if inputs.out_page_indices is not None:
        slots = (
            req_to_token[
                inputs.req_rows[:, None], (j * inputs.compress_ratio)[None, :]
            ].to(torch.int64)
            // inputs.compress_ratio
        )
        inputs.out_page_indices[:num_rows, :bound] = torch.where(valid, slots, -1)


class FullTopKIndexer:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
        use_deep_gemm_prefill: bool,
        use_deep_gemm_decode: bool,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.use_deep_gemm_prefill = use_deep_gemm_prefill
        self.use_deep_gemm_decode = use_deep_gemm_decode
        self.use_deep_select_decode = (
            not use_deep_gemm_decode and is_deep_select_supported()
        )

    def topk_prefill(self, inputs: PrefillInputs) -> None:
        if not self.use_deep_gemm_prefill:
            bound = _all_causal_bound(inputs, self.req_to_token)
            if bound is not None and bound <= 512:
                _write_all_causal(inputs, self.req_to_token, bound)
                return
        if self.use_deep_gemm_prefill:
            self._deep_gemm_prefill(inputs)
        else:
            self._torch_prefill(inputs)

    def topk_prefill_captured(self, inputs: CapturedPrefillInputs) -> None:
        self._deep_gemm_prefill_captured(inputs)

    def topk_decode(self, inputs: DecodeInputs) -> None:
        if not self.use_deep_gemm_decode:
            bound = _all_causal_bound(inputs, self.req_to_token)
            if bound is not None and bound <= 512:
                _write_all_causal(inputs, self.req_to_token, bound)
                return
        if self.use_deep_gemm_decode:
            self._deep_gemm_decode(inputs)
        else:
            self._torch_decode(inputs)

    def _deep_gemm_prefill(self, inputs: PrefillInputs) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return
        kv = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
            inputs.layer_id, data.k_slots
        )
        dense_prefill_topk(data, kv, out=inputs.out_raw_indices[: data.num_rows])
        data.write_page_indices(inputs)

    def _torch_prefill(self, inputs: PrefillInputs) -> None:
        requests = prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )
        for request, chunks in requests:
            for chunk in chunks:
                idx = chunk.scores.topk(request.k, dim=-1, sorted=False).indices
                write_prefill(inputs, request, chunk, idx)

    def _deep_gemm_decode(self, inputs: DecodeInputs) -> None:
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        metadata = inputs.paged_metadata
        if isinstance(metadata.deep_gemm_metadata, list):
            topk_plans = metadata.topk_metadata_chunks
            assert not metadata.use_topk_v2 or topk_plans is not None
            for chunk_idx, (rows, plan) in enumerate(metadata.row_chunks()):
                logits = deep_gemm_fp4_paged_mqa_logits(
                    (data.q_fp4[rows], data.q_sf[rows]),
                    data.k_cache,
                    data.weights[rows],
                    metadata.compressed_seq_lens[rows],
                    metadata.page_table[rows],
                    plan,
                    metadata.max_compressed_seq_len,
                )
                # TODO(dark): add bf16 topk
                topk_transform_paged_from_metadata(
                    logits,
                    metadata,
                    inputs.out_page_indices,
                    rows=rows,
                    topk_metadata=(
                        topk_plans[chunk_idx] if topk_plans is not None else None
                    ),
                )
            return
        logits = deep_gemm_fp4_paged_mqa_logits(
            (data.q_fp4, data.q_sf),
            data.k_cache,
            data.weights,
            metadata.compressed_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
        )
        # TODO(dark): add bf16 topk
        topk_transform_paged_from_metadata(
            logits, metadata, inputs.out_page_indices, None
        )

    def _torch_decode(self, inputs: DecodeInputs) -> None:
        d = decode_scores(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            compute_topk_plan=not self.use_deep_select_decode,
        )
        if d is None:
            return
        if self.use_deep_select_decode:
            metadata = inputs.paged_metadata
            topk_page_transform(
                d.scores,
                inputs.indexer.index_topk,
                page_table=metadata.page_table[: d.bs],
                page_size=metadata.compressed_page_size,
                end=d.lens.to(torch.int32),
                sorted_index=False,
                output_idx=inputs.out_page_indices[: d.bs],
            )
            return
        select_decode(inputs, d, inputs.indexer.index_topk)

    def _deep_gemm_prefill_captured(self, inputs: CapturedPrefillInputs) -> None:
        indexer = inputs.indexer
        metadata = inputs.paged_metadata
        assert indexer.n_local_heads == indexer.n_heads
        q_fp4, q_sf = quantize_index_q(inputs.q)
        num_tokens, num_heads = q_fp4.shape[0], q_fp4.shape[1]
        q_fp4 = q_fp4.view(num_tokens, 1, num_heads, 64)
        q_sf = q_sf.view(num_tokens, 1, num_heads)
        weights = inputs.weights.float()
        page_size = metadata.compressed_page_size
        k_cache = get_index_k_cache(
            token_to_kv_pool=self.token_to_kv_pool,
            layer_id=inputs.layer_id,
            page_size=page_size,
        )

        width = metadata.max_compressed_seq_len
        lens = metadata.compressed_seq_lens
        page_table = metadata.page_table
        topk = min(indexer.index_topk, width)
        for rows, plan in metadata.row_chunks():
            logits = deep_gemm_fp4_paged_mqa_logits(
                (q_fp4[rows], q_sf[rows]),
                k_cache,
                weights[rows],
                lens[rows],
                page_table[rows],
                plan,
                width,
            )
            topk_transform_paged_torch(
                logits,
                lens[rows],
                page_table[rows],
                inputs.out_page_indices[rows, :topk],
                page_size,
                inputs.out_raw_indices[rows, :topk],
            )
