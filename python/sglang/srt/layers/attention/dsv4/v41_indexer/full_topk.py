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
        if self.use_deep_gemm_prefill:
            self._deep_gemm_prefill(inputs)
        else:
            self._torch_prefill(inputs)

    def topk_prefill_captured(self, inputs: CapturedPrefillInputs) -> None:
        self._deep_gemm_prefill_captured(inputs)

    def topk_decode(self, inputs: DecodeInputs) -> None:
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
