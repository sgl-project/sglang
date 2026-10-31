"""The full top-k of the ratio-1/2 index layers outside the candidate scheme:
DeepGEMM on SM100, torch elsewhere."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.deep_select import (
    is_deep_select_supported,
    topk_page_transform,
)
from sglang.kernels.ops.attention.dsa.litetopk_prefill import fp4_indexer_topk
from sglang.kernels.ops.attention.dsv4.index_logits import (
    deep_gemm_fp4_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_from_metadata

from .scoring import (
    decode_scores,
    dense_prefill_topk,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    prefill_requests,
    select_decode,
    write_prefill,
)
from .types import (
    DecodeInputs,
    PrefillInputs,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    from .scoring import DeepGEMMPrefillData


class FullTopKIndexer:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
        use_deep_gemm_prefill: bool,
        use_deep_gemm_decode: bool,
        use_prefill_lite_topk: bool = False,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.use_deep_gemm_prefill = use_deep_gemm_prefill
        self.use_prefill_lite_topk = use_prefill_lite_topk
        self.use_deep_gemm_decode = use_deep_gemm_decode
        self.use_deep_select_decode = (
            not use_deep_gemm_decode and is_deep_select_supported()
        )

    def topk_prefill(self, inputs: PrefillInputs) -> None:
        if self.use_deep_gemm_prefill:
            self._deep_gemm_prefill(inputs)
        else:
            self._torch_prefill(inputs)

    def topk_decode(self, inputs: DecodeInputs) -> None:
        if self.use_deep_gemm_decode:
            self._deep_gemm_decode(inputs)
        else:
            self._torch_decode(inputs)

    def _deep_gemm_prefill(self, inputs: PrefillInputs) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs)
        if data is None:
            return
        out = inputs.out_raw_indices[: data.num_rows]
        # LiteTopK needs rows x keys to fill the GPU: with few rows a CTA per 4 rows scans the whole key range and the
        # dense path wins.
        max_len = max(data.lens_per_request)
        use_lite = data.num_rows >= 512 and data.num_rows * max_len >= 1 << 26
        # Rows of at most 6144 keys make every block dense (no sampling), where LiteTopK wins at any size.
        use_lite = use_lite or max_len <= 6144
        if self.use_prefill_lite_topk and use_lite:
            self._prefill_lite_topk(inputs, data, out)
        else:
            kv = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, data.k_slots
            )
            dense_prefill_topk(data, kv, out=out)
        data.write_page_indices(inputs)

    def _prefill_lite_topk(
        self,
        inputs: PrefillInputs,
        data: DeepGEMMPrefillData,
        out: torch.Tensor,
    ) -> None:
        """Request-local compressed positions into ``out`` from the fp4 indexer + bf16 top-k."""
        # The indexer's scale tensor map needs a multiple of 4 keys: pad with slot 0, past every row's range.
        slots = data.k_slots
        pad = -slots.shape[0] % 4
        if pad:
            slots = torch.cat([slots, slots[:1].expand(pad)])
        kv_fp4, kv_sf = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
            inputs.layer_id, slots
        )
        fp4_indexer_topk(
            data.q_fp4,
            data.q_sf,
            data.weights.to(torch.bfloat16),
            kv_fp4,
            kv_sf,
            data.request_starts,
            data.request_starts + data.compress_lens,
            out.shape[1],
            out=out,
        )

    def _torch_prefill(self, inputs: PrefillInputs) -> None:
        requests = prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
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
                q, lens, table = data.paged_logits_args(metadata, rows)
                logits = deep_gemm_fp4_paged_mqa_logits(
                    q,
                    data.k_cache,
                    data.weights[rows],
                    lens,
                    table,
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
        q, lens, table = data.paged_logits_args(metadata)
        logits = deep_gemm_fp4_paged_mqa_logits(
            q,
            data.k_cache,
            data.weights,
            lens,
            table,
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
