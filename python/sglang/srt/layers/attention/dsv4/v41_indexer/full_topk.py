"""The full top-k of the ratio-1/2 index layers outside the candidate scheme:
DeepGEMM on SM100+, torch elsewhere."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.utils import is_xpu

from .scoring import (
    _index_q_and_weights,
    decode_scores,
    dense_prefill_topk,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    get_index_k_cache,
    prefill_requests,
    select_decode,
    sharded_dense_prefill_topk,
    write_prefill,
)
from .types import (
    DecodeInputs,
    PrefillInputs,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    from .scoring import DeepGEMMPrefillData


def _lite_topk_wins(num_rows: int, max_len: int) -> bool:
    """Whether LiteTopK beats the dense path on a chunk of ``num_rows`` query rows scored by this rank."""
    # LiteTopK needs rows x keys to fill the GPU: with few rows a CTA per 4 rows scans the whole key range and the
    # dense path wins. Rows of at most 6144 keys make every block dense (no sampling), where LiteTopK wins at any size.
    return (num_rows >= 512 and num_rows * max_len >= 1 << 26) or max_len <= 6144


_XPU_MQA_LOGITS_BUDGET_BYTES = 64 << 20


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
        if is_xpu():
            self._xpu_topk_prefill(inputs)
        elif self.use_deep_gemm_prefill:
            self._deep_gemm_prefill(inputs)
        else:
            self._torch_prefill(inputs)

    def topk_decode(self, inputs: DecodeInputs) -> None:
        if is_xpu():
            self._xpu_topk_decode(inputs)
        elif self.use_deep_gemm_decode:
            self._deep_gemm_decode(inputs)
        else:
            self._torch_decode(inputs)

    def _xpu_logits_and_topk(
        self,
        q_fp4: torch.Tensor,
        q_scales: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        k_cache: torch.Tensor,
        page_tables: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        out_page_indices: Optional[torch.Tensor],
        out_raw_indices: Optional[torch.Tensor],
    ) -> None:
        from sgl_kernel import fp4_paged_mqa_logits, topk_transform

        rows_per_chunk = max(1, _XPU_MQA_LOGITS_BUDGET_BYTES // max(max_seq_len * 4, 1))
        for start in range(0, q_fp4.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, q_fp4.shape[0])
            seq_lens_chunk = seq_lens[start:end].to(torch.int32).contiguous()
            page_tables_chunk = page_tables[start:end].to(torch.int32).contiguous()
            logits = fp4_paged_mqa_logits(
                (q_fp4[start:end], q_scales[start:end]),
                k_cache,
                weights[start:end].to(torch.float32),
                seq_lens_chunk,
                page_tables_chunk,
                None,
                max_seq_len,
                clean_logits=False,
            )
            page_out = (
                out_page_indices[start:end] if out_page_indices is not None else None
            )
            raw_out = (
                out_raw_indices[start:end] if out_raw_indices is not None else None
            )
            if page_out is not None:
                topk_transform(
                    logits,
                    seq_lens_chunk,
                    page_tables_chunk,
                    page_out,
                    page_size,
                    raw_out,
                )
            else:
                from sgl_kernel import topk_transform_ragged

                assert raw_out is not None
                topk_transform_ragged(
                    logits,
                    seq_lens_chunk,
                    raw_out,
                    torch.zeros_like(seq_lens_chunk),
                )

    def _xpu_topk_prefill(self, inputs: PrefillInputs) -> None:
        inputs.reset_outputs()
        if not inputs.x.shape[0]:
            return
        assert inputs.seq_lens_cpu is not None
        assert inputs.rows_per_request is not None
        assert len(inputs.rows_per_request) == inputs.req_pool_indices.numel()

        q_fp4, q_scales, weights = _index_q_and_weights(
            indexer=inputs.indexer,
            x=inputs.x,
            q_lora=inputs.q_lora,
            freqs_cis=inputs.freqs_cis,
            positions=inputs.positions,
        )
        seq_lens = ((inputs.positions + 1) // inputs.compress_ratio).to(torch.int32)
        max_seq_len = max(
            (length // inputs.compress_ratio for length in inputs.seq_lens_cpu),
            default=0,
        )
        if max_seq_len == 0:
            return

        page_size = self.token_to_kv_pool.get_index_k_page_size(inputs.compress_ratio)
        page_tables = expand_index_page_table(
            inputs.kv_page_table,
            full_page_size=self.token_to_kv_pool.page_size,
            compress_ratio=inputs.compress_ratio,
            index_page_size=page_size,
        ).contiguous()
        k_cache = get_index_k_cache(
            token_to_kv_pool=self.token_to_kv_pool,
            layer_id=inputs.layer_id,
            page_size=page_size,
        )

        row_start = 0
        for request_idx, request_rows in enumerate(inputs.rows_per_request):
            row_end = row_start + request_rows
            if request_rows:
                request_table = page_tables[request_idx : request_idx + 1]
                self._xpu_logits_and_topk(
                    q_fp4[row_start:row_end].unsqueeze(1),
                    q_scales[row_start:row_end].unsqueeze(1),
                    weights[row_start:row_end],
                    seq_lens[row_start:row_end],
                    k_cache,
                    request_table.expand(request_rows, -1),
                    max_seq_len,
                    page_size,
                    (
                        inputs.out_page_indices[row_start:row_end]
                        if inputs.out_page_indices is not None
                        else None
                    ),
                    inputs.out_raw_indices[row_start:row_end],
                )
            row_start = row_end
        assert row_start == inputs.x.shape[0]

    def _xpu_topk_decode(self, inputs: DecodeInputs) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        metadata = inputs.paged_metadata
        if metadata.max_compressed_seq_len == 0:
            return
        rows_per_request = metadata.rows_per_request
        q_fp4 = data.q_fp4
        q_scales = data.q_sf
        seq_lens = metadata.compressed_seq_lens.reshape(-1).to(torch.int32).contiguous()
        page_tables = (
            metadata.page_table[::rows_per_request]
            .repeat_interleave(rows_per_request, dim=0)
            .contiguous()
        )
        self._xpu_logits_and_topk(
            q_fp4,
            q_scales,
            data.weights,
            seq_lens,
            data.k_cache,
            page_tables,
            metadata.max_compressed_seq_len,
            metadata.compressed_page_size,
            inputs.out_page_indices,
            None,
        )

    def _deep_gemm_prefill(self, inputs: PrefillInputs) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs)
        if data is None:
            return
        out = inputs.out_raw_indices[: data.num_rows]
        shard = inputs.row_shard
        if self.use_prefill_lite_topk and _lite_topk_wins(
            data.num_rows if shard is None else shard.rows_per_rank(data.num_rows),
            max(data.lens_per_request),
        ):
            if shard is None:
                self._prefill_lite_topk(inputs, data, out)
            else:
                local = out.new_full(
                    (shard.rows_per_rank(data.num_rows), out.shape[1]), -1
                )
                local_data = data.narrow(shard.local_rows(data.num_rows))
                self._prefill_lite_topk(
                    inputs, local_data, local[: local_data.num_rows]
                )
                out.copy_(shard.gather(local, data.num_rows))
            data.write_page_indices(inputs)
            return
        kv = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
            inputs.layer_id, data.k_slots
        )
        if shard is None:
            dense_prefill_topk(data, kv, out=out)
        else:
            sharded_dense_prefill_topk(data, kv, out=out, shard=shard)
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
                q, lens, table, indices = data.paged_logits_args(metadata, rows)
                logits = deep_gemm_fp4_paged_mqa_logits(
                    q,
                    data.k_cache,
                    data.weights[rows],
                    lens,
                    table,
                    plan,
                    metadata.max_compressed_seq_len,
                    indices=indices,
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
        q, lens, table, indices = data.paged_logits_args(metadata)
        logits = deep_gemm_fp4_paged_mqa_logits(
            q,
            data.k_cache,
            data.weights,
            lens,
            table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
            indices=indices,
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
