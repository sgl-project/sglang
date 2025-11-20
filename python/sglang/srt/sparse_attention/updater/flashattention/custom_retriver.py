import time
import types
from functools import cache
from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.srt.sparse_attention.cache_manager.cache_manager import (
    CacheManager,
    RetriveResult,
    RetriveQuery,
)

from sglang.srt.sparse_attention.kernels.combine_indices_paged import combine_indices
from sglang.srt.sparse_attention.kernels.compute_scores.compute_scores_quest import (
    compute_quest_score as compute_score,
)
from sglang.srt.sparse_attention.kernels.proxy_k_tensor.proxy_k_tensor_quest import (
    proxy_k_tensor_decode,
    proxy_k_tensor_extend,
)
from sglang.srt.sparse_attention.kernels.score_copy import score_copy

if TYPE_CHECKING:
    from sglang.srt.layers.attention import FlashInferAttnBackend
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class DenseRetriver:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    def retrive_extend(
        self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata"
    ) -> "FlashAttentionMetadata":
        # Dense Implementation as default
        metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
        metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, : metadata.max_seq_len_k
        ]

        if any(forward_batch.extend_prefix_lens_cpu):
            extend_seq_lens = forward_batch.extend_seq_lens
            metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
            metadata.cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
        else:
            metadata.max_seq_len_q = metadata.max_seq_len_k
            metadata.cu_seqlens_q = metadata.cu_seqlens_k

    def retrive_decode(
        self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata"
    ) -> "FlashAttentionMetadata":
        # Dense Implementation as default
        metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
        metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
        metadata.cu_seqlens_q = torch.arange(
            0,
            forward_batch.batch_size + 1,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, : metadata.max_seq_len_k
        ]


class NaiveDecodeSparseRetriver:
    def __init__(self, cache_manager: "CacheManager"):
        self.cache_manager = cache_manager
        self.dense_retriver = DenseRetriver(self.cache_manager)
        self.retrive_extend = self.dense_retriver.retrive_extend
        self.top_k = cache_manager.config.top_k
        self.budget_per_seq = self.cache_manager.config.retrive_budget_per_seq
        self.stream_budget = cache_manager.config.stream_budget
        self.stream_len = sum(self.stream_budget) // cache_manager.config.page_size
        self.retrived_cache = (
            {}
        )  # {paged_indices: torch.Tensor, cu_seq_len: torch.Tensor}
        self.cache_manager._retrive_cache_indices = self._retrive_cache_indices
        self.cache_manager._call_after_update_query = self._call_after_update_query

    def update_extend(self, forward_batch: "ForwardBatch", layer_id: int):
        proxy_k_tensor_extend(
            key_cache=self.cache_manager.config.keys[layer_id],
            seq_lens=forward_batch.seq_lens,
            prefix_lens=forward_batch.extend_prefix_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            page_size=self.cache_manager.config.page_size,
            proxy_k_tensor=self.cache_manager.retrived_query[layer_id].proxy_k_tensor,
        )

    def build_stream(
        self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata"
    ) -> "FlashAttentionMetadata":
        if not self.cache_manager.config.async_retrive:
            self.stream_indices_page = self._get_stream_indices(forward_batch)

    def _get_stream_indices(self, forward_batch: "ForwardBatch"):
        token_indices = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]

        strided_indices = torch.arange(
            0,
            token_indices.shape[1],
            self.cache_manager.config.page_size,
            device=token_indices.device,
        )
        token_indices = (
            token_indices[:, strided_indices] // self.cache_manager.config.page_size
        )

        num_sink_pages = self.stream_budget[0] // self.cache_manager.config.page_size
        num_local_pages = self.stream_budget[1] // self.cache_manager.config.page_size

        seq_lens = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        num_pages_per_seq_end = (
            seq_lens + self.cache_manager.config.page_size - 1
        ) // self.cache_manager.config.page_size

        num_pages_per_seq_start = num_pages_per_seq_end - num_local_pages

        col_offsets = torch.arange(
            num_local_pages, device=token_indices.device
        ).unsqueeze(
            0
        )  # [1, num_local_pages]
        col_indices = (
            num_pages_per_seq_start.unsqueeze(1) + col_offsets
        )  # [batch_size, num_local_pages]

        batch_indices = torch.arange(batch_size, device=token_indices.device).unsqueeze(
            1
        )  # [batch_size, 1]
        local_pages = token_indices[
            batch_indices, col_indices
        ]  # [batch_size, num_local_pages]

        sink_pages = token_indices[:, :num_sink_pages]

        stream_indices_page = torch.cat([sink_pages, local_pages], dim=1)
        return stream_indices_page

    def _retrive_cache_indices(
        self,
        query: RetriveQuery,
        req_to_token: torch.Tensor,
        top_k: int,
    ):
        kv_pages_num_per_seq = (
            query.seq_lens + self.cache_manager.config.page_size - 1
        ) // self.cache_manager.config.page_size
        bs = query.bs

        compute_score(
            q=query.query[:bs],
            k=query.proxy_k_tensor,
            out=query.score[:bs, :, :query.max_num_pages],
            req_to_token=req_to_token,
            req_pool_indices=query.req_pool_indices[:bs],
            kv_pages_num_per_seq=kv_pages_num_per_seq[:bs],
            num_sink_pages=self.stream_budget[0] // self.cache_manager.config.page_size,
            num_local_pages=self.stream_budget[1]
            // self.cache_manager.config.page_size,
            page_size=self.cache_manager.config.page_size
        )
        _, topk_indices = torch.topk(query.score[:bs, :, :query.max_num_pages], k=top_k, dim=2, sorted=False)
        score_copy(topk_indices, req_to_token, query.req_pool_indices[:bs], bs, self.cache_manager.config.page_size)
        query.selected_page_indices[:bs, :, :] = topk_indices

    def _call_after_update_query(
        self,
        key_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        page_size: int,
        seq_lens: torch.Tensor,
        count_steps: torch.Tensor,  # [bs]
        accumlation_step: int,
        proxy_k_tensor: torch.Tensor,
    ):
        proxy_k_tensor_decode(
            key_cache=key_cache,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            page_size=page_size,
            count_steps=count_steps,
            accumlation_step=accumlation_step,
            proxy_k_tensor=proxy_k_tensor,
        )

    def _combine_indices_async(
        self,
        retrive_result: RetriveResult,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens: torch.Tensor,
        diff: torch.Tensor,
    ):
        num_sink_pages = self.stream_budget[0] // self.cache_manager.config.page_size
        num_local_pages = self.stream_budget[1] // self.cache_manager.config.page_size

        return combine_indices(
            retrived_cache_indices=retrive_result.retrived_cache_indices_page,
            cur_req_pool_indices=req_pool_indices,
            pre_req_pool_indices=retrive_result.req_pool_indices,
            req_to_token=req_to_token,
            page_table=retrive_result.page_table,
            seq_lens=seq_lens,
            new_seq_lens=retrive_result.sparse_seq_lens,
            diff=diff,
            num_sink_pages=num_sink_pages,
            num_local_pages=num_local_pages,
            page_size=self.cache_manager.config.page_size,
            budget_size=self.budget_per_seq,
            async_retrive=self.cache_manager.config.async_retrive,
        )

    def retrive_decode(
        self,
        query: torch.Tensor,
        forward_batch: "ForwardBatch",
        metadata: "FlashAttentionMetadata",
        layer: "RadixAttention",
    ):
        self.cache_manager.update_query(
            query,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.layer_id,
        )

        positions_in_page = (
            (forward_batch.seq_lens - 1) % self.cache_manager.config.page_size
        ).to(torch.int32)
        diff = self.cache_manager.config.page_size - positions_in_page - 1

        metadata.cu_seqlens_q = torch.arange(
            0,
            forward_batch.batch_size + 1,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )

        if self.cache_manager.config.async_retrive:
            retrive_result = self.cache_manager.get_result(layer.layer_id)
        else:
            self.cache_manager._retrive_one_layer(self.cache_manager.retrived_query[layer.layer_id], 
                                                  self.cache_manager.retrived_result[layer.layer_id])
            retrive_result = self.cache_manager.get_result(layer.layer_id)

        new_seq_lens = self._combine_indices_async(
            retrive_result,
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            diff,
        )

        num_heads = self.cache_manager.config.keys[0].shape[1]

        # set metadata info
        metadata.page_table = retrive_result.page_table[
            forward_batch.req_pool_indices, :, :
        ].reshape(forward_batch.batch_size * num_heads, -1)
        metadata.cache_seqlens_int32 = new_seq_lens
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )