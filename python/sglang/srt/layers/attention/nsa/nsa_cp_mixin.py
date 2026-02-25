from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.layers.attention.nsa.nsa_indexer_metadata import BaseIndexerMetadata
from sglang.srt.layers.attention.nsa.utils import (
    cp_all_gather_rerange_output,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_in_seq_split,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda, is_npu

global _use_multi_stream
_is_cuda = is_cuda()
if _is_cuda:
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if is_npu():
    import torch_npu

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool


class IndexerContextParallelMixin:
    """Mixin class for context parallel operations executed on NSA indexer"""

    def init_cp(self):
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attn_context_model_parallel_world_size()
            self.cp_rank = get_attn_context_model_parallel_rank()
        else:
            self.cp_size = None
            self.cp_rank = None

    def cp_allgather_and_rerange_keys(
        self, key: torch.Tensor, forward_batch: ForwardBatch, stream
    ):
        # allgather+rerange when enabling cp
        if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                stream,
            )
        return key

    def _get_topk_ragged_with_cp_core(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        kv_len = (
            forward_batch.seq_lens_cpu[0].item()
            - forward_batch.extend_seq_lens_cpu[0]
            + kv_len
        )
        k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
            layer_id,
            kv_len,
            block_tables[0],
        )
        k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
            layer_id,
            kv_len,
            block_tables[0],
        )

        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
        k_scale = k_scale.view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
        ke_offset = torch.arange(
            (kv_len - actual_seq_q) + 1,
            kv_len + 1,
            dtype=torch.int32,
            device="cuda",
        )
        ke = ks + ke_offset

        with self._with_real_sm_count():
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                ks,
                ke,
                clean_logits=False,
            )
        actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
            device="cuda", non_blocking=True
        )
        topk_result = metadata.topk_transform(
            logits,
            self.index_topk,
            ks=ks,
            cu_seqlens_q=actual_seq_q,
            ke_offset=ke_offset,
        )

        return topk_result

    def get_topk_ragged_with_cp_in_seq_split(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ):
        if not (
            forward_batch.nsa_cp_metadata is not None
            and is_nsa_prefill_cp_in_seq_split()
        ):
            return None

        kv_len_prev = forward_batch.nsa_cp_metadata.kv_len_prev
        kv_len_next = forward_batch.nsa_cp_metadata.kv_len_next
        actual_seq_q_prev = forward_batch.nsa_cp_metadata.actual_seq_q_prev
        actual_seq_q_next = forward_batch.nsa_cp_metadata.actual_seq_q_next

        # TODO support mutil-batch
        # cp_batch_seq_index_prev = forward_batch.nsa_cp_metadata["cp_batch_seq_index_prev"]
        # cp_batch_seq_index_next = forward_batch.nsa_cp_metadata["cp_batch_seq_index_next"]
        # TODO prev, next, combined into a single call
        q_fp8_prev, q_fp8_next = torch.split(q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0)
        weights_prev, weights_next = torch.split(
            weights, (weights.shape[0] + 1) // 2, dim=0
        )
        topk_result_prev = self._get_topk_ragged_with_cp_core(
            forward_batch,
            layer_id,
            q_fp8_prev,
            weights_prev,
            metadata,
            kv_len_prev,
            actual_seq_q_prev,
        )

        topk_result_next = self._get_topk_ragged_with_cp_core(
            forward_batch,
            layer_id,
            q_fp8_next,
            weights_next,
            metadata,
            kv_len_next,
            actual_seq_q_next,
        )
        return torch.cat([topk_result_prev, topk_result_next], dim=0)

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return topk_indices_prev[0], topk_indices_next[0]
