from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import get_dsa_index_kpool
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMultiStepDraftBackend,
    ForwardMetadata,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.kpool_plan import (
    KPoolExtendPlan,
    KPoolWritePlan,
    init_kpool_extend_metadata,
    init_kpool_write_plan_capture,
    update_kpool_write_plan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class AscendDSAForwardMetadata(ForwardMetadata):
    """Attention/read inputs and shared extend or speculative write plans."""

    # ── Core attention ──
    page_size: int = 0
    cache_lens_int32: Optional[torch.Tensor] = None
    real_page_table: Optional[torch.Tensor] = None

    # Inputs for pooled_key_indexer (batch-level)
    pool_query_ends: Optional[torch.Tensor] = None
    pool_key_lens: Optional[torch.Tensor] = None
    pool_tail_lens: Optional[torch.Tensor] = None
    pool_page_table: Optional[torch.Tensor] = None

    # Inputs for compress_write_npu
    kpool_extend_plan: Optional[KPoolExtendPlan] = None
    kpool_write_plan: Optional[KPoolWritePlan] = None


@dataclass(frozen=True)
class AscendDSAIndexerMetadata(BaseIndexerMetadata):
    """Read-only adapter from Ascend attention metadata to the indexer API."""

    attn_metadata: AscendDSAForwardMetadata

    def get_seqlens_int32(self) -> torch.Tensor:
        assert self.attn_metadata.cache_lens_int32 is not None
        return self.attn_metadata.cache_lens_int32

    def get_page_table_64(self) -> torch.Tensor:
        assert self.attn_metadata.real_page_table is not None
        return self.attn_metadata.real_page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        raise NotImplementedError("NPU pooled-key reads use request-level lengths")

    def get_page_table_1(self) -> torch.Tensor:
        raise NotImplementedError(
            "NPU kpool path uses get_page_table_64, not get_page_table_1"
        )

    def get_token_to_batch_idx(self) -> torch.Tensor:
        raise NotImplementedError("NPU pooled-key reads use request-level page tables")

    def topk_transform(self, logits: torch.Tensor, topk: int, **_) -> torch.Tensor:
        raise NotImplementedError(
            "NPU KPool uses a vendor indexer instead of topk_transform"
        )


class AscendDSAAttnBackend(AscendAttnBackend):
    """Add the KPool metadata required by Ascend attention for DSA models."""

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        # Physical page-table sizing still uses the existing host length mirror.
        self.needs_cpu_seq_lens = True
        hf_config = model_runner.model_config.hf_config
        self.dsa_index_kpool = get_dsa_index_kpool(hf_config)

        if self.dsa_index_kpool > 1:
            assert self.page_size % self.dsa_index_kpool == 0, (
                "Ascend DSA KPool requires a pool size that "
                f"divides page_size; got page_size={self.page_size}, "
                f"index_kpool={self.dsa_index_kpool}."
            )
            # Allocator pages own a whole index page; attention still reads
            # 128-token physical KV pages. Metadata inherits this value.
            self.page_size //= self.dsa_index_kpool

    def _create_forward_metadata(self) -> AscendDSAForwardMetadata:
        return AscendDSAForwardMetadata(page_size=self.page_size)

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> AscendDSAForwardMetadata:
        metadata = super()._init_cuda_graph_metadata(
            bs, forward_mode, seq_lens, out_cache_loc
        )
        # DSA derives effective lengths from the raw graph inputs after the base
        # update. Keep a separate buffer for each backend and capture bucket.
        metadata.seq_lens = seq_lens.clone()
        return metadata

    def _query_lens_device(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Return per-request query lengths without a device-host round trip."""
        mode = forward_batch.forward_mode
        # Target verify and draft extend v2 both use a fixed batch_size * N query layout.
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return torch.full(
                (forward_batch.batch_size,),
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
        if mode.is_extend():
            return forward_batch.extend_seq_lens[: forward_batch.batch_size].to(
                torch.int32
            )
        return torch.ones(
            (forward_batch.batch_size,),
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )

    def _cache_lens_device(self, forward_batch: ForwardBatch) -> torch.Tensor:
        cache_seqlens = forward_batch.seq_lens[: forward_batch.batch_size].to(
            torch.int32
        )
        if forward_batch.forward_mode.is_target_verify():
            return cache_seqlens + self.speculative_num_draft_tokens
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            return cache_seqlens + self.speculative_step_id + 1
        return cache_seqlens

    def _max_kv_len(self, forward_batch: ForwardBatch) -> int:
        """Return the effective maximum KV length from the existing CPU mirror."""
        max_seq_len = int(
            forward_batch.seq_lens_cpu[: forward_batch.batch_size].max().item()
        )
        if forward_batch.forward_mode.is_target_verify():
            return max_seq_len + self.speculative_num_draft_tokens
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            return max_seq_len + self.speculative_step_id + 1
        return max_seq_len

    def _initialize_dsa_metadata(self, forward_batch: ForwardBatch, *, capture=False):
        """Construct complete metadata for eager execution or graph capture."""
        metadata = self.forward_metadata
        assert isinstance(metadata, AscendDSAForwardMetadata)
        cache_lens = self._cache_lens_device(forward_batch)  #
        query_lens = self._query_lens_device(forward_batch)
        max_kv_len = self._max_kv_len(forward_batch)
        batch = forward_batch.batch_size
        device = cache_lens.device

        # prepare page table
        token_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[:batch], :max_kv_len
        ]
        page_table = (token_table[:, :: self.page_size] // self.page_size).to(
            torch.int32
        )

        # --- prepare real_page_table metadata ---
        metadata.cache_lens_int32 = cache_lens
        if capture:
            max_num_pages = max(
                page_table.shape[1],
                self.req_to_token_pool.req_to_token.shape[1] // self.page_size,
            )
            metadata.real_page_table = torch.zeros(
                (batch, max_num_pages), dtype=torch.int32, device=device
            )
            metadata.real_page_table[:, : page_table.shape[1]].copy_(page_table)
        else:
            metadata.real_page_table = page_table

        # --- prepare kpool metadata ---
        if self.dsa_index_kpool > 1:
            ratio = self.dsa_index_kpool

            # --- prepare pooled_key_indexer metadata ---
            metadata.pool_query_ends = query_lens.cumsum(0, dtype=torch.int64)
            metadata.pool_key_lens = (cache_lens // ratio).to(torch.int64)
            metadata.pool_tail_lens = (cache_lens % ratio).to(torch.int64)
            metadata.pool_page_table = metadata.real_page_table[:, ::ratio] // ratio

            # --- prepare compress_write_npu metadata ---
            mode = forward_batch.forward_mode
            if mode.is_extend_without_speculative():  # extend
                plan = init_kpool_extend_metadata(
                    metadata,
                    forward_batch,
                    kpool=ratio,
                    physical_page_size=self.page_size,
                    index_page_size=self.token_to_kv_pool.index_page_size,
                    topk_transform_method=None,
                    full_real_page_table=metadata.real_page_table,
                    build_ragged_metadata=False,
                )
                metadata.kpool_extend_plan = plan.kpool_extend_plan
            elif (
                mode.is_target_verify() or mode.is_draft_extend_v2()
            ):  # target_verify, draft_extend_v2
                plan = init_kpool_write_plan_capture(
                    metadata,
                    max_bs=batch,
                    kpool=ratio,
                    physical_page_size=self.page_size,
                    num_draft_tokens=self.speculative_num_draft_tokens,
                    device=device,
                    is_verify=False,
                    index_page_size=self.token_to_kv_pool.index_page_size,
                    is_v2=mode.is_draft_extend_v2(),
                    build_schedule_metadata=False,
                )
                metadata.kpool_write_plan = plan.kpool_write_plan
                self._mtp_update_metadata(forward_batch)

    def _replay_dsa_metadata(self, forward_batch: ForwardBatch):
        """Refresh captured metadata in place without replacing its buffers."""
        metadata = self.forward_metadata
        cache_lens = self._cache_lens_device(forward_batch)
        query_lens = self._query_lens_device(forward_batch)
        max_kv_len = self._max_kv_len(forward_batch)
        batch = forward_batch.batch_size

        # prepare page table
        token_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[:batch], :max_kv_len
        ]
        page_table = (token_table[:, :: self.page_size] // self.page_size).to(
            torch.int32
        )

        # --- update real_page_table metadata ---
        metadata.cache_lens_int32.copy_(cache_lens)
        metadata.real_page_table[:, : page_table.shape[1]].copy_(page_table)

        # --- update kpool metadata ---
        if self.dsa_index_kpool > 1:
            ratio = self.dsa_index_kpool

            # --- update pooled_key_indexer metadata ---
            metadata.pool_query_ends.copy_(query_lens.cumsum(0, dtype=torch.int64))
            metadata.pool_key_lens.copy_(cache_lens // ratio)
            metadata.pool_tail_lens.copy_(cache_lens % ratio)
            metadata.pool_page_table.copy_(
                metadata.real_page_table[:, ::ratio] // ratio
            )

            # --- update compress_write_npu metadata ---
            mode = forward_batch.forward_mode
            if mode.is_target_verify() or mode.is_draft_extend_v2():
                self._mtp_update_metadata(forward_batch)

    def _mtp_update_metadata(self, forward_batch: ForwardBatch):
        """Refresh the shared MTP plan."""
        batch = forward_batch.batch_size
        mode = forward_batch.forward_mode
        num_draft_tokens = self.speculative_num_draft_tokens

        write_start = forward_batch.seq_lens[:batch].to(torch.int32)
        if mode.is_draft_extend_v2():  # seq_len already includes draft_len
            write_start = write_start - num_draft_tokens
        effective_n = None  # commit length
        if mode.is_draft_extend_v2():
            effective_n = forward_batch.spec_info.num_accept_tokens

        # --- update compress_write_npu metadata ---
        update_kpool_write_plan(
            self.forward_metadata,
            write_start=write_start,
            req_pool_indices=forward_batch.req_pool_indices,
            real_page_table=self.forward_metadata.real_page_table,
            kpool=self.dsa_index_kpool,
            physical_page_size=self.page_size,
            num_draft_tokens=num_draft_tokens,
            forward_mode=mode,
            index_page_size=self.token_to_kv_pool.index_page_size,
            effective_n_per_batch=effective_n,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        super().init_forward_metadata(forward_batch)
        self._initialize_dsa_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        if in_capture:
            self._initialize_dsa_metadata(forward_batch, capture=True)
        else:
            self._replay_dsa_metadata(forward_batch)

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> AscendDSAIndexerMetadata:
        metadata = self.forward_metadata
        if not isinstance(metadata, AscendDSAForwardMetadata):
            raise RuntimeError("Ascend DSA metadata is not initialized")
        return AscendDSAIndexerMetadata(attn_metadata=metadata)


class AscendDSAAttnMultiStepDraftBackend(AscendAttnMultiStepDraftBackend):
    """Create an Ascend backend that generates DSA metadata for each draft decode step."""

    backend_cls = AscendDSAAttnBackend
