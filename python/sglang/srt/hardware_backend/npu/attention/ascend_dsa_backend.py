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
from sglang.srt.hardware_backend.npu.attention.ascend_kpool_plan import (
    AscendKpoolWritePlan,
    init_kpool_write_plan,
    update_kpool_write_plan,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class AscendDSAForwardMetadata(ForwardMetadata):
    """Attention/indexer inputs and the bucket-owned KeyPool write plan.

    The backend normalizes lengths; kpool owns write mappings and layer scratch.
    """

    # ── Core attention ──
    page_size: int = 0
    cache_seqlens_int32: Optional[torch.Tensor] = None
    real_page_table: Optional[torch.Tensor] = None

    # Inputs for pooled key indexer (batch-level)
    pool_query_ends: Optional[torch.Tensor] = None
    pool_key_lens: Optional[torch.Tensor] = None
    pool_tail_lens: Optional[torch.Tensor] = None
    pool_page_table: Optional[torch.Tensor] = None

    # Inputs for key pool
    kpool: Optional[AscendKpoolWritePlan] = None


@dataclass(frozen=True)
class AscendDSAIndexerMetadata(BaseIndexerMetadata):
    """Read-only adapter from Ascend attention metadata to the indexer API."""

    attn_metadata: AscendDSAForwardMetadata

    def get_seqlens_int32(self) -> torch.Tensor:
        assert self.attn_metadata.cache_seqlens_int32 is not None
        return self.attn_metadata.cache_seqlens_int32

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

    def _resolve_sequence_layout(
        self, forward_batch: ForwardBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Normalize computed query, read length and committed write length once."""
        mode = forward_batch.forward_mode
        batch = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            query_lens = torch.full(
                (batch,),
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=device,
            )
        elif mode.is_extend():
            query_lens = forward_batch.extend_seq_lens
            if query_lens is None or query_lens.shape[0] < batch:
                raise ValueError("DSA requires one query length per request")
            query_lens = query_lens[:batch].to(torch.int32)
        else:
            query_lens = torch.ones((batch,), dtype=torch.int32, device=device)

        # Share the same adjustment between device lengths and the existing
        # CPU maximum used to size page-table work. No device-to-host reads.
        offset = 0
        if mode.is_target_verify():
            offset = self.speculative_num_draft_tokens
        elif mode.is_decode_or_idle() and forward_batch.spec_info is not None:
            offset = self.speculative_step_id + 1
        cache_lens = forward_batch.seq_lens[:batch].to(torch.int32) + offset
        assert forward_batch.seq_lens_cpu is not None
        host_lens = forward_batch.seq_lens_cpu[:batch]
        max_kv_len = (int(host_lens.max().item()) if host_lens.numel() else 0) + offset

        commit_lens = query_lens
        if mode.is_draft_extend_v2():
            commit_lens = torch.zeros_like(query_lens)
            accepted = getattr(forward_batch.spec_info, "num_accept_tokens", None)
            if accepted is not None:
                count = min(batch, accepted.numel())
                commit_lens[:count] = accepted[:count].to(torch.int32).clamp_min(0)
            commit_lens = torch.minimum(commit_lens, query_lens)
        return cache_lens, query_lens, commit_lens, max_kv_len

    def _kpool_layout(self, forward_batch: ForwardBatch):
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return "mtp", self.speculative_num_draft_tokens
        return "normal", (
            1 if mode.is_decode_or_idle() else forward_batch.out_cache_loc.numel()
        )

    def _initialize_dsa_metadata(self, forward_batch: ForwardBatch, *, capture=False):
        """Allocate once, then use the same fill path as graph replay."""
        metadata = self.forward_metadata
        assert isinstance(metadata, AscendDSAForwardMetadata)
        layout = self._resolve_sequence_layout(forward_batch)
        cache_lens, _, _, max_kv_len = layout
        columns = (max_kv_len + self.page_size - 1) // self.page_size
        if capture:
            columns = max(
                columns, self.req_to_token_pool.req_to_token.shape[1] // self.page_size
            )
        batch = forward_batch.batch_size
        device = cache_lens.device
        metadata.cache_seqlens_int32 = torch.empty_like(cache_lens)
        metadata.real_page_table = torch.zeros(
            (batch, columns), dtype=torch.int32, device=device
        )
        if self.dsa_index_kpool > 1:
            metadata.pool_query_ends = torch.empty(
                (batch,), dtype=torch.int64, device=device
            )
            metadata.pool_key_lens = torch.empty(
                (batch,), dtype=torch.int64, device=device
            )
            metadata.pool_tail_lens = torch.empty(
                (batch,), dtype=torch.int64, device=device
            )
            metadata.pool_page_table = torch.empty(
                (batch, (columns + self.dsa_index_kpool - 1) // self.dsa_index_kpool),
                dtype=torch.int32,
                device=device,
            )
            pool = getattr(self.token_to_kv_pool, "full_kv_pool", self.token_to_kv_pool)
            mode, max_query_len = self._kpool_layout(forward_batch)
            metadata.kpool = init_kpool_write_plan(
                batch_size=batch,
                num_tokens=forward_batch.out_cache_loc.numel(),
                cmp_ratio=self.dsa_index_kpool,
                indexer_head_dim=pool.index_head_dim,
                index_page_size=pool.index_page_size,
                tail_capacity=self.dsa_index_kpool + pool.tail_extra_slots,
                mode=mode,
                max_query_len=max_query_len,
                device=device,
            )
        self._update_dsa_metadata(forward_batch, sequence_layout=layout)

    def _update_dsa_metadata(
        self, forward_batch: ForwardBatch, *, sequence_layout=None
    ):
        """Refresh attention/read inputs and the owned write plan in place."""
        metadata = self.forward_metadata
        assert isinstance(metadata, AscendDSAForwardMetadata)
        cache_lens, query_lens, commit_lens, max_kv_len = (
            self._resolve_sequence_layout(forward_batch)
            if sequence_layout is None
            else sequence_layout
        )
        batch = forward_batch.batch_size
        if metadata.cache_seqlens_int32.shape != cache_lens.shape:
            raise RuntimeError("DSA length shape differs from allocated metadata")
        req_slots = forward_batch.req_pool_indices[:batch]
        token_table = self.req_to_token_pool.req_to_token[req_slots, :max_kv_len]
        pages = (token_table[:, :: self.page_size] // self.page_size).to(torch.int32)
        if (
            metadata.real_page_table.shape[0] != batch
            or metadata.real_page_table.shape[1] < pages.shape[1]
        ):
            raise RuntimeError("DSA page table exceeds allocated metadata")
        metadata.cache_seqlens_int32.copy_(cache_lens)
        metadata.real_page_table[:, : pages.shape[1]].copy_(pages)
        # The ordinary Ascend attention call sites still consume this field.
        # Keep compatibility separate from pure length normalization.
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            forward_batch.extend_seq_lens = query_lens
        if self.dsa_index_kpool <= 1:
            return
        self._update_indexer_inputs(metadata, cache_lens, query_lens)
        write_plan = metadata.kpool
        assert write_plan is not None
        write_mode, max_query_len = self._kpool_layout(forward_batch)
        if write_plan.mode != write_mode or write_plan.max_query_len != max_query_len:
            raise RuntimeError(
                "KeyPool replay mode/query capacity differs from capture"
            )
        # Replay keeps real requests before the bucket's padded suffix. Use that
        # boundary directly; out_cache_loc may contain only the unpadded tokens.
        num_padding = getattr(forward_batch, "num_padding", 0)
        if not 0 <= num_padding <= batch:
            raise ValueError("Invalid DSA request padding")
        actual_mode = getattr(forward_batch, "actual_forward_mode", mode)
        num_real_requests = 0 if actual_mode.is_idle() else batch - num_padding
        valid = query_lens > 0
        valid[num_real_requests:] = False
        start_pos = torch.where(valid, (cache_lens - query_lens).clamp_min(0), 0)
        update_kpool_write_plan(
            write_plan,
            start_pos=start_pos,
            query_lens=query_lens,
            commit_lens=torch.where(valid, commit_lens, 0),
            req_pool_indices=req_slots,
            valid_requests=valid,
            index_page_table=metadata.pool_page_table,
        )

    def _update_indexer_inputs(self, metadata, cache_lens, query_lens):
        """Read-side lengths use the final computed cache length, not acceptance."""
        ratio = self.dsa_index_kpool
        metadata.pool_query_ends.copy_(query_lens.cumsum(0, dtype=torch.int64))
        metadata.pool_key_lens.copy_(cache_lens // ratio)
        metadata.pool_tail_lens.copy_(cache_lens % ratio)
        metadata.pool_page_table.copy_(metadata.real_page_table[:, ::ratio] // ratio)

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
            self._update_dsa_metadata(forward_batch)

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
