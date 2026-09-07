from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.dsa.kpool_plan import (
    init_kpool_extend_metadata,
    init_kpool_write_plan,
    init_kpool_write_plan_capture,
    init_pooled_paged_mqa_metadata,
    update_kpool_write_plan,
    update_pooled_paged_mqa_metadata,
)
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
        PrecomputedMetadata,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.attention.dsa_backend import _DSA_IMPL_T, DSAMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def _is_kpool_metadata_fusion_supported(
    index_kpool: int, page_size: int, index_topk: int
) -> bool:
    return (
        index_kpool > 1
        and page_size == 64
        and page_size % index_kpool == 0
        and index_topk % index_kpool == 0
    )


@dataclass
class _KPoolForwardInputs:
    full_real_page_table: Optional[torch.Tensor] = None
    full_seqlens_expanded: Optional[torch.Tensor] = None


class DeepseekSparseAttnBackendKPoolMixin:
    """KPool-specific metadata and tail handling for the DSA backend."""

    def _check_kpool_tail_backend(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
        phase: str,
    ) -> None:
        if (
            topk_indices is None
            or self.dsa_index_kpool <= 1
            or dsa_impl in ("fa3", "tilelang", "trtllm")
        ):
            return
        raise NotImplementedError(
            "index_kpool > 1 appends tail tokens to topk_indices and is "
            f"currently only supported by the FA3/TileLang/TRTLLM DSA {phase} "
            "backend."
        )

    def _resolve_kpool_tail_backend(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
    ) -> _DSA_IMPL_T:
        if (
            topk_indices is None
            or self.dsa_index_kpool <= 1
            or dsa_impl != "flashmla_sparse"
        ):
            return dsa_impl
        if self.device_sm_major >= 10:
            return "trtllm"
        if self.device_sm_major == 9:
            return "fa3"
        return dsa_impl

    def _kpool_slots_per_page(self) -> int:
        return getattr(self.token_to_kv_pool, "slots_per_page", self.real_page_size)

    def _build_kpool_paged_mqa_schedule_metadata(self) -> bool:
        if self.device_sm_major == 9:
            return self.num_q_heads in (32, 64)
        return True

    def _init_kpool_metadata(
        self,
        metadata: DSAMetadata,
        forward_batch: ForwardBatch,
        topk_transform_method: Optional[TopkTransformMethod] = None,
        kpool_inputs: Optional[_KPoolForwardInputs] = None,
    ) -> DSAMetadata:
        if self.dsa_index_kpool <= 1:
            return metadata

        forward_mode = forward_batch.forward_mode
        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_extend_without_speculative():
            assert topk_transform_method is not None
            assert kpool_inputs is not None
            return init_kpool_extend_metadata(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                topk_transform_method=topk_transform_method,
                full_real_page_table=kpool_inputs.full_real_page_table,
                full_seqlens_expanded=kpool_inputs.full_seqlens_expanded,
            )

        if forward_mode.is_decode_or_idle():
            metadata = init_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=1,
                write_start=(forward_batch.seq_lens - 1).to(torch.int32),
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if forward_mode.is_target_verify():
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=forward_batch.seq_lens.to(torch.int32),
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if forward_mode.is_draft_extend_v2():
            spec_info = forward_batch.spec_info
            effective_n_per_batch = (
                spec_info.num_accept_tokens
                if spec_info is not None
                and getattr(spec_info, "num_accept_tokens", None) is not None
                else None
            )
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=(
                    forward_batch.seq_lens - self.speculative_num_draft_tokens
                ).to(torch.int32),
                slots_per_page=slots_per_page,
                effective_n_per_batch=effective_n_per_batch,
                build_schedule_metadata=build_schedule_metadata,
            )

        return metadata

    def _init_kpool_metadata_capture(
        self, metadata: DSAMetadata, bs: int, forward_mode: ForwardMode
    ) -> DSAMetadata:
        if self.dsa_index_kpool <= 1:
            return metadata

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            metadata = init_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            is_decode = forward_mode.is_decode_or_idle()
            is_v2 = forward_mode.is_draft_extend_v2()
            metadata = init_kpool_write_plan_capture(
                metadata,
                max_bs=bs,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                num_draft_tokens=(
                    1 if is_decode else self.speculative_num_draft_tokens
                ),
                device=self.device,
                is_verify=not is_decode,
                slots_per_page=slots_per_page,
                is_v2=is_v2,
                build_schedule_metadata=build_schedule_metadata,
            )

        return metadata

    def _update_kpool_metadata_replay(
        self,
        metadata: DSAMetadata,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        effective_n_per_batch: Optional[torch.Tensor] = None,
    ) -> None:
        if self.dsa_index_kpool <= 1:
            return

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            update_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if not (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            return

        is_decode = forward_mode.is_decode_or_idle()
        is_v2 = forward_mode.is_draft_extend_v2()
        if is_decode:
            write_start = seq_lens.to(torch.int32) - 1
        elif is_v2:
            write_start = seq_lens.to(torch.int32) - self.speculative_num_draft_tokens
        else:
            # Target verify: write_start == seq_lens exactly; the plan kernel
            # casts on load, so skip the per-replay int32 alloc + conversion.
            write_start = seq_lens
        update_kpool_write_plan(
            metadata,
            write_start=write_start,
            req_pool_indices=req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=(1 if is_decode else self.speculative_num_draft_tokens),
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
            effective_n_per_batch=effective_n_per_batch,
        )

    def _update_kpool_metadata_from_precomputed(
        self,
        metadata: DSAMetadata,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ) -> None:
        if self.dsa_index_kpool <= 1:
            return

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            update_pooled_paged_mqa_metadata(
                metadata,
                precomputed.cache_seqlens,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if not (forward_mode.is_decode_or_idle() or forward_mode.is_target_verify()):
            return

        is_verify = forward_mode.is_target_verify()
        write_start = precomputed.cache_seqlens.to(torch.int32)
        write_start = (
            write_start - self.speculative_num_draft_tokens
            if is_verify
            else write_start - 1
        )
        update_kpool_write_plan(
            metadata,
            write_start=write_start,
            req_pool_indices=precomputed.req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=self.speculative_num_draft_tokens if is_verify else 1,
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
        )

    def _copy_kpool_metadata_from_sibling(
        self, metadata: DSAMetadata, src_metadata: DSAMetadata
    ) -> None:
        """Copy KPool metadata derived from identical inputs from a sibling."""
        if self.dsa_index_kpool <= 1 or not is_cuda():
            return

        if metadata.pooled_cache_seqlens_int32 is not None:
            metadata.pooled_cache_seqlens_int32.copy_(
                src_metadata.pooled_cache_seqlens_int32
            )
        if metadata.pooled_real_page_table is not None:
            metadata.pooled_real_page_table.copy_(src_metadata.pooled_real_page_table)
        if metadata.pooled_paged_mqa_schedule_metadata is not None:
            metadata.pooled_paged_mqa_schedule_metadata.copy_(
                src_metadata.pooled_paged_mqa_schedule_metadata
            )

        dst_plan = metadata.kpool_write_plan
        src_plan = src_metadata.kpool_write_plan
        if dst_plan is None:
            return
        dst_plan.req.copy_(src_plan.req)
        dst_plan.write_start.copy_(src_plan.write_start)
        dst_plan.tail_logical_start.copy_(src_plan.tail_logical_start)
        dst_plan.write_loc.copy_(src_plan.write_loc)
        if dst_plan.pool_seqlens_per_q is not None:
            dst_plan.pool_seqlens_per_q.copy_(src_plan.pool_seqlens_per_q)
        if dst_plan.seqlens_per_q is not None:
            dst_plan.seqlens_per_q.copy_(src_plan.seqlens_per_q)
        if dst_plan.pool_schedule_metadata is not None:
            dst_plan.pool_schedule_metadata.copy_(src_plan.pool_schedule_metadata)
        if dst_plan.effective_n_per_batch is not None:
            dst_plan.effective_n_per_batch.copy_(src_plan.effective_n_per_batch)
