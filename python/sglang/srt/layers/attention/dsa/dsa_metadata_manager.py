"""DSA metadata fusion selection and MTP replay reuse."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from sglang.kernels.ops.attention.dsa_metadata import (
    fused_dsa_decode_metadata,
    fused_dsa_draft_extend_metadata,
    fused_dsa_target_verify_metadata,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
        PrecomputedMetadata,
    )
    from sglang.srt.layers.attention.dsa_backend import (
        DeepseekSparseAttnBackend,
        DSAMetadata,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_is_hip = is_hip()

logger = logging.getLogger(__name__)


def kpool_metadata_fusion_supported(pool_size, page_size, topk):
    return (
        pool_size > 1
        and page_size == 64
        and page_size % pool_size == 0
        and topk % pool_size == 0
    )


class DSAMetadataManagementMixin:
    experimental_kpool_metadata_fusion = False

    def _init_kpool_metadata_fusion(self):
        requested = envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION.get()
        supported = kpool_metadata_fusion_supported(
            self.dsa_index_kpool, self.real_page_size, self.dsa_index_topk
        )
        self.experimental_kpool_metadata_fusion = (
            requested and supported and is_cuda() and not is_hip()
        )
        self._fused_decode_metadata = fused_dsa_decode_metadata
        self._fused_verify_metadata = fused_dsa_target_verify_metadata
        self._fused_draft_extend_metadata = fused_dsa_draft_extend_metadata
        if self.experimental_kpool_metadata_fusion:
            from sglang.kernels.ops.attention.dsa_kpool_metadata.decode import (
                fused_dsa_decode_metadata as decode,
            )
            from sglang.kernels.ops.attention.dsa_kpool_metadata.draft_extend import (
                fused_dsa_draft_extend_metadata as draft_extend,
            )
            from sglang.kernels.ops.attention.dsa_kpool_metadata.verify import (
                fused_dsa_target_verify_metadata as verify,
            )

            self._fused_decode_metadata = partial(
                decode, index_kpool=self.dsa_index_kpool
            )
            self._fused_verify_metadata = partial(
                verify, index_kpool=self.dsa_index_kpool
            )
            self._fused_draft_extend_metadata = partial(
                draft_extend, index_kpool=self.dsa_index_kpool
            )
            logger.info(
                "DSA KPool metadata fusion enabled (pool=%d)", self.dsa_index_kpool
            )
        elif requested and self.dsa_index_kpool > 1:
            logger.warning(
                "DSA KPool metadata fusion unsupported for this platform/geometry; retaining ordinary metadata"
            )

    def _copy_base_replay_buffers(self, bs, metadata, precomputed, forward_mode):
        # Track whether fused kernel succeeded
        fused_kernel_succeeded = False

        # Use fused CUDA kernel for all copy operations
        if not _is_hip:
            try:
                from sglang.kernels.ops.attention.fused_metadata_copy import (
                    fused_metadata_copy_cuda,
                )

                # Map forward_mode to integer enum
                if forward_mode.is_decode_or_idle():
                    mode_int = 0  # DECODE
                elif forward_mode.is_target_verify():
                    mode_int = 1  # TARGET_VERIFY
                else:
                    raise ValueError(f"Unsupported forward_mode: {forward_mode}")

                # Prepare FlashMLA tensors if needed
                flashmla_num_splits_src = None
                flashmla_num_splits_dst = None
                flashmla_metadata_src = None
                flashmla_metadata_dst = None
                if precomputed.flashmla_metadata is not None:
                    flashmla_num_splits_src = precomputed.flashmla_metadata.num_splits
                    flashmla_num_splits_dst = metadata.flashmla_metadata.num_splits
                    flashmla_metadata_src = (
                        precomputed.flashmla_metadata.flashmla_metadata
                    )
                    flashmla_metadata_dst = metadata.flashmla_metadata.flashmla_metadata

                # Call fused kernel
                fused_metadata_copy_cuda(
                    # Source tensors
                    precomputed.cache_seqlens,
                    precomputed.cu_seqlens_k,
                    precomputed.page_indices,
                    precomputed.dsa_cache_seqlens,
                    precomputed.seqlens_expanded,
                    precomputed.dsa_cu_seqlens_k,
                    precomputed.real_page_table,
                    flashmla_num_splits_src,
                    flashmla_metadata_src,
                    # Destination tensors
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table_1,
                    metadata.dsa_cache_seqlens_int32,
                    metadata.dsa_seqlens_expanded,
                    metadata.dsa_cu_seqlens_k,
                    (
                        metadata.real_page_table
                        if precomputed.real_page_table is not None
                        else None
                    ),
                    flashmla_num_splits_dst,
                    flashmla_metadata_dst,
                    # Parameters
                    mode_int,
                    bs,
                    precomputed.max_len,
                    precomputed.max_seqlen_k,
                    precomputed.seqlens_expanded_size,
                )

                # Successfully used fused kernel
                fused_kernel_succeeded = True

            except ImportError:
                print(
                    "Warning: Fused metadata copy kernel not available, falling back to individual copies."
                )
            except Exception as e:
                print(
                    f"Warning: Fused metadata copy kernel failed with error: {e}, falling back to individual copies."
                )

        # Fallback to individual copy operations if the fused kernel is unavailable
        # or fails at runtime.
        if not fused_kernel_succeeded:
            # Copy basic seqlens
            metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

            # Mode-specific copy logic
            if forward_mode.is_decode_or_idle():
                # Decode mode
                metadata.page_table_1[:, : precomputed.max_len].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)
                # seqlens_expanded is same as cache_seqlens (already copied)

            elif forward_mode.is_target_verify():
                # Target verify mode
                metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)

            # Copy DSA cu_seqlens
            size = precomputed.seqlens_expanded_size
            metadata.dsa_cu_seqlens_k[1 : 1 + size].copy_(
                precomputed.dsa_cu_seqlens_k[1 : 1 + size]
            )

            # Copy real page table
            if precomputed.real_page_table is not None:
                rows, cols = precomputed.real_page_table.shape
                metadata.real_page_table[:rows, :cols].copy_(
                    precomputed.real_page_table
                )

            # Copy FlashMLA metadata in fallback path
            if precomputed.flashmla_metadata is not None:
                size = precomputed.seqlens_expanded_size
                flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
                flashmla_metadata.copy_(precomputed.flashmla_metadata)

    @staticmethod
    def _sibling_replay_metadata_compatible(dst: DSAMetadata, src: DSAMetadata) -> bool:
        """Check that both sides expose the same optional derived buffers."""

        def _match(a, b) -> bool:
            return (a is None) == (b is None)

        if not (
            _match(dst.paged_mqa_schedule_metadata, src.paged_mqa_schedule_metadata)
            and _match(dst.topk_v2_plan, src.topk_v2_plan)
            and _match(dst.pooled_cache_seqlens_int32, src.pooled_cache_seqlens_int32)
            and _match(dst.pooled_real_page_table, src.pooled_real_page_table)
            and _match(
                dst.pooled_paged_mqa_schedule_metadata,
                src.pooled_paged_mqa_schedule_metadata,
            )
            and _match(dst.kpool_write_plan, src.kpool_write_plan)
        ):
            return False
        dst_plan, src_plan = dst.kpool_write_plan, src.kpool_write_plan
        if dst_plan is not None and not (
            _match(dst_plan.pool_seqlens_per_q, src_plan.pool_seqlens_per_q)
            and _match(dst_plan.seqlens_per_q, src_plan.seqlens_per_q)
            and _match(dst_plan.pool_schedule_metadata, src_plan.pool_schedule_metadata)
            and _match(dst_plan.effective_n_per_batch, src_plan.effective_n_per_batch)
        ):
            return False
        return True

    def _copy_replay_metadata_from_sibling(
        self,
        src_backend: DeepseekSparseAttnBackend,
        bs: int,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ) -> None:
        """Copy replay metadata from a sibling using the same precomputed input."""
        metadata = self.decode_cuda_graph_metadata.get(bs)
        src_metadata = src_backend.decode_cuda_graph_metadata.get(bs)
        if (
            # The derived-copy body below is CUDA-only; any other platform
            # must take the full recompute, not a partial copy that would
            # leave the DeepGEMM schedule / top-k plan / kpool metadata
            # stale.
            not is_cuda()
            or _is_hip
            or not forward_mode.is_decode_or_idle()
            or metadata is None
            or src_metadata is None
            # `src_backend` must have run the full recompute path for this bs
            # in this replay, so its derived buffers are fresh.
            or src_backend.forward_metadata is not src_metadata
            or not self._sibling_replay_metadata_compatible(metadata, src_metadata)
        ):
            self.init_forward_metadata_replay_cuda_graph_from_precomputed(
                bs=bs, precomputed=precomputed, forward_mode=forward_mode
            )
            return

        self.set_dsa_prefill_impl(forward_batch=None)
        self._copy_base_replay_buffers(bs, metadata, precomputed, forward_mode)

        if is_cuda():
            if metadata.paged_mqa_schedule_metadata is not None:
                metadata.paged_mqa_schedule_metadata.copy_(
                    src_metadata.paged_mqa_schedule_metadata
                )
            if metadata.topk_v2_plan is not None:
                metadata.topk_v2_plan.copy_(src_metadata.topk_v2_plan)
            # Decode: the 2D ctx lens are a (bs, 1) view of this backend's own
            # cache_seqlens_int32 (just refreshed by the base copy above); keep
            # the exact refresh the recompute path performs -- it is a single
            # small view/copy, not part of the duplicated derived work.
            seqlens_32_2d = metadata.cache_seqlens_int32.contiguous().view(bs, 1)
            if metadata.paged_mqa_ctx_lens_2d is None:
                object.__setattr__(metadata, "paged_mqa_ctx_lens_2d", seqlens_32_2d)
            else:
                metadata.paged_mqa_ctx_lens_2d.copy_(seqlens_32_2d)

        self._copy_kpool_metadata_from_sibling(metadata, src_metadata)

        self.forward_metadata = metadata

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
