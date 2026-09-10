"""Capture DSA target-verify metadata refreshes into the model CUDA graph."""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if is_cuda():
    import deep_gemm

_is_hip = is_hip()
logger = logging.getLogger(__name__)


@cache
def get_plan_topk_v2():
    # Resolving the JIT module at srt import time would initialize CUDA too early.
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

    return plan_topk_v2


class _DSAInGraphVerifyMetadataState:
    """Captured refreshes require fixed input-buffer identities at replay.
    Mismatches must fail because captured nodes would overwrite fallback metadata."""

    __slots__ = (
        "prefill_impl_state",
        "_seq_lens_ptr",
        "_seq_lens_stride",
        "_req_pool_indices_ptr",
        "_req_pool_indices_stride",
    )

    def __init__(
        self,
        *,
        prefill_impl_state: Tuple[bool, str],
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        self.prefill_impl_state = prefill_impl_state
        self._seq_lens_ptr = seq_lens.data_ptr()
        self._seq_lens_stride = seq_lens.stride(0)
        self._req_pool_indices_ptr = req_pool_indices.data_ptr()
        self._req_pool_indices_stride = req_pool_indices.stride(0)

    def matches(self, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor) -> bool:
        # A [:bs] slice keeps the base data_ptr/stride, so the raw incoming
        # graph-runner buffers compare equal to the sliced views recorded at
        # capture time.
        return (
            seq_lens.data_ptr() == self._seq_lens_ptr
            and req_pool_indices.data_ptr() == self._req_pool_indices_ptr
            and seq_lens.stride(0) == self._seq_lens_stride
            and req_pool_indices.stride(0) == self._req_pool_indices_stride
        )


class DSAInGraphVerifyMetadataMixin:
    ingraph_verify_metadata_enabled = False

    def _init_ingraph_verify_metadata(self):
        self.ingraph_verify_metadata_enabled = (
            envs.SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA.get()
        )

    def _replay_ingraph_verify_metadata(
        self, metadata, seq_lens, req_pool_indices, forward_mode
    ):
        if not forward_mode.is_target_verify():
            return False
        state = getattr(metadata, "_ingraph_verify_metadata", None)
        if state is None:
            return False
        if not state.matches(seq_lens, req_pool_indices):
            raise RuntimeError(
                "DSA in-graph verify metadata replay requires the static buffers "
                "for seq_lens and req_pool_indices used during capture"
            )
        self.use_mha, self.dsa_prefill_impl = state.prefill_impl_state
        self.forward_metadata = metadata
        return True

    def _ingraph_verify_metadata_eligible(self) -> bool:
        """Config-level eligibility for the in-graph TARGET_VERIFY refresh.

        The recorded nodes reproduce the fused verify sequence. Configurations
        that would take the unfused fallback
        (kpool > 1 without the fusion gate), DCP, HIP, or flashmla_kv (whose
        per-replay metadata recompute is host-side) stay out-of-graph.
        """
        return (
            self.ingraph_verify_metadata_enabled
            and is_cuda()
            and not _is_hip
            and not get_parallel().dcp_enabled
            and (self.dsa_index_kpool <= 1 or self.experimental_kpool_metadata_fusion)
            and self.dsa_decode_impl != "flashmla_kv"
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Record target-verify refreshes only for capture-safe configurations.
        Replay reads the same static input buffers with updated contents."""
        if not forward_batch.forward_mode.is_target_verify():
            return
        if not self._ingraph_verify_metadata_eligible():
            return
        bs = forward_batch.batch_size
        metadata = self.decode_cuda_graph_metadata.get(bs)
        if metadata is None:
            return
        self._record_ingraph_verify_metadata(
            metadata,
            bs,
            forward_batch.seq_lens[:bs],
            forward_batch.req_pool_indices[:bs],
        )

    def _record_ingraph_verify_metadata(
        self,
        metadata: DSAMetadata,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        """Publish replay state only after every structural check passes.
        Otherwise the generic out-of-graph path remains responsible."""
        if metadata.paged_mqa_schedule_metadata is None:
            return
        next_n = self.speculative_num_draft_tokens
        if not next_n:
            return
        expanded_size = bs * next_n
        max_seqlen_k = self._graph_page_table_width(metadata)

        paged_mqa_ctx_lens_2d = None
        if (
            next_n >= 2
            and get_platform().is_sm100
            and metadata.paged_mqa_ctx_lens_2d is not None
            and metadata.paged_mqa_ctx_lens_2d.dim() == 2
            and metadata.paged_mqa_ctx_lens_2d.size(0) == bs
            and metadata.paged_mqa_ctx_lens_2d.size(1) == next_n
        ):
            paged_mqa_ctx_lens_2d = metadata.paged_mqa_ctx_lens_2d
        ctx_lens_written = paged_mqa_ctx_lens_2d is not None

        if ctx_lens_written:
            # DG-native layout: the fused kernel writes ctx lens straight
            # into the captured (bs, next_n) buffer; the schedule reads it.
            schedule_src_2d = metadata.paged_mqa_ctx_lens_2d
            ctx_lens_copy_src = None
        else:
            if (
                next_n >= 2 and get_platform().is_sm100
            ):  # Degenerate capture (DG-native ctx-lens layout expected but
                # missing); keep the whole refresh out-of-graph.
                return
            seqlens_view = metadata.dsa_seqlens_expanded[:expanded_size]
            if not seqlens_view.is_contiguous():
                return
            schedule_src_2d = seqlens_view.view(-1, 1)
            if metadata.paged_mqa_ctx_lens_2d is None:
                # Verify capture always materializes the 2D ctx-lens buffer;
                # if missing, leave the object.__setattr__ publication to
                # the generic path.
                return
            ctx_lens_copy_src = schedule_src_2d

        self._fused_verify_metadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=self.req_to_token,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_k=metadata.cu_seqlens_k,
            page_table_1=metadata.page_table_1,
            seqlens_expanded=metadata.dsa_seqlens_expanded,
            dsa_cache_seqlens=metadata.dsa_cache_seqlens_int32,
            dsa_cu_seqlens_k=metadata.dsa_cu_seqlens_k,
            real_page_table=metadata.real_page_table,
            bs=bs,
            max_seqlen_k=max_seqlen_k,
            dsa_index_topk=self.dsa_index_topk,
            real_page_size=self.real_page_size,
            next_n=next_n,
            paged_mqa_ctx_lens_2d=paged_mqa_ctx_lens_2d,
        )

        metadata.paged_mqa_schedule_metadata.copy_(
            deep_gemm.get_paged_mqa_logits_metadata(
                schedule_src_2d, 64, deep_gemm.get_num_sms()
            )
        )
        if ctx_lens_copy_src is not None:
            metadata.paged_mqa_ctx_lens_2d.copy_(ctx_lens_copy_src)

        if metadata.topk_v2_plan is not None:
            # A preallocated output avoids allocation during capture.
            get_plan_topk_v2()(metadata.dsa_seqlens_expanded, out=metadata.topk_v2_plan)

        self._update_kpool_metadata_replay(
            metadata,
            seq_lens,
            req_pool_indices,
            ForwardMode.TARGET_VERIFY,
        )

        self.set_dsa_prefill_impl(forward_batch=None)
        state = _DSAInGraphVerifyMetadataState(
            prefill_impl_state=(self.use_mha, self.dsa_prefill_impl),
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
        )
        # Undeclared attribute on the frozen dataclass: dropped by both
        # recapture (fresh DSAMetadata) and dataclasses.replace().
        object.__setattr__(metadata, "_ingraph_verify_metadata", state)
        logger.info(
            "DSA in-graph verify metadata recorded: bs=%d next_n=%d "
            "ctx_lens_written=%s",
            bs,
            next_n,
            ctx_lens_written,
        )
