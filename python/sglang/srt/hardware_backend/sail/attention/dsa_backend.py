from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.sail.attention.flash_mla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    PrecomputedMetadata,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAFlashMLAMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner


class SailDSAFlashMLAMetadata(DSAFlashMLAMetadata):
    """FlashMLA metadata produced by the SAIL FlashMLA build.

    ``flashmla_metadata`` is a scheduler metadata object that the kernel fills
    in lazily on its first call, so there is nothing to slice or copy into the
    CUDA graph buffers.
    """

    def slice(self, sli):
        return SailDSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits,
        )

    def copy_(self, other: DSAFlashMLAMetadata):
        return


class SailDeepseekSparseAttnBackend(DeepseekSparseAttnBackend):
    """DSA backend running on the SAIL FlashMLA build."""

    def _call_flash_mla_sparse_fwd(self, **kwargs):
        return flash_mla_sparse_fwd(**kwargs)

    def _call_flash_mla_with_kvcache(self, **kwargs):
        return flash_mla_with_kvcache(**kwargs)

    def _call_get_mla_metadata(self, **kwargs):
        return get_mla_metadata(**kwargs)

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        metadata = super()._compute_flashmla_metadata(cache_seqlens, seq_len_q)
        return SailDSAFlashMLAMetadata(
            flashmla_metadata=metadata.flashmla_metadata,
            num_splits=metadata.num_splits,
        )

    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> PrecomputedMetadata:
        precomputed = super()._precompute_replay_metadata(
            bs, req_pool_indices, seq_lens, seq_lens_cpu, forward_mode
        )
        # The scheduler metadata is regenerated inside the graph on replay;
        # there are no tensors to share across the multi-step backends.
        precomputed.flashmla_metadata = None
        return precomputed

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        # Reset the scheduler metadata before capture so that its lazy
        # initialization in the first kernel call is recorded into the graph.
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self.forward_metadata.flashmla_metadata
            assert flashmla_metadata is not None
            assert flashmla_metadata.flashmla_metadata is not None
            flashmla_metadata.flashmla_metadata.have_initialized = False


class SailDeepseekSparseAttnMultiStepBackend(DeepseekSparseAttnMultiStepBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        seed_dsa_topk_from_draft_extend: bool = False,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self._sink_pad_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                SailDeepseekSparseAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    seed_dsa_topk_from_draft_extend=seed_dsa_topk_from_draft_extend,
                )
            )
