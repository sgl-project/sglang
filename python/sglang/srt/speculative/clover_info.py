# SPDX-License-Identifier: Apache-2.0
"""Draft state and preview/verify metadata for Clover."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.speculative.clover_penalty import CloverPenaltyState


@dataclass
class CloverDraftInput(DFlashDraftInputV2):
    penalty_state: "CloverPenaltyState | None" = field(
        default=None, init=False, repr=False
    )

    def prepare_for_decode(self, batch):
        from sglang.srt.speculative.clover_penalty import (
            get_clover_penalty_state,
        )

        self.penalty_state = None
        super().prepare_for_decode(batch)
        orchestrator = batch.sampling_info.penalizer_orchestrator
        self.penalty_state = get_clover_penalty_state(orchestrator)
        self.penalty_state.sync_native(orchestrator)

    def __post_init__(self):
        super().__post_init__()
        self.spec_input_type = SpecInputType.CLOVER_DRAFT

    @classmethod
    def from_tokens(cls, bonus_tokens, new_seq_lens):
        device = bonus_tokens.device
        bs = bonus_tokens.numel()
        return cls(
            topk_p=torch.empty((bs, 0), device=device),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            bonus_tokens=bonus_tokens.clone(),
            new_seq_lens=new_seq_lens.clone(),
            hidden_states=torch.empty((bs, 0), device=device),
        )


@dataclass
class CloverPreviewInput(SpecInput):
    """Fixed-width noncausal draft forward; KV lengths already include the block.

    Graph replay rebuilds indices from the current batch's KV plan, including
    padded request slots.
    """

    num_tokens_per_req: int
    hidden_states: torch.Tensor | None = None

    @property
    def draft_token_num(self):
        return self.num_tokens_per_req

    def __post_init__(self):
        super().__init__(SpecInputType.CLOVER_DRAFT)
        if self.num_tokens_per_req <= 0:
            raise ValueError("Clover preview width must be positive")
        self.num_tokens_for_logprob_per_req = self.num_tokens_per_req

    def generate_attn_arg_prefill(
        self,
        *,
        req_pool_indices,
        paged_kernel_lens,
        paged_kernel_lens_sum,
        translator,
        plan,
        kind=IdSpaceKind.FULL,
    ):
        bs = req_pool_indices.numel()
        device = req_pool_indices.device
        width = self.num_tokens_per_req
        qo_indptr = torch.arange(
            0, (bs + 1) * width, width, dtype=torch.int32, device=device
        )
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indices = torch.empty(
            paged_kernel_lens_sum + 256, dtype=torch.int32, device=device
        )
        translator.pack_read_stream(
            plan,
            req_pool_indices=req_pool_indices,
            seq_lens=paged_kernel_lens,
            indptr=kv_indptr,
            out=kv_indices,
            kind=kind,
        )
        return kv_indices, kv_indptr, qo_indptr, None


@dataclass
class CloverVerifyInput(DFlashVerifyInput):
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    def __post_init__(self):
        super().__post_init__()
        self.spec_input_type = SpecInputType.CLOVER_VERIFY

    def prepare_for_verify(self, batch, target_worker):
        forward_batch, can_run_graph = super().prepare_for_verify(batch, target_worker)
        # The inherited GPU path already loaded graph buffers or eager metadata.
        if not batch.forward_mode.is_idle():
            forward_batch.mark_forward_metadata_ready(replan_equivalent=True)
        return forward_batch, can_run_graph
