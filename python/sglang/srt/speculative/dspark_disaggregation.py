from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    draft_state_in_future_map,
)
from sglang.srt.speculative.dspark_components.dspark_draft import make_next_draft_input

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpecInput


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> SpecInput:
    spec_info = make_next_draft_input(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )
    if draft_state_in_future_map(batch.enable_overlap):
        spec_info.future_dsa_topk_indices_available = False
        spec_info.future_indices = batch.req_pool_indices
        future_map.seed(
            spec_info.future_indices,
            batch.seq_lens,
            RelayPayload.from_draft_input(spec_info),
            indices_cpu=batch.req_pool_indices_cpu,
        )
    return spec_info
