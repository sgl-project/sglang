from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    draft_state_in_future_map,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    spec_info = make_draft_input_v2(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )
    if draft_state_in_future_map(batch.enable_overlap):
        spec_info.future_indices = batch.req_pool_indices
        future_map.seed(
            spec_info.future_indices,
            batch.seq_lens,
            RelayPayload(bonus_tokens=last_tokens_tensor),
            indices_cpu=batch.req_pool_indices_cpu,
        )
    return spec_info
