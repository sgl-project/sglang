from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.dspark_info import DSparkDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DSparkDraftInputV2:
    del server_args
    hidden_tensors = [
        getattr(req, "hidden_states_tensor", None) for req in batch.reqs
    ]
    has_main_hidden = all(hidden is not None for hidden in hidden_tensors)
    main_hidden = (
        torch.stack(hidden_tensors, dim=0).to(batch.device)
        if has_main_hidden and len(hidden_tensors) > 0
        else None
    )

    spec_info = DSparkDraftInputV2(
        bonus_tokens=last_tokens_tensor.to(dtype=torch.int64),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
        main_hidden=main_hidden,
        main_hidden_mask=(
            torch.ones(
                (len(hidden_tensors),), dtype=torch.bool, device=batch.device
            )
            if main_hidden is not None
            else None
        ),
        cur_allocated_seq_lens_cpu=batch.seq_lens_cpu,
        topk_p=torch.empty((0, 0), dtype=torch.float32, device=batch.device),
        topk_index=torch.empty((0, 0), dtype=torch.int64, device=batch.device),
        hidden_states=torch.empty((0, 0), dtype=torch.float16, device=batch.device),
    )

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices,
            RelayPayload(bonus_tokens=spec_info.bonus_tokens),
        )

    return spec_info
