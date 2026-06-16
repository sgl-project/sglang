from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    bs = len(batch.reqs)
    device = batch.device
    cur_allocated_seq_lens_cpu = batch.seq_lens_cpu.to(torch.int32)

    spec_info = DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
        topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
        verified_id=last_tokens_tensor.to(dtype=torch.int32),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
        verify_done=None,
        cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
    )

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(spec_info.future_indices, spec_info)

    return spec_info
