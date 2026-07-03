from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.dspark_info import DSparkDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DSparkDraftInputV2:
    del server_args
    warmup_rounds = max(0, _env_int("SGLANG_DSPARK_TRANSFER_WARMUP_ROUNDS", 0))
    req_hidden_states = [req.hidden_states_tensor for req in batch.reqs]
    if all(hidden is not None for hidden in req_hidden_states):
        hidden_states = torch.stack(req_hidden_states, dim=0).to(batch.device)
    else:
        hidden_states = torch.empty((0, 0), dtype=torch.float16, device=batch.device)

    spec_info = DSparkDraftInputV2(
        bonus_tokens=last_tokens_tensor.to(dtype=torch.int64),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
        cur_allocated_seq_lens_cpu=batch.seq_lens_cpu,
        topk_p=torch.empty((0, 0), dtype=torch.float32, device=batch.device),
        topk_index=torch.empty((0, 0), dtype=torch.int64, device=batch.device),
        hidden_states=hidden_states,
        transfer_warmup_rounds=torch.full(
            (batch.batch_size(),),
            warmup_rounds,
            dtype=torch.int32,
            device=batch.device,
        ),
    )

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices,
            RelayPayload(
                bonus_tokens=spec_info.bonus_tokens,
                hidden_states=spec_info.hidden_states,
                transfer_warmup_rounds=spec_info.transfer_warmup_rounds,
            ),
        )

    return spec_info
