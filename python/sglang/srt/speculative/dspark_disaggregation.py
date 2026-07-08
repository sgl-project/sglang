from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dspark_info import DSparkDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_dspark_disagg_warmup_rounds() -> int:
    if _env_flag("SGLANG_DSPARK_DEEPSPEC_PREFILL_HANDOFF", True):
        return 0

    value = os.getenv("SGLANG_DSPARK_PREFILL_TRANSFER_WARMUP_ROUNDS")
    if value is None:
        # Backward-compatible alias used by the early PD handoff path.
        value = os.getenv("SGLANG_DSPARK_TRANSFER_WARMUP_ROUNDS")
    if value is None:
        return 1
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DSparkDraftInputV2:
    del server_args
    warmup_rounds = _get_dspark_disagg_warmup_rounds()
    req_hidden_states = [req.hidden_states_tensor for req in batch.reqs]
    if all(hidden is not None for hidden in req_hidden_states):
        hidden_states = torch.stack(req_hidden_states, dim=0).to(batch.device)
        hidden_valid_mask = torch.ones(
            (batch.batch_size(),), dtype=torch.bool, device=batch.device
        )
    else:
        hidden_states = torch.empty((0, 0), dtype=torch.float16, device=batch.device)
        hidden_valid_mask = torch.empty((0, 0), dtype=torch.bool, device=batch.device)

    req_tail_hidden = [
        getattr(req, "prefill_tail_hidden_states_tensor", None) for req in batch.reqs
    ]
    req_tail_mask = [
        getattr(req, "prefill_tail_valid_mask", None) for req in batch.reqs
    ]
    if (
        req_tail_hidden
        and all(tail is not None for tail in req_tail_hidden)
        and all(mask is not None for mask in req_tail_mask)
    ):
        prefill_tail_hidden_states = torch.stack(req_tail_hidden, dim=0).to(
            batch.device
        )
        prefill_tail_valid_mask = torch.stack(req_tail_mask, dim=0).to(batch.device)
    else:
        prefill_tail_hidden_states = torch.empty(
            (0, 0, 0), dtype=torch.float16, device=batch.device
        )
        prefill_tail_valid_mask = torch.empty(
            (0, 0), dtype=torch.bool, device=batch.device
        )

    spec_info = DSparkDraftInputV2(
        bonus_tokens=last_tokens_tensor.to(dtype=torch.int64),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
        cur_allocated_seq_lens_cpu=batch.seq_lens_cpu,
        topk_p=torch.empty((0, 0), dtype=torch.float32, device=batch.device),
        topk_index=torch.empty((0, 0), dtype=torch.int64, device=batch.device),
        hidden_states=hidden_states,
        hidden_valid_mask=hidden_valid_mask,
        prefill_tail_hidden_states=prefill_tail_hidden_states,
        prefill_tail_valid_mask=prefill_tail_valid_mask,
        prefill_tail_hidden_projected=False,
        transfer_warmup_rounds=torch.full(
            (batch.batch_size(),),
            warmup_rounds,
            dtype=torch.int32,
            device=batch.device,
        ),
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.FULL

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices,
            RelayPayload(
                bonus_tokens=spec_info.bonus_tokens,
                hidden_states=spec_info.hidden_states,
                hidden_valid_mask=spec_info.hidden_valid_mask,
                prefill_tail_hidden_states=spec_info.prefill_tail_hidden_states,
                prefill_tail_valid_mask=spec_info.prefill_tail_valid_mask,
                transfer_warmup_rounds=spec_info.transfer_warmup_rounds,
            ),
        )

    return spec_info
