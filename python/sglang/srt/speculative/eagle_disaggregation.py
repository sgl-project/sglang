from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.dsa.utils import (
    should_remap_pd_dsa_seed_to_local_slots,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def _stack_relay_to_device(
    values, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    tensors = [torch.as_tensor(value, dtype=dtype) for value in values]
    stacked = torch.stack(tensors, dim=0)
    if stacked.device.type == "cpu" and is_pin_memory_available(device):
        stacked = stacked.pin_memory()
    return stacked.to(device, non_blocking=True)


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    num_states = server_args.speculative_eagle_topk
    if server_args.enable_multi_layer_eagle:
        num_states *= server_args.speculative_num_steps

    topk_p = _stack_relay_to_device(
        [req.output_topk_p[:num_states] for req in batch.reqs],
        batch.device,
        torch.float32,
    )
    topk_index = _stack_relay_to_device(
        [req.output_topk_index[:num_states] for req in batch.reqs],
        batch.device,
        torch.int64,
    )

    hidden_states = _stack_relay_to_device(
        [req.hidden_states_tensor for req in batch.reqs], batch.device
    )

    dsa_topk_indices = None
    dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
    if dsa_indices_list and all(t is not None for t in dsa_indices_list):
        dsa_topk_indices = _stack_relay_to_device(dsa_indices_list, batch.device)
        if should_remap_pd_dsa_seed_to_local_slots(server_args):
            # PD sends request-relative positions; fused TopK consumes
            # decode-local physical slots. Remap once before the draft loop/graph.
            req_to_token = batch.req_to_token_pool.req_to_token
            table_width = req_to_token.shape[1]
            valid_positions = dsa_topk_indices >= 0
            gather_positions = dsa_topk_indices.clamp(min=0, max=table_width - 1).to(
                torch.int64
            )
            local_slots = req_to_token[
                batch.req_pool_indices[:, None], gather_positions
            ]
            invalid_rows = torch.any(
                (dsa_topk_indices < -1)
                | (dsa_topk_indices >= batch.seq_lens[:, None])
                | (dsa_topk_indices >= table_width)
                # Slot 0 is the reserved padding sink; real KV allocations
                # start at 1, and untouched req-to-token entries remain 0.
                | (valid_positions & (local_slots <= 0)),
                dim=1,
            )
            local_slots.masked_fill_(~valid_positions, -1)
            local_slots.masked_fill_(invalid_rows[:, None], -1)
            dsa_topk_indices = local_slots
        if torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item():
            dsa_topk_indices = None

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
        dsa_topk_indices=dsa_topk_indices,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    if batch.enable_overlap:
        spec_info.future_dsa_topk_indices_available = dsa_topk_indices is not None
        spec_info.future_indices = batch.req_pool_indices
        # Seed the relay buf with the known seq_lens; publish's chained record
        # keeps the in-flight forward's fence intact (see FutureMap.publish).
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )

    return spec_info
