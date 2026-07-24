from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.dsa.utils import (
    should_remap_pd_dsa_seed_to_local_slots,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    num_states = server_args.speculative_eagle_topk
    if server_args.enable_multi_layer_eagle:
        num_states *= server_args.speculative_num_steps

    topk_p = torch.stack(
        [
            torch.as_tensor(
                req.output_topk_p[:num_states],
                device=batch.device,
                dtype=torch.float32,
            )
            for req in batch.reqs
        ],
        dim=0,
    )
    topk_index = torch.stack(
        [
            torch.as_tensor(
                req.output_topk_index[:num_states],
                device=batch.device,
                dtype=torch.int64,
            )
            for req in batch.reqs
        ],
        dim=0,
    )

    hidden_states = torch.stack(
        [req.hidden_states_tensor for req in batch.reqs], dim=0
    ).to(batch.device)

    dsa_topk_indices = None
    dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
    if dsa_indices_list and all(t is not None for t in dsa_indices_list):
        dsa_topk_indices = torch.stack(dsa_indices_list, dim=0).to(batch.device)
        if should_remap_pd_dsa_seed_to_local_slots(server_args):
            # PD transports request-relative positions because the prefill and
            # decode allocators are independent. The fused TopK path consumes
            # physical slots, so materialize them once through the decode-local
            # page table before the seed enters the draft loop/CUDA graph.
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
