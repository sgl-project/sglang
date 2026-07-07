from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        # Seed the relay buf with the known seq_lens; not a forward-completion
        # signal, so skip the publish_ready record (see FutureMap.publish).
        future_map.publish(spec_info.future_indices, batch.seq_lens, record_event=False)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )

    return spec_info
