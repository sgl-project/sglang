from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.utils.common import async_h2d_stack

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def build_standalone_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    """Rebuild the prefill draft's seed without reading target hidden states.

    The draft KV is transferred through the existing PD buffers; this only
    initializes the next draft step after decode admits the transferred request.
    """
    topk = get_spec().speculative_eagle_topk
    topk_p_rows = []
    topk_index_rows = []
    for req in batch.reqs:
        if req.output_topk_p is None or req.output_topk_index is None:
            raise ValueError(f"Missing STANDALONE PD draft seed for request {req.rid}")
        probabilities = torch.as_tensor(req.output_topk_p, dtype=torch.float32)
        indices = torch.as_tensor(req.output_topk_index, dtype=torch.int64)
        if (
            probabilities.ndim != 1
            or indices.shape != probabilities.shape
            or not 0 < topk <= probabilities.shape[0]
        ):
            raise ValueError("Invalid STANDALONE PD draft seed shape")
        probabilities = probabilities[:topk]
        indices = indices[:topk]
        if (
            not bool(torch.all(torch.isfinite(probabilities) & (probabilities >= 0)))
            or not bool(torch.any(probabilities > 0))
            or not bool(
                torch.all((indices >= 0) & (indices < batch.model_config.vocab_size))
            )
        ):
            raise ValueError("Invalid or unwritten STANDALONE PD draft seed")
        topk_p_rows.append(probabilities)
        topk_index_rows.append(indices)

    spec_info = EagleDraftInput(
        topk_p=async_h2d_stack(topk_p_rows, device=batch.device),
        topk_index=async_h2d_stack(topk_index_rows, device=batch.device),
        hidden_states=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        bonus_tokens=last_tokens_tensor,
    )
    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )
    return spec_info
