from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.ngram_info import NgramVerifyInput

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def build_ngram_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> NgramVerifyInput:
    # PD prefill already put the first token in req.output_ids, which NGRAM drafts
    # from: accept_lens 0 splices nothing at the first draft prep.
    batch_size = len(batch.reqs)
    draft_token_num = get_spec().speculative_num_draft_tokens
    # int32 like NgramWorker's next_draft_input: the first stash fixes the relay
    # buffers' dtype.
    spec_info = NgramVerifyInput(
        draft_token_num=draft_token_num,
        new_seq_lens=batch.seq_lens,
        accept_tokens=torch.zeros(
            batch_size * draft_token_num, dtype=torch.int32, device=batch.device
        ),
        accept_lens=torch.zeros(batch_size, dtype=torch.int32, device=batch.device),
    )
    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(spec_info.future_indices, RelayPayload.from_ngram(spec_info))
    return spec_info
