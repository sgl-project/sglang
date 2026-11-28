from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


def num_prefill_rows(batch: ScheduleBatch) -> int:
    """The leading rows of the batch that prefill; the rest decode. 0 unless the
    batch extends."""
    if not batch.forward_mode.is_extend_without_speculative():
        return 0
    if batch.forward_mode.is_mixed():
        return len(batch.reqs) - len(batch.mix_running_indices_cpu)
    if batch.decoding_reqs is not None:
        return 0
    return len(batch.reqs)


def prefill_completes(req: Req) -> bool:
    """Whether the request's prefill row runs its last chunk in this step."""
    return req.extend_end == len(req.full_untruncated_fill_ids)
