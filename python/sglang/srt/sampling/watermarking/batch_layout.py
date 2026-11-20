from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def num_prefill_rows(batch: ScheduleBatch) -> int:
    """The leading rows of an extend batch that prefill; the rest decode."""
    if batch.forward_mode.is_mixed():
        return len(batch.reqs) - len(batch.mix_running_indices_cpu)
    if batch.decoding_reqs is not None:
        return 0
    return len(batch.reqs)
