"""Pre-sample capture of beam logits in the TP worker's forward path.

The sampler rewrites next_token_logits in place (temperature/softmax), and
the scheduler-side joint selection runs later, at the relay point -- so the
raw logits it needs are preserved here, before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class BeamLogitsCapture:
    """The preserved raw logits, stashed on LogitsProcessorOutput.beam."""

    leader_logits: torch.Tensor  # pre-sample clone of the leader rows
    tail_logits: Optional[torch.Tensor] = None  # member rows' slice (decode)
    leader_rows: Optional[List[int]] = None  # leaders' batch indices (extend)


def capture_pre_sample_logits(
    batch: Optional[ScheduleBatch],
    forward_batch: ForwardBatch,
    logits_output: Optional[LogitsProcessorOutput],
) -> None:
    """Decode: split the member-row tail off logits/hidden_states/positions
    and clone the leader rows. Extend: clone the beam leaders' rows. No-op
    for a forward without logits or without beam rows."""
    if (
        batch is None
        or logits_output is None
        or logits_output.next_token_logits is None
    ):
        return
    if batch.beam_tail is not None:
        # The sampler and every consumer downstream are reqs-aligned, so the
        # member tail is split off their view. Outside that view the tail's
        # raw logits survive the in-place sampling writes as-is; the leader
        # rows do not, hence the clone.
        n = batch.beam_tail.num_base_rows
        logits = logits_output.next_token_logits
        logits_output.next_token_logits = logits[:n]
        leader_rows = [e[1] for e in batch.beam_tail.entries]
        logits_output.beam = BeamLogitsCapture(
            leader_logits=logits[leader_rows].clone(),
            tail_logits=logits[n:],
        )
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[:n]
        forward_batch.positions = forward_batch.positions[:n]
    elif forward_batch.forward_mode.is_extend():
        # The leaders' first selection reads these prefill logits at the
        # relay point, after sampling -- clone before it clobbers them.
        leader_rows = [i for i, r in enumerate(batch.reqs) if r.beam_group is not None]
        if leader_rows:
            logits_output.beam = BeamLogitsCapture(
                leader_logits=logits_output.next_token_logits[leader_rows].clone(),
                leader_rows=leader_rows,
            )
