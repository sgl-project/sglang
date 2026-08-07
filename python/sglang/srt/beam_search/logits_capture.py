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
        # Split the beam member rows off before sampling so the
        # sampler and every reqs-aligned consumer downstream see
        # only the reqs-aligned rows. The tail then sits outside
        # the sampler's view and keeps its raw logits for the
        # scheduler-side selection; the leader rows do not, hence
        # the pre-sample clone.
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
        # Same reason, prefill side: the leader's first selection
        # runs at the relay point, after the sampler's in-place
        # writes would have clobbered these logits.
        leader_rows = [i for i, r in enumerate(batch.reqs) if r.beam_group is not None]
        if leader_rows:
            logits_output.beam = BeamLogitsCapture(
                leader_logits=logits_output.next_token_logits[leader_rows].clone(),
                leader_rows=leader_rows,
            )
