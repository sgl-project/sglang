# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-sample capture of beam logits in the TP worker's forward path.

The sampler rewrites next_token_logits in place (temperature/softmax), and
the scheduler-side joint selection runs later, at the relay point -- so the
raw logits it needs are preserved here, before sampling.

A beam-only batch skips the sampler: the coordinator selects every row's next
token, so a sampled one would never be read. Nothing then rewrites the logits,
and the capture keeps views instead of list-indexed clones, whose index upload
would sync the host on the forward and stall the launch path behind it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BeamLogitsCapture(msgspec.Struct):
    """The preserved raw logits, stashed on LogitsProcessorOutput.beam."""

    leader_logits: torch.Tensor  # leader rows: a pre-sample clone, a view if unsampled
    tail_logits: Optional[torch.Tensor] = None  # member rows' slice (decode)
    leader_rows: Optional[List[int]] = None  # leaders' batch indices (extend)
    leader_normalizer: Optional[torch.Tensor] = None
    tail_normalizer: Optional[torch.Tensor] = None


def _copy_leader_rows(tensor: torch.Tensor, rows: List[int]) -> torch.Tensor:
    import torch

    if not rows:
        return tensor[:0].clone()
    start = rows[0]
    if all(row == start + index for index, row in enumerate(rows)):
        return tensor[start : start + len(rows)].clone(
            memory_format=torch.contiguous_format
        )
    if tensor.device.type == "cuda":
        indices = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
            tensor.device, non_blocking=True
        )
    else:
        indices = torch.tensor(rows, dtype=torch.int64, device=tensor.device)
    return tensor.index_select(0, indices)


def is_beam_only_batch(batch: ScheduleBatch) -> bool:
    """Every reqs-aligned row is a beam leader, so the coordinator relays every
    row's next token at this forward and the regular sampler has no reader."""
    tail = batch.beam_tail
    if tail is not None:
        return len(tail.entries) == tail.num_base_rows
    mode = batch.forward_mode
    return (
        mode.is_extend()
        and not mode.is_mixed()
        and all(req.beam_group is not None for req in batch.reqs)
    )


def capture_pre_sample_logits(
    batch: Optional[ScheduleBatch],
    forward_batch: ForwardBatch,
    logits_output: Optional[LogitsProcessorOutput],
) -> None:
    """Decode: split the member tail off logits/hidden_states/positions and
    clone the leader rows. Extend: clone the leaders' rows. No-op otherwise.
    A beam-only batch keeps views instead of clones."""
    if (
        batch is None
        or logits_output is None
        or logits_output.next_token_logits is None
    ):
        return
    if batch.beam_tail is not None:
        # Split off the reqs-aligned view: outside it the tail survives the
        # in-place sampling writes, but the leader rows do not, hence the clone.
        n = batch.beam_tail.num_base_rows
        logits = logits_output.next_token_logits
        logits_output.next_token_logits = logits[:n]
        normalizer = logits_output.beam_normalizer
        if is_beam_only_batch(batch):
            # Unsampled, so views stay valid; leaders are rows [0, n) in entry order.
            logits_output.beam = BeamLogitsCapture(
                leader_logits=logits[:n],
                tail_logits=logits[n:],
                leader_normalizer=normalizer[:n] if normalizer is not None else None,
                tail_normalizer=normalizer[n:] if normalizer is not None else None,
            )
        else:
            leader_rows = [e.leader_idx for e in batch.beam_tail.entries]
            logits_output.beam = BeamLogitsCapture(
                leader_logits=_copy_leader_rows(logits, leader_rows),
                tail_logits=logits[n:],
                leader_normalizer=(
                    _copy_leader_rows(normalizer, leader_rows)
                    if normalizer is not None
                    else None
                ),
                tail_normalizer=(
                    normalizer[n:].clone() if normalizer is not None else None
                ),
            )
        logits_output.beam_normalizer = None
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[:n]
        forward_batch.positions = forward_batch.positions[:n]
    elif forward_batch.forward_mode.is_extend():
        # The leaders' first selection reads these prefill logits at the
        # relay point, after sampling -- clone before it clobbers them.
        leader_rows = [i for i, r in enumerate(batch.reqs) if r.beam_group is not None]
        if leader_rows:
            normalizer = logits_output.beam_normalizer
            if is_beam_only_batch(batch):
                logits_output.beam = BeamLogitsCapture(
                    leader_logits=logits_output.next_token_logits,
                    leader_rows=leader_rows,
                    leader_normalizer=normalizer,
                )
            else:
                logits_output.beam = BeamLogitsCapture(
                    leader_logits=_copy_leader_rows(
                        logits_output.next_token_logits, leader_rows
                    ),
                    leader_rows=leader_rows,
                    leader_normalizer=(
                        _copy_leader_rows(normalizer, leader_rows)
                        if normalizer is not None
                        else None
                    ),
                )
            logits_output.beam_normalizer = None
