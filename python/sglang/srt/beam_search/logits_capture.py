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

    leader_logits: torch.Tensor  # pre-sample clone of the leader rows
    tail_logits: Optional[torch.Tensor] = None  # member rows' slice (decode)
    leader_rows: Optional[List[int]] = None  # leaders' batch indices (extend)


def capture_pre_sample_logits(
    batch: Optional[ScheduleBatch],
    forward_batch: ForwardBatch,
    logits_output: Optional[LogitsProcessorOutput],
) -> None:
    """Decode: split the member tail off logits/hidden_states/positions and
    clone the leader rows. Extend: clone the leaders' rows. No-op otherwise."""
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
        leader_rows = [e.leader_idx for e in batch.beam_tail.entries]
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
