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
"""Beam search data structures for managing beam search state.

This module defines the core data structures used in beam search:
- BeamSearchSequence: Represents a single beam candidate sequence
- BeamSearchList: Manages the collection of beam candidates for a request
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchSequence:
    """A single beam candidate sequence in beam search.

    This class tracks tokens and log probabilities for one beam candidate.
    It must remain object-based (not fully vectorized) because tokens need to be
    stored dynamically during beam execution.

    The text field is optional and only filled when the sequence is about to be
    returned to the user.
    """

    tokens: List[int]  # Generated tokens (excluding prompt)
    cum_logprob: float = 0.0  # Cumulative log probability for sorting

    finish_reason: Optional[object] = None  # Reason for completion (if finished)
    text: Optional[str] = None  # Decoded text (filled on completion)
    beam_score: Optional[float] = None  # Beam search score, for return

    def finished(self):
        return self.finish_reason is not None


@dataclass
class BeamSearchList:
    """Manages the collection of beam candidates for a beam search request.

    This class maintains both object-based (BeamSearchSequence) and tensor-based
    representations for efficient computation. Tensor fields enable parallel
    operations while BeamSearchSequence objects store complex state.

    Attributes:
        batch_slot_start_idx: Starting index in batch.req_pool_indices array where
            this beam group's incomplete beams are stored (consecutive beam_width slots)
        completed: List of finished beam sequences
        incomplete: List of active beam sequences still being explored

        # Tensor-based state for parallel operations on incomplete beams:
        cum_logprobs: Cumulative log probabilities, Shape: [num_incomplete_beams]
        last_tokens: Last token of each beam (updated when incomplete refreshes),
            Shape: [num_incomplete_beams]
        prompt_lens: Prompt lengths for KV cache (set only at beamlist construction),
            Shape: [num_incomplete_beams]
    """

    batch_slot_start_idx: int = -1
    completed: List[BeamSearchSequence] = field(default_factory=list)
    incomplete: List[BeamSearchSequence] = field(default_factory=list)

    # Tensor-based state for parallel operations (only for incomplete beams)
    cum_logprobs: Optional[torch.Tensor] = None
    last_tokens: Optional[torch.Tensor] = None
    prompt_lens: Optional[torch.Tensor] = None

    def empty(self):
        return len(self.completed) + len(self.incomplete) == 0
