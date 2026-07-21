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
"""Beam search carrier types shared across process boundaries.

BeamSearchSequence is the per-sequence payload of the beam_results carrier
(scheduler -> detokenizer -> tokenizer manager); it must stay import-light
since io_struct pulls it into every IPC participant.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BeamSearchSequence:
    """One beam candidate sequence; text is filled only when the sequence is
    about to be returned to the user."""

    tokens: List[int]  # Generated tokens (excluding prompt)
    cum_logprob: float = 0.0  # Cumulative log probability for sorting

    finish_reason: Optional[object] = None  # Reason for completion (if finished)
    text: Optional[str] = None  # Decoded text (filled on completion)
    beam_score: Optional[float] = None  # Beam search score, for return

    def finished(self):
        return self.finish_reason is not None
