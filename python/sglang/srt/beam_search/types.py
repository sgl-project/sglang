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

from typing import List, Optional

import msgspec


class BeamSearchSequence(msgspec.Struct, omit_defaults=True):
    """One beam candidate sequence; text is filled only when the sequence is
    about to be returned to the user."""

    tokens: List[int]  # generated only, no prompt
    cum_logprob: float = 0.0

    finish_reason: Optional[object] = None
    text: Optional[str] = None
    beam_score: Optional[float] = None  # length-normalized; the sort key
