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
"""Beam-only inputs a ForwardBatch carries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.beam_search import shared_context_attention
from sglang.srt.beam_search.trie_output_head import build_trie_row_levels

if TYPE_CHECKING:
    from sglang.srt.beam_search.shared_context_attention import BeamAttentionMetadata
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class BeamForwardInputs:
    # Per-row SID codebook depths for trie-conditioned decode batches.
    trie_levels: Optional[torch.Tensor] = None
    # Shared-context attention metadata for all-beam decode batches.
    attention_metadata: Optional[BeamAttentionMetadata] = None


def build_beam_forward_inputs(
    batch: ScheduleBatch, model_runner: ModelRunner
) -> Optional[BeamForwardInputs]:
    trie_levels = build_trie_row_levels(batch, model_runner)
    attention_metadata = shared_context_attention.build_metadata(batch, model_runner)
    if trie_levels is None and attention_metadata is None:
        return None
    return BeamForwardInputs(trie_levels, attention_metadata)
