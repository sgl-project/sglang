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

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.beam_search import shared_context_attention
from sglang.srt.beam_search.recurrent import (
    GdnBeamReplayInputs,
    build_gdn_beam_replay_inputs,
    build_read_only_state_mask,
)
from sglang.srt.beam_search.trie_output_head import build_trie_row_levels

if TYPE_CHECKING:
    from sglang.srt.beam_search.shared_context_attention import BeamAttentionMetadata
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class BeamForwardInputs(msgspec.Struct):
    # Per-row SID codebook depths for trie-conditioned decode batches.
    trie_levels: Optional[torch.Tensor] = None
    # Shared-context attention metadata for all-beam decode batches.
    attention_metadata: Optional[BeamAttentionMetadata] = None
    read_only_state_rows: Optional[torch.Tensor] = None
    gdn_replay: Optional[GdnBeamReplayInputs] = None


def build_beam_forward_inputs(
    batch: ScheduleBatch, model_runner: ModelRunner
) -> Optional[BeamForwardInputs]:
    trie_levels = build_trie_row_levels(batch, model_runner)
    attention_metadata = shared_context_attention.build_metadata(batch, model_runner)
    tail = batch.beam_tail
    read_only_rows = replay = None
    if tail is not None and batch.forward_mode.is_decode():
        read_only_rows = build_read_only_state_mask(
            tail=tail, num_rows=len(batch.req_pool_indices), device=batch.device
        )
        replay = build_gdn_beam_replay_inputs(
            tail=tail, num_rows=len(batch.req_pool_indices), device=batch.device
        )
    if (
        trie_levels is None
        and attention_metadata is None
        and read_only_rows is None
        and replay is None
    ):
        return None
    return BeamForwardInputs(trie_levels, attention_metadata, read_only_rows, replay)
