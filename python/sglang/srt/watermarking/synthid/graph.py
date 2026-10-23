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
"""CUDA graph eligibility for SynthID draft sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.watermarking.synthid import SynthIDWatermarkProcessor

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def synthid_graph_config(
    *,
    processor: Optional[SynthIDWatermarkProcessor],
    num_steps: int,
    rejection_sampling: bool,
) -> tuple[bool, bool]:
    supported = (
        processor is not None
        and processor.config.context_history_size <= 2048
        and processor.config.context_width <= 16
        and num_steps <= 8
    )
    return supported, supported and rejection_sampling


def can_run_synthid_graph(
    *,
    forward_batch: ForwardBatch,
    graph_supported: bool,
    sparse_draft_probs: bool,
    rejection_sampling: bool,
) -> bool:
    sampling_info = forward_batch.sampling_info
    if (
        sampling_info is None
        or sampling_info.synthid_state is None
        or not graph_supported
        or sampling_info.sampling_seed is not None
        or sampling_info.need_top_k_sampling
        or sampling_info.need_top_p_sampling
        or sampling_info.need_min_p_sampling
    ):
        return False
    if not forward_batch.forward_mode.is_idle():
        if rejection_sampling and forward_batch.spec_info.draft_probs is None:
            return False
        has_sparse_ids = forward_batch.spec_info.draft_prob_token_ids is not None
        if has_sparse_ids != sparse_draft_probs:
            return False
    return True
