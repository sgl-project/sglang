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
"""Ordinary SynthID sampling dispatch, separate from the generic sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch

from sglang.srt.watermarking.synthid import (
    SynthIDSparseOutput,
    SynthIDWatermarkProcessor,
)

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class SynthIDSamplePlan(msgspec.Struct, frozen=True, kw_only=True):
    token_ids: Optional[torch.Tensor] = None
    sparse_output: Optional[SynthIDSparseOutput] = None
    committed: bool = False
    sampled: bool = False
    use_sparse_sampling: bool = False


def prepare_synthid_sample(
    *,
    processor: SynthIDWatermarkProcessor,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    ordinary: bool,
    fused: bool,
    simple_sampling: bool,
    return_logprob: bool,
    return_sampling_mask: bool,
    filter_probs: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> SynthIDSamplePlan:
    state = sampling_info.synthid_state
    if state is None:
        raise RuntimeError(
            "SynthID watermarking is enabled but the sampling batch has no state."
        )
    if ordinary and sampling_info.is_all_greedy:
        return SynthIDSamplePlan()

    use_sparse_sampling = (
        not sampling_info.is_all_greedy
        and simple_sampling
        and sampling_info.sampling_seed is None
        and not return_logprob
        and not return_sampling_mask
    )
    if use_sparse_sampling and fused and processor.supports_fused_sampling(logits):
        return SynthIDSamplePlan(
            token_ids=processor.sample_stochastic_and_commit(
                logits, sampling_info.temperatures, state
            ),
            committed=True,
            sampled=True,
            use_sparse_sampling=True,
        )
    if use_sparse_sampling and fused and processor.supports_wide_sampling(logits):
        return SynthIDSamplePlan(
            token_ids=processor.sample_wide_stochastic_and_commit(
                logits, sampling_info.temperatures, state
            ),
            committed=True,
            sampled=True,
            use_sparse_sampling=True,
        )

    sparse_output = processor.apply(
        logits,
        sampling_info.temperatures,
        state,
        filter_probs=filter_probs if ordinary and not simple_sampling else None,
        scatter_to_logits=(
            (not sampling_info.is_all_greedy and not use_sparse_sampling)
            or return_logprob
            or return_sampling_mask
        ),
    )
    return SynthIDSamplePlan(
        sparse_output=sparse_output,
        use_sparse_sampling=use_sparse_sampling,
    )
