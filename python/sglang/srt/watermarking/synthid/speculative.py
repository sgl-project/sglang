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
"""SynthID draft proposals for linear speculative decoding."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_spec
from sglang.srt.sampling.watermarking.synthid_request import (
    apply_mixed_logits,
    is_enabled,
    is_mixed,
)
from sglang.srt.utils.common import fast_topk
from sglang.srt.watermarking.synthid import (
    SynthIDSparseOutput,
    SynthIDWatermarkProcessor,
)

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def _has_request_filters(sampling_info: SamplingBatchInfo) -> bool:
    return (
        sampling_info.need_top_k_sampling
        or sampling_info.need_top_p_sampling
        or sampling_info.need_min_p_sampling
    )


def _filter_unwatermarked_target_probs(
    probs: torch.Tensor,
    *,
    sampling_info: SamplingBatchInfo,
    rows_per_request: int,
) -> torch.Tensor:
    """Retain EAGLE's sequential top-k/top-p target law, without min-p."""
    top_ks = sampling_info.top_ks.repeat_interleave(rows_per_request)
    top_ps = sampling_info.top_ps.repeat_interleave(rows_per_request)
    if probs.is_cuda:
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        if sampling_info.need_top_k_sampling:
            probs = top_k_renorm_prob(probs, top_ks)
        if sampling_info.need_top_p_sampling:
            probs = top_p_renorm_prob(probs, top_ps)
        return probs

    from sglang.srt.layers.sampler import (
        top_k_top_p_min_p_filter_probs_torch,
        top_p_normalize_probs_torch,
    )

    if sampling_info.need_top_k_sampling:
        values, indices = top_k_top_p_min_p_filter_probs_torch(
            probs, top_ks, torch.ones_like(top_ps), torch.zeros_like(top_ps), False
        )
        probs = torch.zeros_like(probs).scatter_(1, indices, values)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if sampling_info.need_top_p_sampling:
        probs = top_p_normalize_probs_torch(probs, top_ps)
    return probs


def _pin_greedy_logits(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    rows_per_request: int,
) -> None:
    greedy_rows = sampling_info.top_ks <= 1
    enabled = sampling_info.synthid_state.request_enabled
    if enabled is not None:
        greedy_rows = greedy_rows & enabled
    if rows_per_request != 1:
        greedy_rows = greedy_rows.repeat_interleave(rows_per_request)
    greedy_rows = greedy_rows[:, None]
    greedy_tokens = logits.argmax(dim=-1, keepdim=True)
    logits.masked_fill_(greedy_rows, -float("inf"))
    logits.scatter_(
        1,
        greedy_tokens,
        torch.where(greedy_rows, 0.0, logits.gather(1, greedy_tokens)),
    )


def apply_synthid_to_target(
    *,
    processor: SynthIDWatermarkProcessor,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    candidates: torch.Tensor,
    tree_topk: int,
    return_logprob: bool,
) -> tuple[
    Optional[SynthIDSparseOutput],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if sampling_info.synthid_state is None:
        raise RuntimeError(
            "SynthID watermarking is enabled but the speculative batch has no state."
        )
    if tree_topk != 1:
        raise RuntimeError(
            "SynthID speculative watermarking requires a linear top-k=1 tree."
        )
    if sampling_info.is_all_greedy:
        return None, None, None, None

    original_logits = logits.clone() if return_logprob else None
    has_filters = _has_request_filters(sampling_info)
    if has_filters and sampling_info.is_any_greedy:
        _pin_greedy_logits(logits, sampling_info, candidates.shape[1])
    if is_mixed(sampling_info.synthid_state):
        if not has_filters and processor.try_apply_mixed_target(
            logits=logits,
            temperatures=sampling_info.temperatures,
            state=sampling_info.synthid_state,
            verify_tokens=candidates,
        ):
            return None, None, None, original_logits
        filter_probs = None
        if has_filters:
            from sglang.srt.watermarking.synthid.sampling import filter_synthid_probs

            filter_probs = partial(
                filter_synthid_probs,
                sampling_info=sampling_info,
                rows_per_request=candidates.shape[1],
            )
        apply_mixed_logits(
            processor=processor,
            logits=logits,
            temperatures=sampling_info.temperatures,
            state=sampling_info.synthid_state,
            verify_tokens=candidates,
            filter_probs=filter_probs,
            off_filter_probs=(
                partial(
                    _filter_unwatermarked_target_probs,
                    sampling_info=sampling_info,
                    rows_per_request=candidates.shape[1],
                )
                if sampling_info.need_top_k_sampling
                or sampling_info.need_top_p_sampling
                else None
            ),
        )
        return None, None, None, original_logits
    if (
        not has_filters
        and processor._supports_fused_state(logits)
        and not return_logprob
        and get_spec().speculative_use_rejection_sampling
    ):
        sparse_probs, sparse_tokens = processor.apply_sparse_probs(
            logits,
            sampling_info.temperatures,
            sampling_info.synthid_state,
            verify_tokens=candidates,
            sort_tokens=True,
        )
        return None, sparse_probs, sparse_tokens, original_logits

    filter_probs = None
    if has_filters:
        from sglang.srt.watermarking.synthid.sampling import filter_synthid_probs

        filter_probs = partial(
            filter_synthid_probs,
            sampling_info=sampling_info,
            rows_per_request=candidates.shape[1],
        )
    sparse_output = processor.apply(
        logits,
        sampling_info.temperatures,
        sampling_info.synthid_state,
        verify_tokens=candidates,
        scatter_to_logits=True,
        filter_probs=filter_probs,
    )
    return sparse_output, None, None, original_logits


def apply_synthid_to_draft_logits(
    *,
    processor: Optional[SynthIDWatermarkProcessor],
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    prior_draft_tokens: Optional[torch.Tensor] = None,
) -> tuple[
    Optional[SynthIDSparseOutput],
    Optional[tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]],
]:
    if processor is None or not is_enabled(sampling_info.synthid_state):
        return None, None
    if sampling_info.synthid_state is None:
        raise RuntimeError(
            "SynthID watermarking is enabled but the draft batch has no state."
        )
    if sampling_info.is_all_greedy:
        return None, None
    has_filters = _has_request_filters(sampling_info)
    if has_filters and sampling_info.is_any_greedy:
        _pin_greedy_logits(logits, sampling_info, 1)
    if is_mixed(sampling_info.synthid_state):
        if not has_filters:
            sample = processor.try_sample_mixed_draft(
                logits=logits,
                temperatures=sampling_info.temperatures,
                state=sampling_info.synthid_state,
                prior_draft_tokens=prior_draft_tokens,
            )
            if sample is not None:
                probs, sampled_probs, sampled_tokens = sample
                return None, (probs, None, sampled_probs, sampled_tokens)
        filter_probs = None
        if has_filters:
            from sglang.srt.watermarking.synthid.sampling import filter_synthid_probs

            filter_probs = partial(filter_synthid_probs, sampling_info=sampling_info)
        apply_mixed_logits(
            processor=processor,
            logits=logits,
            temperatures=sampling_info.temperatures,
            state=sampling_info.synthid_state,
            prior_tokens=prior_draft_tokens,
            draft=True,
            filter_probs=filter_probs,
        )
        return None, None
    filter_probs = None
    if has_filters:
        from sglang.srt.watermarking.synthid.sampling import filter_synthid_probs

        filter_probs = partial(filter_synthid_probs, sampling_info=sampling_info)
    if sampling_info.synthid_force_dense_draft_probs:
        processor.apply_draft(
            logits,
            sampling_info.temperatures,
            sampling_info.synthid_state,
            prior_draft_tokens=prior_draft_tokens,
            scatter_to_logits=True,
            filter_probs=filter_probs,
        )
        return None, None
    if (
        get_spec().speculative_use_rejection_sampling
        and processor.supports_fused_sampling(logits)
        and not has_filters
    ):
        fused_draft_sample = processor.apply_draft_probs_and_sample(
            logits,
            sampling_info.temperatures,
            sampling_info.synthid_state,
            prior_draft_tokens=prior_draft_tokens,
        )
        return None, fused_draft_sample
    return (
        processor.apply_draft(
            logits,
            sampling_info.temperatures,
            sampling_info.synthid_state,
            prior_draft_tokens=prior_draft_tokens,
            scatter_to_logits=False,
            filter_probs=filter_probs,
        ),
        None,
    )


def ensure_sparse_algo5_draft_probs(
    *,
    processor: Optional[SynthIDWatermarkProcessor],
    draft_probs: Optional[torch.Tensor],
    draft_prob_token_ids: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if draft_probs is None or draft_prob_token_ids is not None or processor is None:
        return draft_probs, draft_prob_token_ids

    top_k = min(processor.config.top_k, draft_probs.shape[-1])
    sparse_probs, sparse_token_ids = fast_topk(draft_probs, top_k, dim=-1)
    sparse_token_ids, order = sparse_token_ids.sort(dim=-1)
    return sparse_probs.gather(-1, order), sparse_token_ids


def sample_synthid_draft_logits(
    *,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    sparse_output: Optional[SynthIDSparseOutput],
    fused_draft_sample: Optional[
        tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]
    ],
    use_rejection_sampling: bool,
    topk: int,
):
    from sglang.srt.speculative.spec_utils import fast_sample

    if sparse_output is None:
        if fused_draft_sample is not None:
            proposal_probs, proposal_tokens, sampled_probs, sampled_tokens = (
                fused_draft_sample
            )
            return (
                proposal_probs,
                proposal_tokens,
                sampled_probs.reshape(-1, 1),
                sampled_tokens.reshape(-1, 1),
            )
        if not use_rejection_sampling:
            raise RuntimeError(
                "SynthID speculative sampling requires rejection sampling."
            )
        proposal_probs = torch.softmax(logits, dim=-1)
        if sampling_info.is_all_greedy:
            topk_index = proposal_probs.argmax(dim=-1, keepdim=True)
            topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            proposal_probs = torch.zeros_like(proposal_probs).scatter_(
                1, topk_index, 1.0
            )
        else:
            topk_p, topk_index = fast_sample(proposal_probs, num_samples=1)
        return proposal_probs, None, topk_p, topk_index

    proposal_probs = torch.softmax(sparse_output.scores, dim=-1)
    if not use_rejection_sampling:
        topk_p, sampled_indices = fast_topk(proposal_probs, topk, dim=-1)
        return (
            None,
            None,
            topk_p,
            sparse_output.token_ids.gather(1, sampled_indices),
        )

    sorted_tokens, order = sparse_output.token_ids.sort(dim=-1)
    sorted_probs = proposal_probs.gather(1, order)
    if sampling_info.is_all_greedy:
        sampled_indices = sorted_probs.argmax(dim=-1, keepdim=True)
        sampled_probs = sorted_probs.gather(1, sampled_indices)
    else:
        sampled_probs, sampled_indices = fast_sample(sorted_probs, num_samples=1)

    return (
        sorted_probs,
        sorted_tokens,
        sampled_probs,
        sorted_tokens.gather(1, sampled_indices),
    )
