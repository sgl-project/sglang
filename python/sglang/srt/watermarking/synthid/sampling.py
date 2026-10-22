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
"""SynthID sampling integration, without changes to the generic sampling path."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import sampler as sampler_module
from sglang.srt.watermarking.synthid.ordinary import prepare_synthid_sample

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def forward_synthid(
    sampler: Sampler,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    return_logprob: bool,
    top_logprobs_nums: List[int],
    token_ids_logprobs: List[List[int]],
    positions: torch.Tensor,
):
    """Sample a watermarked batch and report its selected logprobs."""
    processor = sampler.synthid_watermark
    if processor is None:
        raise RuntimeError("SynthID sampling requires an initialized processor.")
    logits = logits_output.next_token_logits

    # Preprocess logits (custom processors and NaN handling)
    logits = sampler._preprocess_logits(logits, sampling_info)
    if return_logprob and sampler_module.SGLANG_RETURN_ORIGINAL_LOGPROB:
        original_logprobs = torch.log_softmax(logits, dim=-1)
    return_sampling_mask = any(sampling_info.return_sampling_masks or [])
    sampling_mask_capture = None
    simple_sampling_case = (
        not sampling_info.need_top_p_sampling
        and not sampling_info.need_top_k_sampling
        and not sampling_info.need_min_p_sampling
    )
    return_watermarked_logprob = envs.SGLANG_RETURN_WATERMARKED_LOGPROB.get()
    ordinary_synthid = sampler.use_ordinary_synthid_sampling
    return_sampler_logprobs = (
        return_logprob
        and not sampler_module.SGLANG_RETURN_ORIGINAL_LOGPROB
        and (not ordinary_synthid or return_watermarked_logprob)
    )
    if (
        ordinary_synthid
        and return_logprob
        and not sampler_module.SGLANG_RETURN_ORIGINAL_LOGPROB
        and not return_watermarked_logprob
    ):
        reference_logits = (
            logits
            if sampling_info.is_all_greedy
            else logits.clone().div_(sampling_info.temperatures)
        )
        logprobs = (
            torch.log_softmax(reference_logits, dim=-1)
            if sampling_info.is_all_greedy or sampler.enable_deterministic
            else torch.log(torch.softmax(reference_logits, dim=-1))
        )
        del reference_logits
    greedy_token_ids = (
        logits.argmax(dim=-1, keepdim=True)
        if (
            ordinary_synthid
            and sampling_info.is_any_greedy
            and not sampling_info.is_all_greedy
        )
        else None
    )
    synthid_plan = prepare_synthid_sample(
        processor=processor,
        logits=logits,
        sampling_info=sampling_info,
        ordinary=ordinary_synthid,
        fused=sampler.use_fused_synthid_sampling,
        simple_sampling=simple_sampling_case,
        return_logprob=return_logprob,
        return_sampling_mask=return_sampling_mask,
        filter_probs=(
            partial(filter_synthid_probs, sampling_info=sampling_info)
            if ordinary_synthid and not simple_sampling_case
            else None
        ),
    )
    synthid_sparse_output = synthid_plan.sparse_output
    synthid_committed = synthid_plan.committed
    synthid_sampled = synthid_plan.sampled
    use_sparse_synthid_sampling = synthid_plan.use_sparse_sampling
    if synthid_plan.token_ids is not None:
        batch_next_token_ids = synthid_plan.token_ids
    if greedy_token_ids is not None:
        # Preserve the full-vocabulary argmax, including ties at the cap.
        greedy_rows = (sampling_info.top_ks <= 1).view(-1, 1)
        logits.masked_fill_(greedy_rows, -float("inf"))
        logits.scatter_(
            1,
            greedy_token_ids,
            torch.where(greedy_rows, 0.0, logits.gather(1, greedy_token_ids)),
        )
    if sampling_info.is_all_greedy:
        if synthid_sparse_output is not None:
            batch_next_token_ids = synthid_sparse_output.argmax_token_ids()
        elif (
            sampler_module._use_aiter
            and not sampler_module._disable_aiter_greedy_sample
        ):
            batch_next_token_ids = torch.empty(
                logits.shape[0], device=logits.device, dtype=torch.int32
            )
            sampler_module._aiter_greedy_sample(batch_next_token_ids, logits)
        else:
            batch_next_token_ids = torch.argmax(logits, -1)
        if return_sampler_logprobs:
            if ordinary_synthid:
                logprobs = torch.full_like(logits, -float("inf"))
                logprobs.scatter_(1, batch_next_token_ids[:, None].long(), 0.0)
                enabled = sampling_info.synthid_state.request_enabled
                if enabled is not None:
                    logprobs = torch.where(
                        enabled[:, None], logprobs, logits.log_softmax(dim=-1)
                    )
            else:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    else:
        if not synthid_sampled:
            if use_sparse_synthid_sampling:
                sparse_probs = torch.softmax(synthid_sparse_output.scores, dim=-1)
                sampled_candidate_indices = sampler_module.sampling_from_probs_torch(
                    sparse_probs,
                    sampling_seed=sampling_info.sampling_seed,
                    positions=positions,
                    synthid_applied=True,
                ).to(torch.int64)
                batch_next_token_ids = (
                    synthid_sparse_output.token_ids.gather(
                        1, sampled_candidate_indices[:, None]
                    )
                    .view(-1)
                    .to(torch.int32)
                )
            else:
                # Deterministic inference must use F.log_softmax, as prefill
                # rescoring does; log(softmax(x)) differs at ~1e-6.
                logprobs_via_logsoftmax_kernel = (
                    torch.nn.functional.log_softmax(logits, dim=-1)
                    if return_sampler_logprobs and sampler.enable_deterministic
                    else None
                )
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits
                batch_next_token_ids, sampling_mask_capture = sampler._sample_from_probs(
                    probs,
                    sampling_info,
                    positions,
                    simple_sampling_case or ordinary_synthid,
                    synthid_applied=True,
                )
                if return_sampler_logprobs:
                    logprobs = (
                        logprobs_via_logsoftmax_kernel
                        if logprobs_via_logsoftmax_kernel is not None
                        else torch.log(probs)
                    )
                del probs

    sampler._sync_token_ids_across_tp(
        batch_next_token_ids,
        sampling_info,
        force_sync=not synthid_committed,
    )
    if return_sampling_mask:
        if sampling_info.is_all_greedy:
            logits_output.sampling_mask_output = sampler._build_greedy_sampling_mask_output(
                sampling_info.sampling_mask_batch_indices,
                batch_next_token_ids,
                sampling_info.sampling_support_logprobs_capture_indices,
            )
        else:
            assert sampling_mask_capture is not None
            logits_output.sampling_mask_output = sampler._build_sampling_mask_output(
                batch_next_token_ids,
                sampling_mask_capture._replace(selected_weight=None),
                sampling_info.sampling_support_logprobs_capture_indices,
            )

    if return_logprob:
        if sampler_module.SGLANG_RETURN_ORIGINAL_LOGPROB:
            logprobs = original_logprobs
        logprob_result = sampler.output_logprob_processor.compute_logprobs(
            logprobs,
            top_logprobs_nums,
            token_ids_logprobs,
            batch_next_token_ids,
        )
        logprob_result.write_output_to(logits_output)

    if not synthid_committed:
        processor.commit(sampling_info.synthid_state, batch_next_token_ids)

    return batch_next_token_ids


def attach_sampling_mask(
    *,
    sampler: Sampler,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    tokens: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Capture the realized ON/OFF law using the standard device-result format."""
    rows = sampling_info.sampling_mask_batch_indices
    if rows is None:
        return
    capture = sampler_module._SamplingMaskCapture(
        weights=probs.index_select(0, rows),
        token_ids=None,
        selected_weight=None,
        batch_rows=rows,
    )
    logits_output.sampling_mask_output = sampler._build_sampling_mask_output(
        tokens,
        capture,
        sampling_info.sampling_support_logprobs_capture_indices,
    )


def filter_synthid_probs(
    probs: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """Build the request-filtered base law inside the SynthID candidate cap."""
    top_ks = sampling_info.top_ks
    top_ps = sampling_info.top_ps
    min_ps = sampling_info.min_ps
    if rows_per_request != 1:
        top_ks = top_ks.repeat_interleave(rows_per_request)
        top_ps = top_ps.repeat_interleave(rows_per_request)
        min_ps = min_ps.repeat_interleave(rows_per_request)
    top_ks = top_ks.clamp(max=probs.shape[-1])
    backend = sampler_module.get_exec().kernel.sampling_backend
    if backend == "flashinfer":
        top_k_probs = sampler_module.top_k_renorm_prob(probs, top_ks)
        if sampling_info.need_min_p_sampling:
            filtered = sampler_module.top_p_renorm_prob(top_k_probs, top_ps)
            threshold = filtered.max(dim=-1, keepdim=True).values
            threshold = threshold * min_ps.view(-1, 1)
            filtered = filtered.masked_fill(filtered < threshold, 0.0)
        else:
            # Match the ordinary sampler's joint top-k/top-p intersection.
            top_p_probs = sampler_module.top_p_renorm_prob(probs, top_ps)
            filtered = top_k_probs.masked_fill(top_p_probs == 0, 0.0)
    elif backend == "pytorch":
        sorted_probs, indices = sampler_module.top_k_top_p_min_p_filter_probs_torch(
            probs,
            top_ks,
            top_ps,
            min_ps,
            sampling_info.need_min_p_sampling,
        )
        filtered = torch.zeros_like(probs).scatter_(1, indices, sorted_probs)
    else:
        raise ValueError(f"Invalid SynthID sampling backend: {backend}")
    return filtered / filtered.sum(dim=-1, keepdim=True)
