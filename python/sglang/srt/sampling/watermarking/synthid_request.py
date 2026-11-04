"""Request policy and mixed-batch routing around the unchanged SynthID transform."""

from __future__ import annotations

import torch

from sglang.srt.watermarking.synthid.state import SynthIDState


def request_enabled(request, features) -> bool:
    control = request.sampling_params.watermark
    if control is None:
        return features.watermark_default_enabled or features.watermark_enforce_all
    if control.enabled is False and features.watermark_enforce_all:
        raise ValueError("SynthID enforce_all does not allow explicit opt-out")
    return control.enabled is True


def is_mixed(state) -> bool:
    return (
        isinstance(state, SynthIDState)
        and state.request_enabled_host is not None
        and any(state.request_enabled_host)
        and not all(state.request_enabled_host)
    )


def is_enabled(state) -> bool:
    if state is None:
        return False
    return (
        not isinstance(state, SynthIDState)
        or state.request_enabled_host is None
        or any(state.request_enabled_host)
    )


def apply_mixed_logits(
    *,
    processor,
    logits,
    temperatures,
    state,
    verify_tokens=None,
    prior_tokens=None,
    draft=False,
    filter_probs=None,
    off_filter_probs=None,
):
    """Retain the full base vocabulary for OFF rows; only ON rows get Algorithm 5."""
    raw = logits.clone()
    if verify_tokens is not None:
        processor.apply(
            logits,
            temperatures,
            state,
            verify_tokens=verify_tokens,
            filter_probs=filter_probs,
        )
        rows_per_request = verify_tokens.shape[1]
    elif draft:
        processor.apply_draft(
            logits,
            temperatures,
            state,
            prior_draft_tokens=prior_tokens,
            filter_probs=filter_probs,
        )
        rows_per_request = 1
    else:
        processor.apply(logits, temperatures, state, filter_probs=filter_probs)
        rows_per_request = 1
    enabled = state.request_enabled.repeat_interleave(rows_per_request)[:, None]
    scaled = raw / temperatures.repeat_interleave(rows_per_request, dim=0)
    if off_filter_probs is not None:
        scaled = torch.log(off_filter_probs(torch.softmax(scaled, dim=-1)))
    logits.copy_(torch.where(enabled, logits, scaled))


def forward_mixed(
    *,
    sampler,
    logits_output,
    sampling_info,
    return_logprob,
    top_logprobs_nums,
    token_ids_logprobs,
    positions,
):
    from sglang.srt.environ import envs
    from sglang.srt.layers import sampler as sampler_module
    from sglang.srt.watermarking.synthid.sampling import (
        attach_sampling_mask,
        filter_synthid_probs,
    )

    logits = sampler._preprocess_logits(logits_output.next_token_logits, sampling_info)
    raw = logits.clone()
    temperatures = sampling_info.temperatures
    base_probs = torch.softmax(raw / temperatures, dim=-1)
    has_filters = (
        sampling_info.need_top_k_sampling
        or sampling_info.need_top_p_sampling
        or sampling_info.need_min_p_sampling
    )
    filtered_base = (
        filter_synthid_probs(base_probs, sampling_info) if has_filters else base_probs
    )
    apply_mixed_logits(
        processor=sampler.synthid_watermark,
        logits=logits,
        temperatures=temperatures,
        state=sampling_info.synthid_state,
        filter_probs=(
            (lambda probs: filter_synthid_probs(probs, sampling_info))
            if has_filters
            else None
        ),
    )
    enabled = sampling_info.synthid_state.request_enabled[:, None]
    probs = torch.where(enabled, torch.softmax(logits, dim=-1), filtered_base)
    greedy = (sampling_info.top_ks <= 1)[:, None]
    greedy_probs = torch.zeros_like(probs).scatter_(
        1, raw.argmax(dim=-1, keepdim=True), 1
    )
    probs = torch.where(greedy, greedy_probs, probs)
    tokens = sampler_module.sampling_from_probs_torch(
        probs,
        sampling_seed=sampling_info.sampling_seed,
        positions=positions,
        synthid_applied=True,
    ).to(torch.int32)
    if sampling_info.sampling_seed is not None:
        # OFF seeds must retain the generic sampler's filtered candidate order.
        base_tokens, _ = sampler._sample_from_probs(
            base_probs, sampling_info, positions, not has_filters
        )
        tokens = torch.where(enabled.view(-1), tokens, base_tokens)
    sampler._sync_token_ids_across_tp(tokens, sampling_info, force_sync=True)
    if return_logprob:
        if sampler_module.SGLANG_RETURN_ORIGINAL_LOGPROB:
            logprobs = torch.log_softmax(raw, dim=-1)
        elif envs.SGLANG_RETURN_WATERMARKED_LOGPROB.get():
            base_logprobs = (
                torch.log_softmax(raw / temperatures, dim=-1)
                if sampler.enable_deterministic
                else torch.log(base_probs)
            )
            logprobs = torch.where(enabled, torch.log(probs), base_logprobs)
        else:
            reference = torch.where(greedy & enabled, raw, raw / temperatures)
            logprobs = (
                torch.log_softmax(reference, dim=-1)
                if sampler.enable_deterministic
                else torch.log(torch.softmax(reference, dim=-1))
            )
        sampler.output_logprob_processor.compute_logprobs(
            logprobs,
            top_logprobs_nums,
            token_ids_logprobs,
            tokens,
        ).write_output_to(logits_output)
    attach_sampling_mask(
        sampler=sampler,
        logits_output=logits_output,
        sampling_info=sampling_info,
        tokens=tokens,
        probs=probs,
    )
    sampler.synthid_watermark.commit(sampling_info.synthid_state, tokens)
    return tokens
