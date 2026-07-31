"""Request-local sampling for DVR proposal and verification."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils.async_probe import sanitize_nan_logits

DVR_PROPOSAL_RNG_DOMAIN = 0xA511E9B3
DVR_ACCEPT_RNG_DOMAIN = 0x63D83595
DVR_FINAL_RNG_DOMAIN = 0xB8F1A2C7
DVR_SAMPLING_BLOCK_SIZE = 4096


def dvr_proposal_buffer_bytes(*, spec_algorithm, server_args, vocab_size: int) -> int:
    """Return proposal storage allocated after KV-pool memory profiling."""

    if spec_algorithm.is_dvr_self_draft():
        request_rows = max(
            server_args.cuda_graph_config.decode.max_bs or 0,
            server_args.max_running_requests or 0,
            1,
        )
        resident_probability_rows = request_rows * server_args.speculative_num_steps
    elif (
        spec_algorithm.is_dvr_eagle()
        and server_args.speculative_use_rejection_sampling
        and not server_args.disable_overlap_schedule
    ):
        resident_probability_rows = (
            server_args.max_running_requests // server_args.dp_size + 1
        )
    else:
        return 0
    return resident_probability_rows * vocab_size * torch.float32.itemsize


def dvr_sampling_probs(
    probs: torch.Tensor,
    sampling_info,
    repeat: int = 1,
) -> torch.Tensor:
    """Build sampler-equivalent proposal or target probabilities for DVR."""
    top_ks = sampling_info.top_ks
    top_ps = sampling_info.top_ps
    min_ps = sampling_info.min_ps
    if repeat != 1:
        top_ks = torch.repeat_interleave(top_ks, repeat, dim=0)
        top_ps = torch.repeat_interleave(top_ps, repeat, dim=0)
        min_ps = torch.repeat_interleave(min_ps, repeat, dim=0)

    backend = get_server_args().sampling_backend
    if backend == "flashinfer" and probs.is_cuda:
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        if sampling_info.need_min_p_sampling:
            probs = top_k_renorm_prob(probs, top_ks)
            probs = top_p_renorm_prob(probs, top_ps)
            threshold = probs.amax(dim=-1, keepdim=True) * min_ps.unsqueeze(1)
            probs = torch.where(probs >= threshold, probs, 0.0)
            return probs / probs.sum(dim=-1, keepdim=True)
        if sampling_info.need_top_p_sampling:
            probs = top_p_renorm_prob(probs, top_ps)
        if sampling_info.need_top_k_sampling:
            probs = top_k_renorm_prob(probs, top_ks)
        return probs

    if backend == "pytorch" or not probs.is_cuda:
        if not (
            sampling_info.need_top_k_sampling
            or sampling_info.need_top_p_sampling
            or sampling_info.need_min_p_sampling
        ):
            return probs
        filtered, indices = probs.sort(dim=-1, descending=True)
        cumulative = torch.cumsum(filtered, dim=-1)
        filtered[
            torch.arange(probs.shape[-1], device=probs.device).view(1, -1)
            >= top_ks.view(-1, 1)
        ] = 0.0
        filtered[(cumulative - filtered) > top_ps.view(-1, 1)] = 0.0
        if sampling_info.need_min_p_sampling:
            threshold = filtered[:, :1] * min_ps.unsqueeze(1)
            filtered[filtered < threshold] = 0.0
        filtered.div_(filtered.sum(dim=-1, keepdim=True))
        return torch.zeros_like(probs).scatter_(-1, indices, filtered)

    raise ValueError(f"Unsupported DVR sampling backend: {backend}")


def dvr_draft_sample(logits: torch.Tensor, sampling_info, positions: torch.Tensor):
    """Sample a provisional token and return its rejection-sampling proposal."""
    sampling_info.apply_logits_bias(logits)
    sanitize_nan_logits(logits, "dvr draft logits")
    if sampling_info.is_all_greedy:
        return torch.argmax(logits, dim=-1), None

    probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
    proposal = dvr_sampling_probs(probs, sampling_info)
    token_ids = dvr_sample_from_probs(proposal, sampling_info.sampling_seed, positions)
    return token_ids, proposal


@triton.jit
def dvr_stateless_uniform(seed, position, domain: tl.constexpr):
    """Map a request seed and absolute token position to a uniform variate."""

    seed = seed.to(tl.uint64)
    value: tl.uint32 = 0
    value = murmur3_mix(value, (seed & 0xFFFFFFFF).to(tl.uint32))
    value = murmur3_mix(value, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    value = murmur3_mix(value, position.to(tl.uint32))
    value = murmur3_mix(value, domain)
    value ^= 16
    value = fmix32(value)
    return ((value >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)


@triton.jit
def dvr_sample_from_probs_kernel(
    Probs,
    Seeds,
    Positions,
    Output,
    stride_probs_b,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    RNG_DOMAIN: tl.constexpr,
    POSITION_OFFSET: tl.constexpr,
):
    request_id = tl.program_id(0)
    probs = Probs + request_id * stride_probs_b

    norm = 0.0
    for start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = start + tl.arange(0, BLOCK_V)
        norm += tl.sum(tl.load(probs + offsets, mask=offsets < VOCAB_SIZE, other=0.0))

    seed = tl.load(Seeds + request_id)
    position = tl.load(Positions + request_id) + POSITION_OFFSET
    target = dvr_stateless_uniform(seed, position, RNG_DOMAIN) * norm
    cumulative = 0.0
    sampled_token = 0
    last_positive_token = 0
    found = 0

    for start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = start + tl.arange(0, BLOCK_V)
        mask = offsets < VOCAB_SIZE
        values = tl.load(probs + offsets, mask=mask, other=0.0)
        positive = mask & (values > 0.0)
        if tl.max(positive, axis=0):
            last_positive_token = start + tl.argmax(
                tl.where(positive, offsets, 0), axis=0
            )
        if found == 0:
            block_cumulative = tl.cumsum(values, axis=0)
            matches = mask & (cumulative + block_cumulative > target)
            if tl.max(matches, axis=0):
                sampled_token = start + tl.argmax(matches.to(tl.int32), axis=0)
                found = 1
            cumulative += tl.sum(values)

    # Float32 reduction can leave the random target on the final CDF boundary.
    # The last positive support token is the valid endpoint in that case.
    tl.store(
        Output + request_id,
        tl.where(found != 0, sampled_token, last_positive_token),
    )


def dvr_sample_from_probs(probs, seeds, positions, *, position_offset=0):
    """Sample rows reproducibly without global RNG state or FP64 Gumbel noise."""

    if not probs.is_cuda:
        raise RuntimeError("DVR seeded sampling requires CUDA tensors.")
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got {tuple(probs.shape)}.")
    if seeds is None:
        raise ValueError("DVR seeded sampling requires one seed per request.")

    seeds = seeds.reshape(-1)
    positions = positions.reshape(-1)
    if seeds.shape != positions.shape or seeds.shape[0] != probs.shape[0]:
        raise ValueError(
            "DVR seed and position rows must match probabilities: "
            f"probs={tuple(probs.shape)}, seeds={tuple(seeds.shape)}, "
            f"positions={tuple(positions.shape)}."
        )

    output = torch.empty(probs.shape[0], dtype=torch.int64, device=probs.device)
    dvr_sample_from_probs_kernel[(probs.shape[0],)](
        probs,
        seeds,
        positions,
        output,
        probs.stride(0),
        VOCAB_SIZE=probs.shape[1],
        BLOCK_V=DVR_SAMPLING_BLOCK_SIZE,
        RNG_DOMAIN=DVR_PROPOSAL_RNG_DOMAIN,
        POSITION_OFFSET=position_offset,
    )
    return output


@triton.jit
def dvr_chain_rejection_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetrieveIndex,
    SamplingSeeds,
    Positions,
    RootOnlyMask,
    TargetProbs,
    DraftProbs,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_pos_b,
    stride_pos_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ACCEPT_DOMAIN: tl.constexpr,
    FINAL_DOMAIN: tl.constexpr,
    HAS_ROOT_ONLY_MASK: tl.constexpr,
    POINT_PROPOSAL: tl.constexpr,
):
    request_id = tl.program_id(0)
    candidate_base = Candidates + request_id * stride_cand_b
    index_base = RetrieveIndex + request_id * stride_idx_b
    seed = tl.load(SamplingSeeds + request_id)

    current_row = 0
    root_index = tl.load(index_base)
    tl.store(AcceptIndex + request_id * stride_idx_b, root_index)
    last_accepted_index = root_index
    accepted = 0
    step = 1
    root_only = 0
    rejected_token = tl.load(candidate_base)
    if HAS_ROOT_ONLY_MASK:
        root_only = tl.load(RootOnlyMask + request_id)
    continue_verifying = 1 - root_only

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(candidate_base + step * stride_cand_s)
        rejected_token = draft_token
        target_offset = (
            request_id * stride_tp_b
            + current_row * stride_tp_s
            + draft_token * stride_tp_v
        )
        target_prob = tl.load(TargetProbs + target_offset)
        if POINT_PROPOSAL:
            draft_prob = 1.0
        else:
            draft_offset = (
                request_id * stride_dp_b
                + current_row * stride_dp_s
                + draft_token * stride_dp_v
            )
            draft_prob = tl.load(DraftProbs + draft_offset)
        position = tl.load(
            Positions + request_id * stride_pos_b + current_row * stride_pos_s
        )
        coin = dvr_stateless_uniform(seed, position, ACCEPT_DOMAIN)

        if coin * draft_prob < target_prob:
            accepted += 1
            current_row = step
            tl.store(Predicts + last_accepted_index, draft_token)
            current_index = tl.load(index_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + request_id * stride_idx_b + accepted * stride_idx_s,
                current_index,
            )
            last_accepted_index = current_index
            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + request_id, accepted)

    all_drafts_accepted = continue_verifying
    position = tl.load(
        Positions + request_id * stride_pos_b + current_row * stride_pos_s
    )
    final_coin = dvr_stateless_uniform(seed, position, FINAL_DOMAIN)
    target_base = TargetProbs + request_id * stride_tp_b + current_row * stride_tp_s
    # The target has one more row than the proposal. Clamp the all-accepted
    # lookup to a valid row; its value is ignored while sampling target p.
    if not POINT_PROPOSAL:
        draft_row = tl.minimum(current_row, NUM_SLOTS - 2)
        draft_base = DraftProbs + request_id * stride_dp_b + draft_row * stride_dp_s

    target_norm = 0.0
    residual_norm = 0.0
    for start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = start + tl.arange(0, BLOCK_V)
        mask = offsets < VOCAB_SIZE
        target_values = tl.load(
            target_base + offsets * stride_tp_v, mask=mask, other=0.0
        )
        if POINT_PROPOSAL:
            draft_values = tl.where(offsets == rejected_token, 1.0, 0.0)
        else:
            draft_values = tl.load(
                draft_base + offsets * stride_dp_v, mask=mask, other=0.0
            )
        target_norm += tl.sum(target_values)
        residual_norm += tl.sum(tl.maximum(target_values - draft_values, 0.0))

    residual_is_valid = (residual_norm > 0.0) & (residual_norm < float("inf"))
    sample_target = (root_only == 1) | (all_drafts_accepted == 1) | ~residual_is_valid
    sample_norm = tl.where(sample_target, target_norm, residual_norm)
    target = final_coin * sample_norm
    cumulative = 0.0
    final_token = 0
    last_positive_token = 0
    found = 0

    for start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = start + tl.arange(0, BLOCK_V)
        mask = offsets < VOCAB_SIZE
        target_values = tl.load(
            target_base + offsets * stride_tp_v, mask=mask, other=0.0
        )
        if POINT_PROPOSAL:
            draft_values = tl.where(offsets == rejected_token, 1.0, 0.0)
        else:
            draft_values = tl.load(
                draft_base + offsets * stride_dp_v, mask=mask, other=0.0
            )
        residual = tl.maximum(target_values - draft_values, 0.0)
        values = tl.where(sample_target, target_values, residual)
        positive = mask & (values > 0.0)
        if tl.max(positive, axis=0):
            last_positive_token = start + tl.argmax(
                tl.where(positive, offsets, 0), axis=0
            )
        if found == 0:
            block_cumulative = tl.cumsum(values, axis=0)
            matches = mask & (cumulative + block_cumulative > target)
            if tl.max(matches, axis=0):
                final_token = start + tl.argmax(matches.to(tl.int32), axis=0)
                found = 1
            cumulative += tl.sum(values)

    tl.store(
        Predicts + last_accepted_index,
        tl.where(found != 0, final_token, last_positive_token),
    )


def dvr_chain_rejection_sample(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    target_probs,
    draft_probs,
    sampling_seed,
    positions,
    root_only_mask=None,
):
    """Verify a linear chain with request-local stateless RNG.

    ``draft_probs=None`` represents a deterministic point-mass proposal.
    """

    batch_size, num_slots = candidates.shape
    sampling_seed = sampling_seed.reshape(-1)
    positions = positions.reshape(batch_size, num_slots)
    if sampling_seed.shape[0] != batch_size:
        raise ValueError(
            "DVR rejection sampling requires one seed per request: "
            f"batch_size={batch_size}, seeds={tuple(sampling_seed.shape)}."
        )
    has_root_only_mask = root_only_mask is not None
    if has_root_only_mask:
        root_only_mask = root_only_mask.reshape(-1).to(
            device=candidates.device, dtype=torch.bool
        )
        if root_only_mask.shape[0] != batch_size:
            raise ValueError(
                "DVR root-only mask must contain one value per request: "
                f"batch_size={batch_size}, mask={tuple(root_only_mask.shape)}."
            )
    else:
        # The pointer is not read by the unmasked Triton specialization.
        root_only_mask = accept_token_num

    point_proposal = draft_probs is None
    draft_probs = target_probs if point_proposal else draft_probs
    dvr_chain_rejection_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        sampling_seed,
        positions,
        root_only_mask,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrieve_index.stride(0),
        retrieve_index.stride(1),
        positions.stride(0),
        positions.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        0 if point_proposal else draft_probs.stride(0),
        0 if point_proposal else draft_probs.stride(1),
        0 if point_proposal else draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=target_probs.shape[-1],
        BLOCK_V=DVR_SAMPLING_BLOCK_SIZE,
        ACCEPT_DOMAIN=DVR_ACCEPT_RNG_DOMAIN,
        FINAL_DOMAIN=DVR_FINAL_RNG_DOMAIN,
        HAS_ROOT_ONLY_MASK=has_root_only_mask,
        POINT_PROPOSAL=point_proposal,
    )
