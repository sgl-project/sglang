from __future__ import annotations

from typing import Optional

import torch

_MASK32 = 0xFFFFFFFF
_UINT32_SCALE = float(1 << 32)


def _rotl32(value: torch.Tensor, shift: int) -> torch.Tensor:
    return ((value << shift) | (value >> (32 - shift))) & _MASK32


def _murmur3_mix(state: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    value = (value * 0xCC9E2D51) & _MASK32
    value = _rotl32(value, 15)
    value = (value * 0x1B873593) & _MASK32
    state = state ^ value
    state = _rotl32(state, 13)
    return (state * 5 + 0xE6546B64) & _MASK32


def _fmix32(value: torch.Tensor) -> torch.Tensor:
    value = value ^ (value >> 16)
    value = (value * 0x85EBCA6B) & _MASK32
    value = value ^ (value >> 13)
    value = (value * 0xC2B2AE35) & _MASK32
    return value ^ (value >> 16)


def hash_contexts(
    context_token_ids: torch.Tensor, context_lengths: torch.Tensor
) -> torch.Tensor:
    state = torch.zeros(
        context_token_ids.shape[0], dtype=torch.int64, device=context_token_ids.device
    )
    lengths = context_lengths.to(torch.int64)
    for index in range(context_token_ids.shape[1]):
        mixed = _murmur3_mix(state, context_token_ids[:, index].to(torch.int64))
        state = torch.where(index < lengths, mixed, state)
    return _fmix32(state ^ (lengths * 4))


def _watermark_hash32_torch(
    keys: torch.Tensor, context_hashes: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    state = torch.zeros(
        (keys.shape[0], token_ids.shape[0]),
        dtype=torch.int64,
        device=keys.device,
    )
    keys = keys.view(-1, 1)
    state = _murmur3_mix(state, keys & _MASK32)
    state = _murmur3_mix(state, (keys >> 32) & _MASK32)
    state = _murmur3_mix(state, context_hashes.view(-1, 1) & _MASK32)
    state = _murmur3_mix(state, token_ids.view(1, -1) & _MASK32)
    return _fmix32(state ^ 16)


def _dual_key_a_mask_torch(
    keys_a: torch.Tensor,
    keys_b: torch.Tensor,
    context_hashes: torch.Tensor,
    mixing_thresholds: torch.Tensor,
) -> torch.Tensor:
    state = torch.zeros_like(keys_a, dtype=torch.int64)
    state = _murmur3_mix(state, keys_a & _MASK32)
    state = _murmur3_mix(state, (keys_a >> 32) & _MASK32)
    state = _murmur3_mix(state, keys_b & _MASK32)
    state = _murmur3_mix(state, (keys_b >> 32) & _MASK32)
    state = _murmur3_mix(state, context_hashes & _MASK32)
    hashed = _fmix32(state ^ 20)
    return hashed < mixing_thresholds


def _log_uniform_from_hash_torch(hashed: torch.Tensor) -> torch.Tensor:
    lower = torch.log((hashed.to(torch.float32) + 0.5) / _UINT32_SCALE)
    complement = (_MASK32 - hashed).to(torch.float32) + 0.5
    upper = torch.log1p(-complement / _UINT32_SCALE)
    return torch.where(hashed < (1 << 31), lower, upper)


def select_watermark_tokens_torch(
    probabilities: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    keys_b: Optional[torch.Tensor] = None,
    mixing_thresholds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if (keys_b is None) != (mixing_thresholds is None):
        raise ValueError(
            "dual-key watermark selection requires both key B and mixing thresholds"
        )
    token_ids = torch.arange(probabilities.shape[-1], device=probabilities.device)
    hashed = _watermark_hash32_torch(keys, context_hashes, token_ids)
    if keys_b is not None:
        hashed_b = _watermark_hash32_torch(keys_b, context_hashes, token_ids)
        use_key_a = _dual_key_a_mask_torch(
            keys, keys_b, context_hashes, mixing_thresholds
        )
        hashed = torch.where(use_key_a.view(-1, 1), hashed, hashed_b)
    scores = torch.where(
        probabilities > 0,
        _log_uniform_from_hash_torch(hashed) / probabilities.to(torch.float32),
        -torch.inf,
    )
    return scores.argmax(dim=-1)


def _truncate_probabilities(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits / temperatures, dim=-1)
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    ranks = torch.arange(logits.shape[-1], device=logits.device).view(1, -1)
    keep = ranks < top_ks.view(-1, 1)
    keep &= (cumulative_probabilities - sorted_probabilities) <= top_ps.view(-1, 1)
    keep &= sorted_probabilities >= (sorted_probabilities[:, :1] * min_ps.view(-1, 1))
    sorted_probabilities = torch.where(keep, sorted_probabilities, 0.0)
    sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter_(
        dim=-1, index=sorted_indices, src=sorted_probabilities
    )


def force_watermark_tokens(
    logits: torch.Tensor,
    context_hashes: torch.Tensor,
    eligible: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    keys: torch.Tensor,
    keys_b: Optional[torch.Tensor] = None,
    mixing_thresholds: Optional[torch.Tensor] = None,
    max_top_k: Optional[int] = None,
    partial_scores: Optional[torch.Tensor] = None,
    partial_token_ids: Optional[torch.Tensor] = None,
    output_token_ids: Optional[torch.Tensor] = None,
    max_probability: float = 1.0,
) -> torch.Tensor:
    if logits.is_cuda:
        from sglang.kernels.ops.sampling.textseal_selector import (
            force_watermark_tokens_triton,
        )

        selected = force_watermark_tokens_triton(
            logits,
            context_hashes,
            eligible,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            keys,
            keys_b,
            mixing_thresholds,
            max_top_k=max_top_k,
            partial_scores=partial_scores,
            partial_token_ids=partial_token_ids,
            output_token_ids=output_token_ids,
            max_probability=max_probability,
        )
        return eligible & (selected >= 0)

    probabilities = _truncate_probabilities(
        logits, temperatures, top_ks, top_ps, min_ps
    )
    eligible = eligible & (probabilities.amax(dim=-1) <= max_probability)
    rows = eligible.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return eligible
    selected = select_watermark_tokens_torch(
        probabilities[rows].to(torch.float32),
        context_hashes[rows],
        keys[rows],
        keys_b[rows] if keys_b is not None else None,
        mixing_thresholds[rows] if mixing_thresholds is not None else None,
    )
    logits[rows] = -torch.inf
    logits[rows, selected] = 0.0
    return eligible
