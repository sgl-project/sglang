# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
# Modified by the SGLang Team: adapted from google-deepmind/synthid-text@addb4a15
# (src/synthid_text/logits_processing.py and src/synthid_text/hashing_function.py).

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Sequence

import torch

from sglang.kernels.ops.sampling.synthid import (
    G_VALUE_SHIFT,
    HASH_INCREMENT,
    HASH_MULTIPLIER,
    HASH_ROUNDS,
    HASH_SHIFT,
    MAX_CONTEXT_HISTORY_SIZE,
    MAX_CONTEXT_WIDTH,
    MAX_KEYS,
    MAX_NUM_CANDIDATES,
)

_INT64_MASK = (1 << 64) - 1
_INT64_SIGN_BIT = 1 << 63


def _wrap_int64(value: int) -> int:
    value &= _INT64_MASK
    return value - (1 << 64) if value >= _INT64_SIGN_BIT else value


def hash_context(context: Sequence[int], hash_iv: int) -> int:
    """Host twin of ``hash_contexts`` for one context."""
    result = hash_iv
    for token_id in context:
        result = _wrap_int64(
            (result + int(token_id)) * HASH_MULTIPLIER + HASH_INCREMENT
        )
    return result


def compute_hash_iv(keys: Sequence[int]) -> int:
    """Compute the reference SHA-256 initialization vector for integer keys."""
    if not keys:
        raise ValueError("SynthID requires at least one key.")
    if len(set(keys)) != len(keys):
        raise ValueError("SynthID layer keys must be distinct.")
    try:
        key_bytes = struct.pack(f"<{len(keys)}q", *(int(key) for key in keys))
    except (OverflowError, struct.error) as exc:
        raise ValueError("SynthID keys must be signed 64-bit integers.") from exc
    digest = hashlib.sha256(key_bytes).digest()
    return int.from_bytes(digest, byteorder="big") % ((1 << 63) - 1)


@dataclass(frozen=True)
class SynthIDConfig:
    keys: tuple[int, ...] = field(repr=False)
    ngram_len: int = 5
    num_candidates: int = 1024
    context_history_size: int = 1024
    skip_first_ngram_calls: bool = False
    hash_iv: int = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= len(self.keys) <= MAX_KEYS:
            raise ValueError(f"SynthID requires 1 to {MAX_KEYS} keys.")
        if not 2 <= self.ngram_len <= MAX_CONTEXT_WIDTH + 1:
            raise ValueError(
                f"SynthID ngram_len must be from 2 to {MAX_CONTEXT_WIDTH + 1}."
            )
        if not 2 <= self.num_candidates <= MAX_NUM_CANDIDATES:
            raise ValueError(
                f"SynthID num_candidates must be from 2 to {MAX_NUM_CANDIDATES}."
            )
        if not 1 <= self.context_history_size <= MAX_CONTEXT_HISTORY_SIZE:
            raise ValueError(
                "SynthID context_history_size must be from 1 to "
                f"{MAX_CONTEXT_HISTORY_SIZE}."
            )
        # Also rejects repeated keys and keys outside int64.
        object.__setattr__(self, "hash_iv", compute_hash_iv(self.keys))

    @property
    def context_width(self) -> int:
        return self.ngram_len - 1


def _accumulate_hash_torch(
    current_hash: torch.Tensor, data: torch.Tensor | int
) -> torch.Tensor:
    """Apply the reference signed-int64 hash step with two's-complement wrap."""
    return (current_hash + data) * HASH_MULTIPLIER + HASH_INCREMENT


def hash_contexts(contexts: torch.Tensor, hash_iv: int) -> torch.Tensor:
    result = torch.full(
        contexts.shape[:-1],
        hash_iv,
        dtype=torch.int64,
        device=contexts.device,
    )
    for index in range(contexts.shape[-1]):
        result = _accumulate_hash_torch(result, contexts[..., index])
    return result


def compute_g_values_torch(
    context_hashes: torch.Tensor,
    token_ids: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Return SynthID binary g-values with shape ``[rows, candidates, depth]``."""
    ngram_keys = _accumulate_hash_torch(
        context_hashes[:, None, None],
        token_ids.to(torch.int64)[:, :, None],
    )
    ngram_keys = _accumulate_hash_torch(ngram_keys, keys[None, None, :])
    for _ in range(HASH_ROUNDS):
        ngram_keys = torch.bitwise_right_shift(
            _accumulate_hash_torch(ngram_keys, 1), HASH_SHIFT
        )
    return torch.remainder(torch.bitwise_right_shift(ngram_keys, G_VALUE_SHIFT), 2)


def synthid_tournament_torch(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    """Torch twin of ``synthid_tournament_triton``: the reference two-leaf
    tournament over each row's candidate scores."""
    probs = torch.softmax(scores, dim=-1)
    g_values = compute_g_values_torch(context_hashes, token_ids, keys)
    for depth in range(keys.numel()):
        g_values_at_depth = g_values[:, :, depth]
        g_mass = (g_values_at_depth * probs).sum(dim=-1, keepdim=True)
        probs = probs * (1 + g_values_at_depth - g_mass)

    watermarked_scores = torch.log(probs)
    watermarked_scores = torch.where(
        torch.isfinite(watermarked_scores),
        watermarked_scores,
        torch.full_like(watermarked_scores, -1e12),
    )
    return torch.where(apply_mask[:, None], watermarked_scores, scores)
