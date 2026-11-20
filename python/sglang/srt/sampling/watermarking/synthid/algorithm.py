# Copyright 2024 DeepMind Technologies Limited
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
# Modified by the SGLang Team: adapted from google-deepmind/synthid-text@addb4a15
# (src/synthid_text/logits_processing.py and src/synthid_text/hashing_function.py).
"""SynthID Text hashing, configuration, and the tournament transform."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Sequence

import torch

from sglang.kernels.ops.sampling.synthid import (
    MAX_CONTEXT_HISTORY_SIZE,
    MAX_KEYS,
)

_HASH_MULTIPLIER = 6364136223846793005
_HASH_INCREMENT = 1
_HASH_ROUNDS = 12
_HASH_SHIFT = 5
_G_VALUE_SHIFT = 30
_INT64_MASK = (1 << 64) - 1
_INT64_SIGN_BIT = 1 << 63

_MAX_TOP_K = 65536

# Public reference keys from google-deepmind/synthid-text. These are suitable
# for compatibility tests and benchmarks, not for a production watermark.
REFERENCE_SYNTHID_KEYS = (
    654,
    400,
    836,
    123,
    340,
    443,
    597,
    160,
    57,
    29,
    590,
    639,
    13,
    715,
    468,
    990,
    966,
    226,
    324,
    585,
    118,
    504,
    421,
    521,
    129,
    669,
    732,
    225,
    90,
    960,
)


def _wrap_int64(value: int) -> int:
    value &= _INT64_MASK
    return value - (1 << 64) if value >= _INT64_SIGN_BIT else value


def _accumulate_hash_python(current_hash: int, value: int) -> int:
    return _wrap_int64(
        _wrap_int64(current_hash + int(value)) * _HASH_MULTIPLIER + _HASH_INCREMENT
    )


def hash_context_python(context: Sequence[int], hash_iv: int) -> int:
    """Host twin of ``compute_context_hash_torch`` for one context."""
    result = hash_iv
    for token_id in context:
        result = _accumulate_hash_python(result, token_id)
    return result


def compute_synthid_key_iv(keys: Sequence[int]) -> int:
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
    """Watermark parameters. ``top_k`` caps the candidates the tournament sees;
    ``context_history_size`` bounds the ring of recent context hashes that
    suppresses repeated contexts."""

    keys: tuple[int, ...] = field(repr=False)
    ngram_len: int = 5
    top_k: int = 1024
    context_history_size: int = 1024
    skip_first_ngram_calls: bool = False
    hash_iv: int = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= len(self.keys) <= MAX_KEYS:
            raise ValueError(f"SynthID requires 1 to {MAX_KEYS} keys.")
        if self.ngram_len < 2:
            raise ValueError("SynthID ngram_len must be at least 2.")
        if not 2 <= self.top_k <= _MAX_TOP_K:
            raise ValueError(f"SynthID top_k must be from 2 to {_MAX_TOP_K}.")
        if not 1 <= self.context_history_size <= MAX_CONTEXT_HISTORY_SIZE:
            raise ValueError(
                "SynthID context_history_size must be from 1 to "
                f"{MAX_CONTEXT_HISTORY_SIZE}."
            )
        # Also rejects repeated keys.
        object.__setattr__(self, "hash_iv", compute_synthid_key_iv(self.keys))

    @property
    def context_width(self) -> int:
        return self.ngram_len - 1


def _accumulate_hash_torch(
    current_hash: torch.Tensor, data: torch.Tensor | int
) -> torch.Tensor:
    """Apply the reference signed-int64 hash step with two's-complement wrap."""
    return (current_hash + data) * _HASH_MULTIPLIER + _HASH_INCREMENT


def compute_context_hash_torch(contexts: torch.Tensor, hash_iv: int) -> torch.Tensor:
    """Hash the final dimension of an int64 context tensor."""
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
    for _ in range(_HASH_ROUNDS):
        ngram_keys = torch.bitwise_right_shift(
            _accumulate_hash_torch(ngram_keys, 1), _HASH_SHIFT
        )
    return torch.remainder(torch.bitwise_right_shift(ngram_keys, _G_VALUE_SHIFT), 2)


def watermark_topk_scores_torch(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the reference two-leaf SynthID tournament to sparse top-k scores."""
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
