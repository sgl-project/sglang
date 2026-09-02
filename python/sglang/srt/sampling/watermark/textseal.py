"""TextSeal generation core adapted from Meta TextSeal v0.2.0.

Source: facebookresearch/textseal commit c60d0d1da2e59f09a698438e218a07ee779b4616,
Apache-2.0. SGLang adds batched keys, deterministic key choice, and
effective-support inputs.
"""

import hashlib
from collections.abc import Sequence

import torch

_PRIMES = [
    10000019,
    10000247,
    10000439,
    10000643,
    10000747,
    10000867,
    10000993,
    10001213,
    10001357,
    10001501,
]
_P2 = 100000007
_P3 = 500001713
_P4 = 15485863
_M = 8191
_MIXING_PRIME = 40499
_MIXING_SHIFT = 13
_KEY_CHOICE_DOMAIN = 0x53474C545854534C


def _weighted_sum(contexts: torch.Tensor) -> torch.Tensor:
    primes = torch.tensor(
        _PRIMES[: contexts.shape[-1]], dtype=torch.long, device=contexts.device
    )
    return (contexts.long() * primes).sum(dim=-1)


def _weighted_sum_by_ngram(
    contexts: torch.Tensor, ngrams: torch.Tensor
) -> torch.Tensor:
    width = contexts.shape[-1]
    context_positions = (
        torch.arange(width, device=contexts.device).view(1, -1)
        - width
        + ngrams.view(-1, 1)
    )
    primes = torch.tensor(_PRIMES[:width], dtype=torch.long, device=contexts.device)
    weights = primes[context_positions.clamp_min(0)]
    weights = torch.where(context_positions >= 0, weights, 0)
    return (contexts.long() * weights).sum(dim=-1)


def _expand_rows(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def _hash(
    weighted_contexts: torch.Tensor,
    token_ids: torch.Tensor,
    secret_keys: int | torch.Tensor,
) -> torch.Tensor:
    weighted_contexts = _expand_rows(weighted_contexts, token_ids)
    if isinstance(secret_keys, int):
        secret_keys = torch.full_like(token_ids, secret_keys, dtype=torch.long)
    else:
        secret_keys = _expand_rows(secret_keys.to(token_ids.device).long(), token_ids)
    hashed = (weighted_contexts + _P2 * token_ids.long() + _P3 * secret_keys) * _P4
    hashed = hashed * _MIXING_PRIME
    hashed = hashed ^ (hashed >> _MIXING_SHIFT)
    return hashed % _M


def prf_uniform(
    contexts: torch.Tensor,
    token_ids: torch.Tensor,
    secret_keys: int | torch.Tensor,
) -> torch.Tensor:
    token_ids = token_ids.to(contexts.device)
    return _hash(_weighted_sum(contexts), token_ids, secret_keys).float() / _M


def prf_dual(
    contexts: torch.Tensor,
    token_ids: torch.Tensor,
    key_a: int | torch.Tensor,
    key_b: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_contexts = _weighted_sum(contexts)
    token_ids = token_ids.to(contexts.device)
    return (
        _hash(weighted_contexts, token_ids, key_a).float() / _M,
        _hash(weighted_contexts, token_ids, key_b).float() / _M,
    )


def request_nonce(request_id: str) -> int:
    digest = hashlib.blake2b(
        request_id.encode("utf-8"), digest_size=8, person=b"SGLTextSeal"
    ).digest()
    return int.from_bytes(digest, "little", signed=True)


def deterministic_key_a_mask(
    nonces: torch.Tensor,
    positions: torch.Tensor,
    mixing_probabilities: torch.Tensor,
) -> torch.Tensor:
    contexts = nonces.long().view(-1, 1)
    scores = prf_uniform(contexts, positions.long(), _KEY_CHOICE_DOMAIN)
    return scores < mixing_probabilities


def context_from_token_ids(
    token_ids: Sequence[int], ngram: int, *, device: torch.device | str
) -> torch.Tensor:
    tail = list(token_ids[-ngram:])
    return torch.tensor(
        [0] * (ngram - len(tail)) + tail,
        dtype=torch.long,
        device=device,
    )


def select_textseal_tokens(
    effective_probs: torch.Tensor,
    contexts: torch.Tensor,
    key_a: torch.Tensor,
    key_b: torch.Tensor,
    use_key_a: torch.Tensor,
    token_ids: torch.Tensor | None = None,
    ngrams: torch.Tensor | None = None,
) -> torch.Tensor:
    weighted_contexts = (
        _weighted_sum(contexts)
        if ngrams is None
        else _weighted_sum_by_ngram(contexts, ngrams)
    )
    keys = torch.where(use_key_a, key_a, key_b)
    if effective_probs.is_cuda and token_ids is not None:
        from sglang.kernels.ops.sampling.textseal_selector import (
            select_textseal_tokens_triton,
        )

        return select_textseal_tokens_triton(
            effective_probs,
            token_ids,
            weighted_contexts,
            keys,
        )

    if token_ids is None:
        token_ids = torch.arange(
            effective_probs.shape[-1], device=effective_probs.device
        ).expand_as(effective_probs)
    uniform_scores = _hash(weighted_contexts, token_ids, keys).float() / _M
    selection_scores = torch.log(uniform_scores + 1e-30) / (effective_probs + 1e-30)
    selected_indices = torch.argmax(selection_scores, dim=-1, keepdim=True)
    return torch.gather(token_ids, -1, selected_indices).reshape(-1).to(torch.int32)
