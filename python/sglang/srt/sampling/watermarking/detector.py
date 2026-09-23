from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Iterable, Optional, Protocol, Sequence

import msgspec

from sglang.srt.sampling.watermarking.config import (
    MAX_WATERMARK_CONTEXT_WINDOW,
    MAX_WATERMARKED_CONTEXTS_PER_REQUEST,
    parse_watermark_key,
)

_MASK32 = 0xFFFFFFFF
_UINT32_SCALE = 1 << 32


def _rotl32(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & _MASK32


def _murmur3_words(words: Sequence[int]) -> int:
    state = 0
    for word in words:
        value = (int(word) * 0xCC9E2D51) & _MASK32
        value = (_rotl32(value, 15) * 0x1B873593) & _MASK32
        state = (_rotl32(state ^ value, 13) * 5 + 0xE6546B64) & _MASK32
    state ^= 4 * len(words)
    state ^= state >> 16
    state = (state * 0x85EBCA6B) & _MASK32
    state ^= state >> 13
    state = (state * 0xC2B2AE35) & _MASK32
    return state ^ (state >> 16)


def hash_context(token_ids: Sequence[int]) -> int:
    return _murmur3_words([int(token_id) & _MASK32 for token_id in token_ids])


def watermark_hash(key: int, context_hash: int, token_id: int) -> int:
    return _murmur3_words(
        (key & _MASK32, (key >> 32) & _MASK32, context_hash, token_id)
    )


def position_coin(key_a: int, key_b: int, context_hash: int) -> int:
    return _murmur3_words(
        (
            key_a & _MASK32,
            (key_a >> 32) & _MASK32,
            key_b & _MASK32,
            (key_b >> 32) & _MASK32,
            context_hash,
        )
    )


def _gamma_log_survival(score: float, shape: int) -> float:
    if score == 0:
        return 0.0
    log_score = math.log(score)
    terms = [k * log_score - math.lgamma(k + 1) for k in range(shape)]
    peak = max(terms)
    tail = -score + peak + math.log(math.fsum(math.exp(t - peak) for t in terms))
    return min(0.0, tail)


class WatermarkStatistics(msgspec.Struct, frozen=True):
    num_contexts: int
    score: float
    z_score: float
    p_value: float
    # Survives when p_value underflows to 0.0 on long watermarked texts.
    log_p_value: float

    @classmethod
    def from_scores(cls, scores: Sequence[float]) -> WatermarkStatistics:
        n = len(scores)
        if n == 0:
            return cls(
                num_contexts=0, score=0.0, z_score=0.0, p_value=1.0, log_p_value=0.0
            )
        score = math.fsum(scores)
        log_p = _gamma_log_survival(score, n)
        return cls(
            num_contexts=n,
            score=score,
            z_score=(score - n) / math.sqrt(n),
            p_value=math.exp(log_p),
            log_p_value=log_p,
        )


class WatermarkDetection(msgspec.Struct, frozen=True, kw_only=True):
    combined: WatermarkStatistics
    key_a_all_positions: WatermarkStatistics
    key_b_all_positions: Optional[WatermarkStatistics]
    key_a_partition: Optional[WatermarkStatistics]
    key_b_partition: Optional[WatermarkStatistics]
    prefix_limit_reached: bool
    tokens_examined: int
    repeated_contexts: int
    skipped_initial_tokens: int


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def _validate_token_ids(values: Iterable[int]) -> list[int]:
    token_ids = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value <= _MASK32
        ):
            raise ValueError("token IDs must be unsigned 32-bit integers")
        token_ids.append(int(value))
    return token_ids


def _validate_int_range(name: str, value: int, limit: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 1 <= value <= limit
    ):
        raise ValueError(f"{name} must be an integer from 1 to {limit}")
    return int(value)


class WatermarkDetector:
    def __init__(
        self,
        key: str,
        *,
        key_b: Optional[str] = None,
        mixing_probability: float = 0.5,
        context_window: int = 4,
        max_contexts: int = MAX_WATERMARKED_CONTEXTS_PER_REQUEST,
    ) -> None:
        if (
            isinstance(mixing_probability, bool)
            or not isinstance(mixing_probability, Real)
            or not 0 < mixing_probability < 1
        ):
            raise ValueError(
                "watermark mixing probability must be strictly between 0 and 1"
            )
        self._key_a = parse_watermark_key(key)
        self._key_b = parse_watermark_key(key_b) if key_b is not None else None
        self._mixing_threshold = int(mixing_probability * _UINT32_SCALE)
        self.context_window = _validate_int_range(
            "context_window", context_window, MAX_WATERMARK_CONTEXT_WINDOW
        )
        self.max_contexts = _validate_int_range(
            "max_contexts", max_contexts, MAX_WATERMARKED_CONTEXTS_PER_REQUEST
        )

    def detect_tokens(
        self,
        token_ids: Iterable[int],
        *,
        prompt_token_ids: Iterable[int] = (),
    ) -> WatermarkDetection:
        prompt = _validate_token_ids(prompt_token_ids)
        completion = _validate_token_ids(token_ids)
        tokens = prompt[-self.context_window :] + completion
        start = min(len(prompt), self.context_window)
        # Without prompt IDs, the first h contexts would include unknown prompt tokens.
        first = start if prompt else min(self.context_window, len(completion))

        seen = set()
        scores_a, scores_b, partition_a, partition_b = [], [], [], []
        repeated = 0
        examined = first - start
        for position in range(first, len(tokens)):
            examined = position - start + 1
            context = tuple(tokens[max(0, position - self.context_window) : position])
            if context in seen:
                repeated += 1
                continue
            seen.add(context)
            context_hash = hash_context(context)
            score_a = self._score(self._key_a, context_hash, tokens[position])
            scores_a.append(score_a)
            if self._key_b is not None:
                score_b = self._score(self._key_b, context_hash, tokens[position])
                scores_b.append(score_b)
                coin = position_coin(self._key_a, self._key_b, context_hash)
                if coin < self._mixing_threshold:
                    partition_a.append(score_a)
                else:
                    partition_b.append(score_b)
            if len(seen) == self.max_contexts:
                break

        dual = self._key_b is not None
        stats = WatermarkStatistics.from_scores
        return WatermarkDetection(
            combined=stats(partition_a + partition_b if dual else scores_a),
            key_a_all_positions=stats(scores_a),
            key_b_all_positions=stats(scores_b) if dual else None,
            key_a_partition=stats(partition_a) if dual else None,
            key_b_partition=stats(partition_b) if dual else None,
            prefix_limit_reached=len(seen) == self.max_contexts,
            tokens_examined=examined,
            repeated_contexts=repeated,
            skipped_initial_tokens=first - start,
        )

    def detect_text(
        self,
        text: str,
        tokenizer: Tokenizer,
        *,
        prompt_token_ids: Iterable[int] = (),
    ) -> WatermarkDetection:
        return self.detect_tokens(
            tokenizer.encode(text, add_special_tokens=False),
            prompt_token_ids=prompt_token_ids,
        )

    @staticmethod
    def _score(key: int, context_hash: int, token_id: int) -> float:
        uniform = (watermark_hash(key, context_hash, token_id) + 0.5) / _UINT32_SCALE
        return -math.log1p(-uniform)


def detect(
    token_ids: Iterable[int],
    *,
    key: str,
    key_b: Optional[str] = None,
    prompt_token_ids: Iterable[int] = (),
    mixing_probability: float = 0.5,
    context_window: int = 4,
) -> WatermarkDetection:
    detector = WatermarkDetector(
        key,
        key_b=key_b,
        mixing_probability=mixing_probability,
        context_window=context_window,
    )
    return detector.detect_tokens(token_ids, prompt_token_ids=prompt_token_ids)
