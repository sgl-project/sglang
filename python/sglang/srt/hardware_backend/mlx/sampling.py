"""MLX-native in-graph sampling for the MLX backend.

Token selection (temperature / top-k / top-p / min-p / per-request seed)
built entirely from ``mx`` ops, so it lives inside the same lazy graph as
the forward pass.  That is what lets sampling coexist with the overlap
scheduler: ``decode_batch_start_chained`` feeds step N's still-unevaluated
sampled tokens as step N+1's input ids, with no host sync.

Semantics mirror ``top_k_top_p_min_p_sampling_from_probs_torch`` /
``multinomial_with_seed`` in ``sglang/srt/layers/sampler.py``.

Gaps against that backend:

* Penalties (frequency/presence/repetition) are not applied (warned once
  per process).
* Custom logit processors run on pure-decode steps only: the first
  generated token and decode steps mixed into an extend batch are not
  processed (``apply_custom_logit_processor`` requires logits rows to
  match the full ``sampling_info``).
* Logprobs cover the sampled token, top-k, and requested token ids;
  prompt/input logprobs (``logprob_start_len``) are not computed.
* Seeded determinism is MLX-local: noise math runs in float32 (Metal has
  no float64) and tie order follows MLX's sort, so the same seed on a
  CUDA backend may pick a different token from the same distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Seed given to rows without an explicit ``sampling_seed`` when
# --enable-deterministic-inference is on.  Mirrors the literal in
# ``SamplingBatchInfo.from_schedule_batch``.
DEFAULT_SAMPLING_SEED = 42

# Largest ``top_k`` that still takes the bounded candidate path in
# :func:`sample_tokens`.  Past this the [B, K] chain stops being
# meaningfully cheaper than the [B, vocab] one it replaces.
MAX_BOUNDED_TOP_K = 1024

_warned_ignored_penalties = False


@dataclass(frozen=True)
class MlxSamplingParams:
    """Per-request sampling parameters, frozen at prefill registration."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    seed: int | None

    @classmethod
    def from_req(
        cls, req: Any, deterministic_seeding: bool = False
    ) -> MlxSamplingParams:
        sp = req.sampling_params
        global _warned_ignored_penalties
        if not _warned_ignored_penalties and (
            sp.frequency_penalty != 0.0
            or sp.presence_penalty != 0.0
            or sp.repetition_penalty != 1.0
        ):
            _warned_ignored_penalties = True
            logger.warning(
                "MLX sampling ignores frequency/presence/repetition penalties; "
                "a request specified them. (Warning logged once.)"
            )
        # Seed contract, identical to every other backend: SamplingBatchInfo
        # populates sampling_seed only under --enable-deterministic-inference,
        # and then seeds every row (default DEFAULT_SAMPLING_SEED).  Outside
        # that flag a per-request seed is ignored, so it is ignored here too.
        seed = None
        if deterministic_seeding:
            seed = (
                sp.sampling_seed
                if sp.sampling_seed is not None
                else DEFAULT_SAMPLING_SEED
            )
        return cls(
            temperature=sp.temperature,
            top_k=sp.top_k,
            top_p=sp.top_p,
            min_p=sp.min_p,
            seed=seed,
        )

    @property
    def is_greedy(self) -> bool:
        return self.top_k == 1


GREEDY_PARAMS = MlxSamplingParams(
    temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, seed=None
)


@dataclass(frozen=True)
class MlxLogprobSpec:
    """Per-row logprob request for one step.

    Mirrors the CUDA ``OutputLogprobProcessor`` inputs: ``top_ks[i]`` is
    the row's ``top_logprobs_num`` (0 = none) and ``token_ids[i]`` the
    row's requested token ids (None = none).
    """

    top_ks: tuple[int, ...]
    token_ids: tuple[tuple[int, ...] | None, ...]


@dataclass
class MlxLazyLogprobs:
    """Lazy logprob arrays for one step; materialized at finalize."""

    chosen: mx.array  # [B]
    top_val: mx.array | None  # [B, max_k]
    top_idx: mx.array | None  # [B, max_k]
    token_ids_val: list[mx.array | None]  # per row
    spec: MlxLogprobSpec


@dataclass
class MlxStepLogprobs:
    """Materialized per-step logprobs, per row, cut to the request shape."""

    chosen: list[float]
    top_val: list[list[float]]
    top_idx: list[list[int]]
    token_ids_val: list[list[float]]
    token_ids_idx: list[list[int]]


def lazy_logprob_arrays(lazy_logprobs: MlxLazyLogprobs | None) -> list[mx.array]:
    """The mx arrays of a lazy logprob bundle, for eval/async_eval calls."""
    if lazy_logprobs is None:
        return []
    arrays = [lazy_logprobs.chosen]
    if lazy_logprobs.top_val is not None:
        arrays += [lazy_logprobs.top_val, lazy_logprobs.top_idx]
    arrays += [a for a in lazy_logprobs.token_ids_val if a is not None]
    return arrays


def all_greedy(params: list[MlxSamplingParams]) -> bool:
    return all(p.is_greedy for p in params)


def sanitize_logits(logits: mx.array) -> mx.array:
    """Lazy analogue of sglang's ``sanitize_nan_logits``: NaN -> -1e30,
    +-inf -> +-1e30 (not dtype extremes — temperature division would
    overflow those back to inf)."""
    return mx.clip(mx.where(mx.isnan(logits), -1e30, logits), -1e30, 1e30)


def scale_by_temperature(
    last_logits: mx.array, params: list[MlxSamplingParams]
) -> mx.array:
    """``logits / temperature`` per row, in float32.

    Both :func:`sample_tokens` and :func:`compute_logprobs` start here and take
    the result as their ``scaled`` argument: MLX does not eliminate common
    subexpressions, so a step doing both would otherwise divide twice.
    """
    temps = mx.array([p.temperature for p in params], dtype=mx.float32)[:, None]
    return last_logits.astype(mx.float32) / temps


def compute_logprobs(
    last_logits: mx.array,
    params: list[MlxSamplingParams],
    tokens: mx.array,
    spec: MlxLogprobSpec,
    scaled: mx.array | None = None,
) -> MlxLazyLogprobs:
    """Lazy log-probabilities of this step's distribution.

    Matches the pytorch sampler: the distribution is
    ``log_softmax(edited_logits / temperature)`` — after grammar-mask /
    logit_bias / sanitization, before top-k/top-p/min-p filtering (the
    filters affect which token is drawn, not the reported logprobs).
    """
    if scaled is None:
        scaled = scale_by_temperature(last_logits, params)
    logp = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)

    chosen = mx.take_along_axis(logp, tokens[:, None], axis=-1).squeeze(-1)

    max_k = max(spec.top_ks) if spec.top_ks else 0
    if max_k > 0:
        top_idx = mx.argsort(-logp, axis=-1)[:, :max_k]
        top_val = mx.take_along_axis(logp, top_idx, axis=-1)
    else:
        top_idx = None
        top_val = None

    token_ids_val: list[mx.array | None] = [
        logp[row, mx.array(ids)] if ids else None
        for row, ids in enumerate(spec.token_ids)
    ]
    return MlxLazyLogprobs(
        chosen=chosen,
        top_val=top_val,
        top_idx=top_idx,
        token_ids_val=token_ids_val,
        spec=spec,
    )


def sample_tokens(
    last_logits: mx.array,
    params: list[MlxSamplingParams],
    positions: list[int],
    key: mx.array,
    scaled: mx.array | None = None,
) -> mx.array:
    """Select one token per row of ``last_logits`` ([B, vocab], lazy ok).

    Pure ``mx`` ops — the result stays inside the lazy graph.  ``positions``
    are the absolute sequence positions being sampled (seeded rows only).
    Assumes at least one row samples; callers shortcut to ``mx.argmax`` when
    ``all_greedy(params)``.
    """
    batch_size, vocab_size = last_logits.shape
    logits32 = last_logits.astype(mx.float32)
    if scaled is None:
        scaled = scale_by_temperature(logits32, params)

    filtering = any(
        not p.is_greedy and (p.top_k < vocab_size or p.top_p < 1.0 or p.min_p > 0.0)
        for p in params
    )
    # Token ids the Gumbel-max runs over: None means "the whole vocabulary",
    # otherwise a [B, K] candidate array that the argmax result indexes into.
    candidates: mx.array | None = None
    if filtering:
        probs = mx.softmax(scaled, axis=-1)
        width = _candidate_width(params, vocab_size)
        sorted_idx = mx.argsort(-probs, axis=-1)[:, :width]
        p_sort = mx.take_along_axis(probs, sorted_idx, axis=-1)
        ranks = mx.arange(width, dtype=mx.int32)[None, :]
        # SamplingParams normalizes top_k=-1 to TOP_K_ALL and temperature
        # below eps to (temperature=1, top_k=1), so min(top_k, width)
        # is always >= 1: rank 0 survives and log(weights) is never all -inf.
        top_ks = mx.array([min(p.top_k, width) for p in params], dtype=mx.int32)[
            :, None
        ]
        top_ps = mx.array([p.top_p for p in params], dtype=mx.float32)[:, None]
        min_ps = mx.array([p.min_p for p in params], dtype=mx.float32)[:, None]
        cum = mx.cumsum(p_sort, axis=-1)
        masked_out = (
            (ranks >= top_ks)
            | ((cum - p_sort) > top_ps)
            | (p_sort < p_sort[:, :1] * min_ps)
        )
        w_sort = mx.where(masked_out, 0.0, p_sort)
        if width < vocab_size:
            # Bounded top-K: every surviving rank is inside the K
            # candidates, so run the rest of the chain in candidate space
            # and map the winning rank back to its vocab id at the end.
            log_weights = mx.log(w_sort)
            candidates = sorted_idx
        else:
            # Scatter the masked weights back to vocab order: noise must be
            # applied in vocab-id space in every branch, or a seeded row's
            # token would change when a batchmate happens to need filtering.
            weights = mx.put_along_axis(
                mx.zeros_like(probs), sorted_idx, w_sort, axis=-1
            )
            log_weights = mx.log(weights)
    else:
        # Nothing is masked, so log(softmax(scaled)) is just scaled minus a
        # per-row constant: feeding the scaled logits straight to the argmax
        # below drops two full-vocab passes (softmax, log).
        log_weights = scaled

    noise = _gumbel_noise(
        params=params,
        positions=positions,
        shape=log_weights.shape,
        key=key,
        columns=candidates,
    )
    # Gumbel-max over the UNNORMALIZED masked weights: normalization would only
    # shift log(w) by a per-row constant, which argmax is invariant to.  That is
    # also what keeps a seeded row well-defined under min_p.
    sampled = mx.argmax(log_weights + noise, axis=-1)
    if candidates is not None:
        sampled = mx.take_along_axis(candidates, sampled[:, None], axis=-1).squeeze(-1)

    greedy = [p.is_greedy for p in params]
    if not any(greedy):
        return sampled
    # Greedy rows still ran the sampled path above (the row exists either way);
    # overwriting them with the unnoised argmax is what keeps their token
    # independent of the noise.
    return mx.where(mx.array(greedy), mx.argmax(logits32, axis=-1), sampled)


def _candidate_width(params: list[MlxSamplingParams], vocab_size: int) -> int:
    """Rank cut-off the filtered chain can run on, or ``vocab_size``.

    Every rank at or beyond a row's ``top_k`` is masked to weight 0, whose
    ``log`` is ``-inf``, so the Gumbel-max can never select it: when every
    row's ``top_k`` is small the whole chain after the sort can run on
    ``[B, K]`` instead of ``[B, vocab]`` and still pick the same token.
    Falls back to the full vocabulary as soon as one row wants more than
    :data:`MAX_BOUNDED_TOP_K` (or no top-k at all, which ``SamplingParams``
    spells as ``top_k = TOP_K_ALL``).
    """
    largest_top_k = max(p.top_k for p in params)
    if largest_top_k > MAX_BOUNDED_TOP_K or largest_top_k >= vocab_size:
        return vocab_size
    return largest_top_k


def _gumbel_noise(
    params: list[MlxSamplingParams],
    positions: list[int],
    shape: tuple[int, int],
    key: mx.array,
    columns: mx.array | None = None,
) -> mx.array:
    """Gumbel noise shaped like the weights: hashed if seeded, RNG otherwise.

    ``columns`` is the [B, K] token ids each weight column stands for on the
    bounded top-K path; ``None`` means column j is token id j.  Seeded rows
    hash the token id, so their noise — and therefore their token — is the
    same either way.
    """
    seeded_rows = [p.seed is not None for p in params]
    if not any(seeded_rows):
        return mx.random.gumbel(shape=shape, key=key)

    hashed = _murmur_hash32(
        seeds=[p.seed if p.seed is not None else 0 for p in params],
        positions=positions,
        vocab_size=shape[1],
        columns=columns,
    )
    u = hashed.astype(mx.float32) / float(0xFFFFFFFF)
    # uint32(0xFFFFFFFF) rounds UP to 2**32 in float32, so the quotient can land
    # just above 1.0 (log(-log(u)) -> NaN); u == 1.0 gives +inf, which would
    # deterministically force that token.
    u = mx.clip(u, 2.0**-32, 1.0 - 2.0**-24)
    hash_noise = -mx.log(-mx.log(u))

    if all(seeded_rows):
        return hash_noise
    random_noise = mx.random.gumbel(shape=shape, key=key)
    return mx.where(mx.array(seeded_rows)[:, None], hash_noise, random_noise)


def _murmur3_mix_py(h: int, k: int) -> int:
    """One MurmurHash3 block mix on Python ints (exact 32-bit semantics)."""
    k = (k * 0xCC9E2D51) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * 0x1B873593) & 0xFFFFFFFF
    h ^= k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    return h


def _murmur_hash32(
    seeds: list[int],
    positions: list[int],
    vocab_size: int,
    columns: mx.array | None = None,
) -> mx.array:
    """Port of ``murmur_hash32`` (Triton) to mx ops: [B, V] uint32.

    Blocks mixed in kernel order: seed_low, seed_high, position, column.  The
    first three are per-row scalars, folded exactly on the CPU with Python
    ints; only the column block and finalization run as vectorized uint32 ops.
    ``columns`` hashes an explicit [B, K] set of token ids instead of every id
    in ``[0, vocab_size)``.
    """
    row_states = []
    for seed, pos in zip(seeds, positions):
        seed &= 0xFFFFFFFFFFFFFFFF
        h = _murmur3_mix_py(0, seed & 0xFFFFFFFF)
        h = _murmur3_mix_py(h, (seed >> 32) & 0xFFFFFFFF)
        h = _murmur3_mix_py(h, pos & 0xFFFFFFFF)
        row_states.append(h)

    h = mx.array(row_states, dtype=mx.uint32)[:, None]
    if columns is None:
        k = mx.arange(vocab_size, dtype=mx.uint32)[None, :]
    else:
        k = columns.astype(mx.uint32)

    # murmur3_mix on [B, V]
    k = k * mx.array(0xCC9E2D51, dtype=mx.uint32)
    k = (k << 15) | (k >> 17)
    k = k * mx.array(0x1B873593, dtype=mx.uint32)
    h = h ^ k
    h = (h << 13) | (h >> 19)
    h = h * mx.array(5, dtype=mx.uint32) + mx.array(0xE6546B64, dtype=mx.uint32)

    # finalize: len = 16 bytes (seed + pos + col), then fmix32
    h = h ^ mx.array(16, dtype=mx.uint32)
    h = h ^ (h >> 16)
    h = h * mx.array(0x85EBCA6B, dtype=mx.uint32)
    h = h ^ (h >> 13)
    h = h * mx.array(0xC2B2AE35, dtype=mx.uint32)
    h = h ^ (h >> 16)
    return h
