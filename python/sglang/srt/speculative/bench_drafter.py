"""Draft temperature for DSpark and DFLASH-family drafters under simulated acceptance
(SGLANG_SIMULATE_ACC_LEN > 1), which commits whatever the drafter proposes.

Greedy drafts collapse into short repeating loops (DFLASH's default token mode even
commits one fixed id), so the model runs on a few repeated token ids, unlike real
traffic, which can skew expert routing and cache locality. Under simulated acceptance
every draft row therefore samples: a sampled request at its own temperature, a greedy
one at 1000 (close to uniform) by default, while the target still verifies greedily.
SGLANG_BENCH_ACC_DRAFT_TEMPERATURE sets the greedy rows' temperature; 0 keeps greedy
drafts. Left unset, an explicit SGLANG_DSPARK_USE_GREEDY_DRAFT=1 or
SGLANG_SIMULATE_ACC_TOKEN_MODE wins over the default.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any, Callable, Optional

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

ENV_NAME = "SGLANG_BENCH_ACC_DRAFT_TEMPERATURE"
SUPPORTED_ALGORITHMS = ("DFLASH", "DSPARK")
# The greedy rows' draft temperature under simulated acceptance when the variable is unset.
DEFAULT_TEMPERATURE = 1000.0
# The request clamp: below it the draft kernels' value / temperature overflows.
_MIN_TEMPERATURE = 1e-5

# Philox domain of the DFLASH argmax-head noise (DSpark keeps its own draft key).
_HEAD_NOISE_DOMAIN = 0x6B6E6F6973652D31
# Counter words separating the head's two noise streams (draft graph, eager fallback).
GRAPH_STEP, EAGER_STEP = 0, 1


def _requested_temperature() -> Optional[float]:
    """The variable as set, validated; None when unset."""
    value = envs.SGLANG_BENCH_ACC_DRAFT_TEMPERATURE.get()
    if value is None:
        return None
    value = float(value)
    if value != 0.0 and not (
        _MIN_TEMPERATURE <= value <= torch.finfo(torch.float32).max
    ):
        raise ValueError(
            f"{ENV_NAME} must be 0 (greedy drafts) or a float32 temperature of at "
            f"least {_MIN_TEMPERATURE}, got {value!r}."
        )
    return value


def temperature_requested() -> bool:
    return _requested_temperature() is not None


@functools.cache
def bench_drafter_temperature() -> float:
    """The greedy rows' draft temperature: the variable when set, else
    DEFAULT_TEMPERATURE under SIMULATE_ACC_LEN > 1, unless greedy DSpark drafts were
    asked for explicitly."""
    requested = _requested_temperature()
    if requested is not None:
        return requested
    if float(envs.SGLANG_SIMULATE_ACC_LEN.get()) <= 1.0:
        return 0.0
    if envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get():
        return 0.0
    return DEFAULT_TEMPERATURE


def bench_drafter_enabled() -> bool:
    return bench_drafter_temperature() > 0.0


def bench_token_mode(configured: str) -> str:
    """DFLASH's simulated-acceptance token mode: the sampled drafts are committed
    (real-draft-token), unless the token mode was set explicitly and the temperature
    was not."""
    if not bench_drafter_enabled():
        return configured
    if envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.is_set() and not temperature_requested():
        return configured
    return "real-draft-token"


def opt_out_hint() -> str:
    """Appended to refusals of the default, which the user did not ask for."""
    if temperature_requested():
        return ""
    return (
        f" ({ENV_NAME} defaults to {DEFAULT_TEMPERATURE:g} under "
        f"SGLANG_SIMULATE_ACC_LEN; set {ENV_NAME}=0 to keep greedy drafts)"
    )


def validate_bench_args(algorithm: Optional[str]) -> None:
    """Startup gate, run on every launch before weights load. A value that is not a
    float warns and falls back to the default, like any SGLang env var."""
    bench_drafter_temperature.cache_clear()  # this launch's variables, not a cached one
    requested = _requested_temperature()
    if algorithm not in SUPPORTED_ALGORITHMS:
        if requested:
            raise ValueError(
                f"{ENV_NAME} applies to --speculative-algorithm "
                f"{' / '.join(SUPPORTED_ALGORITHMS)}, not {algorithm}."
            )
        return
    if not bench_drafter_enabled():
        return
    # Real acceptance would reject the sampled drafts, and a length of 1 commits none.
    simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
    if not 1.0 < simulate_acc_len < math.inf:
        raise ValueError(
            f"{ENV_NAME} requires forced acceptance (a finite SGLANG_SIMULATE_ACC_LEN "
            f"> 1), got SGLANG_SIMULATE_ACC_LEN={simulate_acc_len:g}."
        )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    platform_worker = SpeculativeAlgorithm.from_string(
        algorithm
    ).get_platform_worker_cls()
    if platform_worker is not None:
        raise ValueError(
            f"{ENV_NAME} is applied by the built-in {algorithm} worker only; this "
            f"platform serves {algorithm} with {platform_worker.__name__}"
            f"{opt_out_hint()}."
        )


def override_draft_sampling(
    *, temperatures: torch.Tensor, greedy_mask: torch.Tensor
) -> None:
    """Greedy rows of these draft-only buffers sample at the benchmark temperature."""
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return
    temperatures.masked_fill_(greedy_mask, temperature)
    greedy_mask.fill_(False)


def bench_noise_key(key: int) -> int:
    """Fold the attention-DP rank into a Philox key: TP ranks agree on every draw,
    while DP groups, which hold unrelated requests, draw independently."""
    from sglang.srt.runtime_context import get_parallel

    dp_rank = int(get_parallel().attn_dp_rank)
    return (int(key) ^ (dp_rank * 0x9E3779B97F4A7C15)) & 0x7FFFFFFFFFFFFFFF


class BenchHeadNoise(msgspec.Struct, frozen=True):
    """Gumbel-max sampling for an argmax draft head over the target's vocabulary."""

    key: int
    # The target's output transform, folded onto the head's raw logits: m * x at T
    # is x at T / m, and softcap_c(m * x) is m * softcap_{c / m}(x).
    output_multiplier: float
    softcap: Optional[float]
    # Ids from here on are padding rows, which the target masks as well.
    vocab_limit: int


def bench_head_noise(
    *,
    random_seed: int,
    vocab_limit: int,
    output_multiplier: float = 1.0,
    final_logit_softcapping: Optional[float] = None,
) -> Optional[BenchHeadNoise]:
    if not bench_drafter_enabled():
        return None
    return BenchHeadNoise(
        key=bench_noise_key(int(random_seed) ^ _HEAD_NOISE_DOMAIN),
        output_multiplier=float(output_multiplier),
        softcap=(
            None
            if final_logit_softcapping is None
            else final_logit_softcapping / output_multiplier
        ),
        vocab_limit=int(vocab_limit),
    )


def head_row_temperatures(
    *,
    sampling_info,
    bs: int,
    rows_per_request: int,
    noise: BenchHeadNoise,
    device: torch.device,
) -> torch.Tensor:
    """Per-row temperatures of an argmax draft head, request-major: a sampled request
    at its own temperature, a greedy one at the benchmark temperature, both folded
    onto the head's raw logits."""
    if sampling_info is None:
        per_request = torch.full(
            (bs,), bench_drafter_temperature(), dtype=torch.float32, device=device
        )
    else:
        per_request = (
            sampling_info.temperatures.view(-1)[:bs].float().clamp_min(_MIN_TEMPERATURE)
        )
        override_draft_sampling(
            temperatures=per_request,
            greedy_mask=sampling_info.top_ks.view(-1)[:bs] <= 1,
        )
    per_request = per_request / noise.output_multiplier
    return per_request.repeat_interleave(rows_per_request)


def gumbel_argmax(
    logits: torch.Tensor,
    *,
    noise: BenchHeadNoise,
    temperatures: torch.Tensor,
    offset: torch.Tensor,
    step: int,
    vocab_start: int,
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Global ids drawn from softmax(logits / T) by DSpark's fused Gumbel-max over the
    rank's padded vocab shard; the noise is keyed by global id, so a shard draws its
    slice of the full-vocab draw. ``offset`` advances once per draft call."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    return MarkovSampleStep.execute(
        base_logits=logits,
        bias=None,
        temperatures=temperatures,
        key=noise.key,
        offset=offset,
        step=step,
        step_dtype=torch.float32,
        vocab_start=vocab_start,
        gather=gather,
        has_sampled_rows=True,
        softcap=noise.softcap,
        vocab_limit=noise.vocab_limit,
    )


def gather_candidates(tp_group: Any, partial: torch.Tensor) -> torch.Tensor:
    """All-gathers every rank's Gumbel-max candidates, rank-major."""
    gathered = torch.empty(
        (tp_group.world_size * partial.shape[0], partial.shape[1]),
        dtype=partial.dtype,
        device=partial.device,
    )
    tp_group.all_gather_into_tensor(gathered, partial)
    return gathered


def padded_org_width(lm_head: torch.nn.Module) -> int:
    """Rows of the rank's original-vocabulary shard, padding included: equal on every
    TP rank, so their Gumbel-max candidates gather in one shape."""
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if isinstance(lm_head, VocabParallelEmbedding):
        return int(lm_head.shard_indices.num_org_elements_padded)
    return int(lm_head.weight.shape[0])


def sample_head(
    *,
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    noise: BenchHeadNoise,
    temperatures: torch.Tensor,
    offset: torch.Tensor,
    tp_group: Any,
    chunk_size: int = 256,
) -> torch.Tensor:
    """The plain DFLASH head's eager draft: the graph sampler's Gumbel-max per chunk,
    at per-row ``temperatures``, on the eager noise stream."""
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.speculative.dflash_utils import is_dense_head_weight

    tp_size = int(tp_group.world_size)
    # A GGUF head stores packed qweight and has no dense `weight`.
    quantized = not is_dense_head_weight(getattr(lm_head, "weight", None))
    if quantized and tp_size != 1:
        raise RuntimeError(
            "DFLASH with a quantized target lm_head is only supported at "
            f"tp=1, got tp_size={tp_size}."
        )
    vocab_start = (
        int(lm_head.shard_indices.org_vocab_start_index)
        if isinstance(lm_head, VocabParallelEmbedding)
        else 0
    )
    width = None if quantized else padded_org_width(lm_head)
    gather = None if tp_size == 1 else functools.partial(gather_candidates, tp_group)
    num_tokens = int(hidden_states.shape[0])
    out_tokens = torch.empty(
        (num_tokens,), dtype=torch.long, device=hidden_states.device
    )
    for start in range(0, num_tokens, chunk_size):
        hs = hidden_states[start : start + chunk_size]
        if quantized:
            logits = lm_head.quant_method.apply(lm_head, hs, None)
        else:
            logits = torch.matmul(hs.to(lm_head.weight.dtype), lm_head.weight[:width].T)
        out_tokens[start : start + hs.shape[0]] = gumbel_argmax(
            logits,
            noise=noise,
            temperatures=temperatures[start : start + hs.shape[0]],
            offset=offset,
            step=EAGER_STEP,
            vocab_start=vocab_start,
            gather=gather,
        )
        offset.add_(1)
    return out_tokens


def check_dflash_drafts(
    *,
    is_domino: bool,
    selector_sampling: Optional[bool],
    lilicorr_sampling: Optional[bool],
    lm_head: Optional[torch.nn.Module],
    is_npu: bool,
    runtime: Optional[object],
    log: bool,
    configured_token_mode: str,
) -> None:
    """Refuse the temperature for DFLASH drafts it cannot sample, and log what it
    changes. ``selector_sampling`` / ``lilicorr_sampling`` are None without that
    head; ``lm_head`` is the plain head's, None for the selector and LiLiCorr."""
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return
    if is_domino:
        raise ValueError(
            f"{ENV_NAME} does not support Domino drafts (a sequential argmax "
            f"rollout){opt_out_hint()}."
        )
    if selector_sampling is False:
        raise ValueError(
            f"{ENV_NAME} needs selector sampling, which this backend does not "
            f"support{opt_out_hint()}."
        )
    if lilicorr_sampling is False:
        raise ValueError(
            f"{ENV_NAME} with LiLiCorr requires SGLANG_ENABLE_LILICORR_SAMPLING=1, so "
            f"the draft graph captures its sampled walk{opt_out_hint()}."
        )
    if lm_head is not None:
        # The plain head's Gumbel-max: its NPU kernel does not take the bias-free
        # draw, and its noise covers token ids in quads, so every TP shard must
        # start on a multiple of 4 (the shard size is alike on all ranks).
        if is_npu:
            raise ValueError(
                f"{ENV_NAME} does not support the plain DFLASH head on NPU"
                f"{opt_out_hint()}."
            )
        rows = padded_org_width(lm_head)
        if rows % 4:
            raise ValueError(
                f"{ENV_NAME} needs the target lm_head's vocab shards in multiples of "
                f"4 rows, got {rows} per TP rank{opt_out_hint()}."
            )
    if not log:
        return
    token_mode = bench_token_mode(configured_token_mode)
    if token_mode != configured_token_mode:
        logger.warning(
            "%s overrides SGLANG_SIMULATE_ACC_TOKEN_MODE=%s: simulated acceptance "
            "commits the sampled drafts (real-draft-token).",
            ENV_NAME,
            configured_token_mode,
        )
    if runtime is not None:
        # TODO(jasperjiaguo): the megakernel's DFlash runtime should sample its
        # drafts at the benchmark temperature too; until then they stay greedy.
        logger.warning(
            "%s is not applied by the megakernel's DFlash runtime (%s) yet: the "
            "drafts it samples stay greedy.",
            ENV_NAME,
            type(runtime).__name__,
        )
        return
    logger.warning(
        "BENCHMARK: DFLASH greedy drafts sample at temperature %s, sampled requests "
        "at their own; acceptance is simulated (SGLANG_SIMULATE_ACC_LEN=%s, %s).",
        temperature,
        envs.SGLANG_SIMULATE_ACC_LEN.get(),
        token_mode,
    )
