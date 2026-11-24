from __future__ import annotations

import asyncio
import itertools
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import msgspec
import numpy as np
import tqdm

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}
# Warmups that only send generate requests, which Rust-server mode runs through
# its listener (`rust_server.warmups`); the others need the TokenizerManager.
_rust_server_warmups = set()


def warmup(name: str, *, rust_server: bool = False):
    def decorator(fn):
        _warmup_registry[name] = fn
        if rust_server:
            _rust_server_warmups.add(name)
        else:
            _rust_server_warmups.discard(name)
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat", rust_server=True)
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("prefill_shapes", rust_server=True)
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    sizes = sorted(sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


# Prompt tokens per round: more than one prefill chunk at common chunk sizes,
# so every round also splits a prompt across a chunk boundary.
_CONCURRENT_PREFILL_TOKENS = 40000
_CONCURRENT_PREFILL_NEW_TOKENS = 8
# Input-logprob probes over an unaligned and a 16-aligned row count.
_CONCURRENT_PREFILL_LOGPROB_SIZES = (513, 512)


class ConcurrentPrefillLimits(msgspec.Struct, frozen=True, kw_only=True):
    """The scheduler's effective limits that the warmup plan must stay inside."""

    # Effective running-request slots, summed over DP ranks.
    max_running_requests: int
    # Per scheduler; None is unlimited.
    max_queued_requests: Optional[int]
    # None or <= 0: chunked prefill is off.
    chunked_prefill_size: Optional[int]
    context_len: int
    max_req_input_len: int
    # Speculative slots the tokenizer adds to each prompt's length.
    num_reserved_tokens: int
    # Smallest full-attention KV pool over DP ranks, in tokens.
    kv_pool_tokens: int
    page_size: int
    serves_input_logprobs: bool


class ConcurrentPrefillPlan(msgspec.Struct, frozen=True, kw_only=True):
    rounds: List[List[int]]
    logprob_sizes: List[int]


def concurrent_prefill_limits(
    internal_states: List[Dict[str, Any]],
    *,
    context_len: int,
    max_req_input_len: int,
    num_reserved_tokens: int,
) -> ConcurrentPrefillLimits:
    """Collect the limits from the schedulers' `get_internal_state` readback
    (one dict per DP rank)."""
    state = internal_states[0]
    spec_algorithm = SpeculativeAlgorithm.from_string(state["speculative_algorithm"])
    return ConcurrentPrefillLimits(
        max_running_requests=sum(
            s["effective_max_running_requests_per_dp"] for s in internal_states
        ),
        max_queued_requests=state["max_queued_requests"],
        chunked_prefill_size=state["chunked_prefill_size"],
        context_len=context_len,
        max_req_input_len=max_req_input_len,
        num_reserved_tokens=num_reserved_tokens,
        kv_pool_tokens=min(
            s["memory_usage"]["token_capacity"] for s in internal_states
        ),
        page_size=state["page_size"] or 1,
        # UNO refuses logprob requests; the MLX sampling path refuses prompt logprobs.
        serves_input_logprobs=not (
            spec_algorithm.is_uno() or state["mlx_enable_sampling"]
        ),
    )


def concurrent_prefill_plan(limits: ConcurrentPrefillLimits) -> ConcurrentPrefillPlan:
    """Prompt lengths of each concurrent round (1, about 1/8 and all of the
    running slots) and of the input-logprob probes.

    A round sends at least one round budget (two prefill chunks, at least
    `_CONCURRENT_PREFILL_TOKENS`) and less than two: each prompt gets an equal
    share plus an offset below that share, stepped by 37 so extends do not align.
    """
    page = limits.page_size
    new_tokens = _CONCURRENT_PREFILL_NEW_TOKENS
    max_prompt = min(
        # The scheduler aborts a prompt of max_req_input_len tokens or more.
        limits.max_req_input_len - 1,
        # The tokenizer counts the reserved slots and the new tokens in the context.
        limits.context_len - limits.num_reserved_tokens - new_tokens - 1,
        # Admission rounds the prompt up to whole pages and reserves one more page.
        (limits.kv_pool_tokens - limits.num_reserved_tokens - new_tokens) // page * page
        - 2 * page,
    )
    slots = limits.max_running_requests
    if limits.max_queued_requests is not None:
        # A round arriving in one scheduler poll must fit in the waiting queue.
        slots = min(slots, limits.max_queued_requests)
    if max_prompt < 1 or slots < 1:
        return ConcurrentPrefillPlan(rounds=[], logprob_sizes=[])

    chunk = limits.chunked_prefill_size
    budget = _CONCURRENT_PREFILL_TOKENS
    if chunk is not None and chunk > 0:
        budget = max(budget, 2 * chunk)
    rounds = []
    for n in sorted({1, slots // 8 or 1, slots}):
        share = max(1, budget // n)
        rounds.append([min(max_prompt, share + 37 * i % share + 1) for i in range(n)])
    logprob_sizes = []
    if limits.serves_input_logprobs:
        logprob_sizes = [min(max_prompt, s) for s in _CONCURRENT_PREFILL_LOGPROB_SIZES]
    return ConcurrentPrefillPlan(rounds=rounds, logprob_sizes=logprob_sizes)


@warmup("concurrent_prefill")
async def concurrent_prefill(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Warm up multi-sequence prefill, mixed steps and input logprobs, so their
    first JIT compiles happen before serving."""
    if disaggregation_mode == "decode" or not tokenizer_manager.is_generation:
        logger.info(
            "Skipping concurrent_prefill: this server runs no generation prefill"
        )
        return
    vocab = min(2**16, tokenizer_manager.model_config.vocab_size)
    # PD prefill servers need a bootstrap room; concurrent requests need distinct ones.
    bootstrap_rooms = itertools.count()

    async def one(size, **kwargs):
        req = GenerateReqInput(
            input_ids=np.random.randint(vocab, size=size).tolist(),
            sampling_params={
                "max_new_tokens": _CONCURRENT_PREFILL_NEW_TOKENS,
                "temperature": 0,
            },
            **kwargs,
        )
        if disaggregation_mode != "null":
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
            req.bootstrap_room = next(bootstrap_rooms)
        await tokenizer_manager.generate_request(req, None).__anext__()

    # An opt-in warmup must not stop the server from starting; the default
    # warmup that follows still checks that the server serves.
    try:
        limits = concurrent_prefill_limits(
            await tokenizer_manager.get_internal_state(),
            context_len=tokenizer_manager.context_len,
            max_req_input_len=tokenizer_manager.max_req_input_len,
            num_reserved_tokens=tokenizer_manager.num_reserved_tokens,
        )
        plan = concurrent_prefill_plan(limits)
        logger.info(
            f"concurrent_prefill: {limits=}, round sizes {[len(r) for r in plan.rounds]}"
        )
        for sizes in plan.rounds:
            await asyncio.gather(*(one(size) for size in sizes))
        for size in plan.logprob_sizes:
            await one(size, return_logprob=True, logprob_start_len=0)
    except Exception as e:
        logger.warning(f"concurrent_prefill warmup stopped early: {e!r}")
