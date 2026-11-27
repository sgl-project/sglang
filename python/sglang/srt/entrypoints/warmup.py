from __future__ import annotations

import asyncio
import fcntl
import itertools
import logging
import os
import random
import tempfile
import time
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

    # Effective running-request slots of one DP rank (the smallest).
    max_running_requests: int
    # Rounds are sized per DP rank and rely on an even dispatch.
    dp_ranks: int = 1
    even_dp_dispatch: bool = True
    # Per scheduler; None is unlimited or not applied.
    max_queued_requests: Optional[int]
    # Tokenizer workers of the server; with a queue limit only one of them
    # sends this warmup (`_claim_tokenizer_worker_warmup`).
    tokenizer_workers: int = 1
    # The scheduler's own chunk size; None when chunked prefill is off.
    chunked_prefill_size: Optional[int]
    # Prefill tokens the scheduler packs into one forward. Without chunking it
    # still admits one longer prompt alone, past what the forward is sized for.
    max_prefill_tokens: int
    context_len: int
    max_req_input_len: int
    # Speculative slots the tokenizer adds to each prompt's length.
    num_reserved_tokens: int
    # Smallest full-attention KV pool over DP ranks, in tokens.
    kv_pool_tokens: int
    # Any sliding-window or per-request ring pool besides the full one.
    has_window_pool: bool = False
    page_size: int
    serves_input_logprobs: bool


class ConcurrentPrefillPlan(msgspec.Struct, frozen=True, kw_only=True):
    rounds: List[List[int]]
    logprob_sizes: List[int]
    # Set when the warmup cannot be shown to be safe; nothing is sent then.
    skip_reason: Optional[str] = None


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
    pd_prefill = state.get("disaggregation_mode") == "prefill"
    load_balance = state["load_balance_method"]
    return ConcurrentPrefillLimits(
        max_running_requests=min(
            s["effective_max_running_requests_per_dp"] for s in internal_states
        ),
        dp_ranks=len(internal_states),
        # Round robin gives each rank an equal share of a round, and so do the
        # consecutive bootstrap rooms this warmup sends to a PD prefill server.
        even_dp_dispatch=len(internal_states) == 1
        or load_balance == "round_robin"
        or (pd_prefill and load_balance == "follow_bootstrap_room"),
        # Only the regular scheduler applies the queue limit; a PD prefill
        # server queues requests for bootstrap instead.
        max_queued_requests=None if pd_prefill else state["max_queued_requests"],
        tokenizer_workers=state["tokenizer_worker_num"],
        chunked_prefill_size=state["effective_chunked_prefill_size"],
        max_prefill_tokens=state["max_prefill_tokens"],
        context_len=context_len,
        max_req_input_len=max_req_input_len,
        num_reserved_tokens=num_reserved_tokens,
        kv_pool_tokens=min(
            s["memory_usage"]["token_capacity"] for s in internal_states
        ),
        has_window_pool=any(
            s["memory_usage"]["token_capacity_swa"] is not None
            or bool(s["memory_usage"]["token_capacity_window"])
            for s in internal_states
        ),
        page_size=state["page_size"] or 1,
        # UNO refuses logprob requests; the MLX sampling path refuses prompt logprobs.
        serves_input_logprobs=not (
            spec_algorithm.is_uno() or state["mlx_enable_sampling"]
        ),
    )


def concurrent_prefill_plan(limits: ConcurrentPrefillLimits) -> ConcurrentPrefillPlan:
    """Prompt lengths of each concurrent round (1, about 1/8 and all of the
    running slots of every DP rank) and of the input-logprob probes.

    The plan only covers configs where it can show that every request is
    admitted; for any other config it sends nothing and says why.

    Each DP rank gets at least one round budget (two prefill chunks, at least
    `_CONCURRENT_PREFILL_TOKENS`) per round and less than two: each prompt gets
    an equal share plus an offset below that share, stepped by 37 so extends do
    not align.
    """
    page = limits.page_size
    new_tokens = _CONCURRENT_PREFILL_NEW_TOKENS
    chunk = limits.chunked_prefill_size
    queue = limits.max_queued_requests
    max_prompt = min(
        # The scheduler aborts a prompt of max_req_input_len tokens or more.
        limits.max_req_input_len - 1,
        # The tokenizer counts the reserved slots and the new tokens in the context.
        limits.context_len - limits.num_reserved_tokens - new_tokens - 1,
        # Admission rounds the prompt up to whole pages and reserves one more page.
        (limits.kv_pool_tokens - limits.num_reserved_tokens - new_tokens) // page * page
        - 2 * page,
    )
    if chunk is None:
        # Never rely on the lone-long-prompt exception: backends size their
        # per-forward buffers to this budget (the Ascend fused MoE prefill
        # takes at most min(max_prefill_tokens, 4096) rows).
        max_prompt = min(max_prompt, limits.max_prefill_tokens)
    skip_reason = None
    if chunk is None and limits.has_window_pool:
        # Without chunking a prompt must fit every pool in one piece; a
        # full-attention-only model is covered by max_prompt.
        skip_reason = "chunked prefill is off and the model has sliding-window pools"
    elif not limits.even_dp_dispatch and queue is not None:
        # Without a queue limit an uneven share only waits in the queue.
        skip_reason = "DP dispatch is load based, so a rank's queue share is unknown"
    elif queue is not None and queue < 1:
        skip_reason = "the request queue limit is 0"
    elif max_prompt < 1:
        skip_reason = "the context or the KV pool is too small"
    if skip_reason is not None:
        return ConcurrentPrefillPlan(
            rounds=[], logprob_sizes=[], skip_reason=skip_reason
        )

    slots = limits.max_running_requests
    if queue is not None:
        # A rank's share of a round can arrive in one scheduler poll.
        slots = min(slots, queue)
    budget = max(_CONCURRENT_PREFILL_TOKENS, 2 * (chunk or 0))
    rounds = []
    for n in sorted({1, slots // 8 or 1, slots}):
        share = max(1, budget // n)
        rounds.append(
            [
                min(max_prompt, share + 37 * i % share + 1)
                for i in range(n * limits.dp_ranks)
            ]
        )
    logprob_sizes = []
    if limits.serves_input_logprobs:
        logprob_sizes = [min(max_prompt, s) for s in _CONCURRENT_PREFILL_LOGPROB_SIZES]
    return ConcurrentPrefillPlan(rounds=rounds, logprob_sizes=logprob_sizes)


# Held until the worker exits, so no other tokenizer worker of this server runs
# the warmup after this one finishes.
_tokenizer_worker_warmup_lock: Optional[int] = None
# How long a worker waits for the elected worker's warmup before going on.
_TOKENIZER_WORKER_WAIT_S = 1800.0
_TOKENIZER_WORKER_POLL_S = 0.1


def _tokenizer_worker_lock_path(suffix: str = "") -> str:
    """A lock file shared by the tokenizer workers of this server only: named
    after its main process, the same id the workers use for their shared args."""
    from sglang.srt.managers.multi_tokenizer_mixin import get_main_process_id

    return os.path.join(
        tempfile.gettempdir(),
        f"sglang_concurrent_prefill_{get_main_process_id()}{suffix}.lock",
    )


def _try_lock(path: str) -> Optional[int]:
    """A file descriptor holding an exclusive lock on `path`, or None when
    another process holds it."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


async def _wait_for_lock(path: str, timeout: float) -> Optional[int]:
    """`_try_lock` retried without blocking the event loop; None once
    `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while (fd := _try_lock(path)) is None:
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(_TOKENIZER_WORKER_POLL_S)
    return fd


def _claim_tokenizer_worker_warmup() -> bool:
    """True in exactly one tokenizer worker of this server: the first to lock
    the server's lock file. The lock goes away with the worker, so a later
    server reusing the process id can claim it again."""
    global _tokenizer_worker_warmup_lock
    if _tokenizer_worker_warmup_lock is None:
        _tokenizer_worker_warmup_lock = _try_lock(_tokenizer_worker_lock_path())
    return _tokenizer_worker_warmup_lock is not None


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
    # PD prefill servers need a bootstrap room; concurrent requests need
    # distinct ones, also across tokenizer workers. Consecutive rooms keep
    # follow_bootstrap_room DP dispatch even.
    bootstrap_rooms = itertools.count(random.randrange(1 << 62))

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
        if plan.skip_reason is not None:
            logger.info(f"Skipping concurrent_prefill: {plan.skip_reason}")
            return
        turn = None
        try:
            if limits.max_queued_requests is not None and limits.tokenizer_workers > 1:
                # The plan bounds one sender's requests per rank queue, so one
                # worker sends it. The workers take this lock in turn and the
                # elected one keeps it until its warmup ends (or it exits), so
                # no worker serves or sends its default warmup before then.
                turn = await _wait_for_lock(
                    _tokenizer_worker_lock_path(".turn"), _TOKENIZER_WORKER_WAIT_S
                )
                if not _claim_tokenizer_worker_warmup():
                    if turn is None:
                        logger.warning(
                            "concurrent_prefill: another tokenizer worker's warmup "
                            f"is still running after {_TOKENIZER_WORKER_WAIT_S:.0f}s"
                        )
                    logger.info(
                        "Skipping concurrent_prefill: another tokenizer worker runs "
                        "it under the request queue limit"
                    )
                    return
            logger.info(
                f"concurrent_prefill: {limits=}, round sizes {[len(r) for r in plan.rounds]}"
            )
            for sizes in plan.rounds:
                await asyncio.gather(*(one(size) for size in sizes))
            for size in plan.logprob_sizes:
                await one(size, return_logprob=True, logprob_start_len=0)
        finally:
            if turn is not None:
                os.close(turn)
    except Exception as e:
        logger.warning(f"concurrent_prefill warmup stopped early: {e!r}")
