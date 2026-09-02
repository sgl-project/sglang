from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
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


@warmup("voice_chat")
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


# Concurrent requests per serving_coverage cohort on one DP rank. The cohort
# exists to drain through every batch size, so it tracks max_running_requests,
# capped so the warmup stays bounded on deployments that allow hundreds of
# running requests.
_SERVING_COVERAGE_MAX_COHORT = 8

# With data parallelism every phase runs once per DP rank (each rank is its own
# scheduler with its own kernels). Ranks run concurrently in windows of this
# size so tokenizer-side in-flight requests stay bounded at
# window * cohort even on wide deployments.
_SERVING_COVERAGE_MAX_CONCURRENT_DP_RANKS = 8

# Longest decode any phase asks for; the prompt budget reserves room for it.
_SERVING_COVERAGE_MAX_DECODE = 320

_SERVING_COVERAGE_TEXT_PROMPTS = [
    "The history of the Roman Empire begins with the founding of",
    "In Python, a list comprehension is a compact way to",
    "Photosynthesis is the process by which plants convert",
    "To make a simple tomato sauce, start by heating olive oil and",
    "The three laws of motion formulated by Newton describe how",
    "A binary search tree is a data structure in which each node",
    "The water cycle describes how water evaporates from the surface,",
    "Machine learning models are trained by adjusting parameters to",
]

_SERVING_COVERAGE_SAMPLING_PROMPTS = [
    "Write two sentences about the ocean at night.",
    "List three uses for a paperclip.",
    "Describe a quiet morning in a small town.",
    "Give a short tip for learning a new language.",
]

# Non-greedy sampling paths: top-p, top-k, min-p and a penalizer.
_SERVING_COVERAGE_SAMPLING_VARIANTS = [
    {"temperature": 0.8, "top_p": 0.95},
    {"temperature": 1.0, "top_p": 0.9, "top_k": 40},
    {"temperature": 0.7, "top_k": 20, "min_p": 0.05},
    {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.05},
]


def _serving_coverage_budget(max_req_input_len: Optional[int]):
    """Prompt and decode token budgets for one request, ``(max_prompt, max_decode)``.

    ``max_req_input_len`` is the engine's input limit; prompt plus decode
    never exceed it: the prompt budget reserves the decode budget and, when
    the limit leaves room, 64 tokens of headroom for the chat template. Any
    positive limit clamps; a short-context model gets shorter prompts and
    shorter decodes, never unclamped requests. ``None`` (limit unknown)
    disables clamping; the real ``TokenizerManager`` always publishes the
    limit.
    """
    limit = int(max_req_input_len or 0)
    if limit <= 0:
        return None, _SERVING_COVERAGE_MAX_DECODE
    if limit >= 1024:
        max_decode = _SERVING_COVERAGE_MAX_DECODE
    else:
        max_decode = max(limit // 4, 1)
    headroom = 64 if limit >= 256 else 0
    max_prompt = max(limit - max_decode - headroom, 1)
    return max_prompt, max_decode


async def _gather_all(coros):
    """Run every coroutine to completion, then raise the first failure.

    ``asyncio.gather`` alone re-raises the first exception while its siblings
    keep running; a phase that "failed" would then return with requests still
    in flight, and anything armed after the phase (the serving-started mark)
    would see their kernel loads as late. Waiting for all of them keeps a
    phase's completion meaningful.
    """
    import asyncio

    results = await asyncio.gather(*coros, return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        if len(failures) > 1:
            logger.debug(
                "serving_coverage: %d of %d requests failed; first: %r",
                len(failures),
                len(results),
                failures[0],
            )
        raise failures[0]


def _serving_coverage_phases(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Build the serving_coverage phases as ``(name, coroutine function)`` pairs.

    Every phase issues genuine requests through ``generate_request`` and
    drains them, so the specializations loaded are exactly the ones serving
    will use. Phase order matters: the greedy phases come first so the
    sampling phase only adds the sampling kernels on top of already-loaded
    decode shapes. With ``dp_size > 1`` each phase body runs once per DP
    rank (``routed_dp_rank``), ranks concurrently in bounded windows, so
    every rank sees the same shapes a single-rank deployment would.
    """
    import itertools

    from sglang.srt.runtime_context import get_parallel, get_schedule

    schedule = get_schedule()
    cap = _SERVING_COVERAGE_MAX_COHORT
    max_running = max(1, min(int(schedule.max_running_requests or cap), cap))
    chunk = int(schedule.chunked_prefill_size or 0)
    if chunk <= 0:
        chunk = 4096
    dp_size = int(get_parallel().dp_size or 1)
    ranks = list(range(dp_size)) if dp_size > 1 else [None]

    max_prompt, max_decode = _serving_coverage_budget(
        getattr(tokenizer_manager, "max_req_input_len", None)
        or getattr(tokenizer_manager, "context_len", None)
    )
    if max_prompt is None:
        logger.warning(
            "serving_coverage: input limit unknown; prompt lengths are not clamped"
        )
    vocab_size = getattr(
        getattr(tokenizer_manager, "model_config", None), "vocab_size", None
    )
    id_high = min(2**16, int(vocab_size)) if vocab_size else 2**16
    # Text prompts need a tokenizer; --skip-tokenizer-init leaves none. They
    # are not clamped (the template adds tokens the warmup cannot count), so
    # they also need a prompt budget that fits a short sentence plus template.
    has_tokenizer = getattr(tokenizer_manager, "tokenizer", None) is not None
    text_ok = has_tokenizer and (max_prompt is None or max_prompt >= 32)
    if has_tokenizer and not text_ok:
        logger.info(
            "serving_coverage: skipping text and image phases: the prompt "
            "budget (%d tokens) is too small for a text prompt",
            max_prompt,
        )
    rooms = itertools.count()

    def _ids(n):
        n = max(n, 1)
        if max_prompt is not None:
            n = min(n, max_prompt)
        return np.random.randint(id_high, size=[n]).tolist()

    def _request(rank, input_ids=None, text=None, image_data=None, max_new_tokens=48):
        req = GenerateReqInput(
            input_ids=input_ids,
            text=text,
            image_data=image_data,
            sampling_params={
                "max_new_tokens": min(max_new_tokens, max_decode),
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        if rank is not None:
            req.routed_dp_rank = rank
        if disaggregation_mode != "null":
            # Distinct rooms: cohort requests are in flight concurrently.
            req.bootstrap_room = next(rooms)
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        return req

    async def _run(req):
        async for _ in tokenizer_manager.generate_request(req, None):
            pass

    async def _gather(reqs):
        await _gather_all(_run(r) for r in reqs)

    def _per_rank(body):
        """Wrap ``body(rank)`` so the phase covers every DP rank."""

        async def phase():
            window = _SERVING_COVERAGE_MAX_CONCURRENT_DP_RANKS
            for start in range(0, len(ranks), window):
                await _gather_all(body(r) for r in ranks[start : start + window])

        return phase

    def _fits(tokens):
        return max_prompt is None or tokens <= max_prompt

    phases = []

    async def _cohort(rank):
        # Staggered decode lengths so the requests finish at different steps
        # and the running batch shrinks one request at a time; staggered
        # prompt lengths straddle the chunk boundary so chunked prefills
        # coexist with decode.
        await _gather(
            _request(
                rank,
                input_ids=_ids(chunk // 2 + (chunk // max_running) * i + 64 * i),
                max_new_tokens=32 + 40 * i,
            )
            for i in range(max_running)
        )

    phases.append(("cohort", _per_rank(_cohort)))

    if _fits(chunk + 64):

        async def _cold_cohort(rank):
            # Every request longer than one chunk, so the number of concurrent
            # multi-chunk extends reaches max_running.
            await _gather(
                _request(
                    rank,
                    input_ids=_ids(chunk + 64 + 256 * i),
                    max_new_tokens=16 + 24 * i,
                )
                for i in range(max_running)
            )

        phases.append(("cold-cohort", _per_rank(_cold_cohort)))
    else:
        logger.info(
            "serving_coverage: skipping cold-cohort phase: prompts are clamped "
            "to %d tokens, which cannot exceed one chunked-prefill chunk (%d)",
            max_prompt,
            chunk,
        )

    chunk_lengths = [
        tokens
        for tokens in (chunk, chunk + 64, 2 * chunk, 2 * chunk + 64, 3 * chunk + 64)
        if _fits(tokens)
    ]
    if chunk_lengths:

        async def _chunk_multiples(rank):
            # Exact chunk multiples and one-block-past-a-multiple, alone.
            for tokens in chunk_lengths:
                await _run(_request(rank, input_ids=_ids(tokens), max_new_tokens=4))

        phases.append(("chunk-multiples", _per_rank(_chunk_multiples)))
    else:
        logger.info(
            "serving_coverage: skipping chunk-multiples phase: prompts are "
            "clamped to %d tokens, below one chunked-prefill chunk (%d)",
            max_prompt,
            chunk,
        )

    async def _short_prompts(rank):
        # Chat-turn sized prompts, alone and as a cohort, plus one long decode.
        for tokens in (16, 48, 128, 320, 768):
            await _run(_request(rank, input_ids=_ids(tokens), max_new_tokens=16))
        await _gather(
            _request(rank, input_ids=_ids(24 + 40 * i), max_new_tokens=24 + 16 * i)
            for i in range(max_running)
        )
        await _run(_request(rank, input_ids=_ids(32), max_new_tokens=256))

    phases.append(("short-prompts", _per_rank(_short_prompts)))

    if text_ok:

        async def _natural_text(rank):
            # Random token ids rarely yield accepted speculative drafts, so
            # the verify shapes that accepted drafts produce are left to real
            # text; natural prompts exercise the drafter on text it can
            # predict.
            prompts = _SERVING_COVERAGE_TEXT_PROMPTS
            for text in prompts[:2]:
                await _run(_request(rank, text=text, max_new_tokens=192))
            await _gather(
                _request(
                    rank,
                    text=prompts[(2 + i) % len(prompts)],
                    max_new_tokens=96 + 32 * i,
                )
                for i in range(max_running)
            )

        phases.append(("natural-text", _per_rank(_natural_text)))

    async def _sampling(rank):
        # The phases above decode greedily; the sampling kernels are built or
        # loaded on the first non-greedy request otherwise.
        prompts = _SERVING_COVERAGE_SAMPLING_PROMPTS
        variants = _SERVING_COVERAGE_SAMPLING_VARIANTS

        def _sampled(i, max_new_tokens):
            if text_ok:
                req = _request(
                    rank, text=prompts[i % len(prompts)], max_new_tokens=max_new_tokens
                )
            else:
                req = _request(
                    rank, input_ids=_ids(32 + 8 * i), max_new_tokens=max_new_tokens
                )
            req.sampling_params.update(variants[i % len(variants)])
            return req

        for i in range(len(variants)):
            await _run(_sampled(i, 48))
        await _gather(_sampled(i, 40 + 8 * i) for i in range(max_running))

    phases.append(("sampling", _per_rank(_sampling)))

    mm_tokens = getattr(
        getattr(tokenizer_manager, "mm_processor", None), "mm_tokens", None
    )
    image_token = getattr(mm_tokens, "image_token", None)
    if isinstance(image_token, list):
        image_token = image_token[0] if image_token else None
    if image_token and text_ok:

        async def _image(rank):
            import base64
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (128, 64, 32)).save(buf, format="PNG")
            data_url = (
                "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            )
            await _run(
                _request(
                    rank,
                    text=f"{image_token} Describe the image.",
                    image_data=[data_url],
                    max_new_tokens=8,
                )
            )

        phases.append(("image", _per_rank(_image)))

    return phases


@warmup("serving_coverage")
async def serving_coverage(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Drive the real serving path before ready so lazy kernel loads land now.

    Triton loads a specialization's cubin at first launch, outside the torch
    allocator; on a deployment whose pools hold nearly all device memory a
    first use minutes into serving can fail in ``cuModuleLoadData``.
    Hand-rolled prewarms miss real specializations, so this warmup issues
    genuine requests instead (see ``_serving_coverage_phases``):

    * a cohort of up to ``max_running_requests`` concurrent requests with
      staggered lengths, and a cold cohort with every prompt longer than one
      chunked-prefill chunk,
    * single prefills at exact chunk multiples,
    * short prompts and one long decode,
    * natural-text prompts (accepted speculative drafts widen verify batches
      in ways random token ids never do),
    * non-greedy sampling variants (the sampling kernels are otherwise built
      on the first sampled request),
    * one image request when the model is multimodal.

    Prompt and decode lengths are clamped to the engine's input limit; phases
    whose shapes cannot fit are skipped with a log line. With data
    parallelism every phase runs once per DP rank. Each phase is best-effort:
    a failure logs, its requests are still drained, and serving proceeds.
    """
    for name, fn in _serving_coverage_phases(disaggregation_mode, tokenizer_manager):
        try:
            await fn()
            logger.info("serving_coverage warmup phase done: %s", name)
        except Exception:
            logger.exception("serving_coverage warmup phase failed: %s", name)


@warmup("prefill_shapes")
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
