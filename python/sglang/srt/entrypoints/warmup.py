from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput

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


# DeepSeek-V4 token-parallel prefill (SGLANG_DSV4_TOKEN_PARALLEL_PREFILL=1) runs
# only on eager EXTEND chunks of at least SGLANG_DSV4_TOKEN_PARALLEL_MIN_TOKENS
# rows, and the first chunk of each size class pays one-time costs (AITER MoE
# configs and FlyDSL builds for the pre-quantized MXFP8 input per token bucket,
# new GEMM / kernel shapes): -3 to -6 % on the first large batch after boot.
# This warm-up sends one prompt per tail size before the server takes traffic:
# a full first chunk (prefill graph) and then a tail chunk of that size
# (eager, token-parallel). On by default when token-parallel prefill is on.
_DSV4_TP_WARMUP_TAILS = (
    2048,
    2049,
    3072,
    4071,
    4096,
    6144,
    8175,
    8192,
    10240,
    12279,
    12288,
    14336,
    16383,
    16384,
)


def default_warmups() -> List[str]:
    """Warm-ups that run without --warmups (token-parallel prefill's, when it is on)."""
    if (
        envs.SGLANG_DSV4_TOKEN_PARALLEL_PREFILL.get()
        and envs.SGLANG_DSV4_TOKEN_PARALLEL_WARMUP.get()
    ):
        return ["dsv4_token_parallel"]
    return []


@warmup("dsv4_token_parallel")
async def dsv4_token_parallel(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    chunk = tokenizer_manager.server_args.chunked_prefill_size or 0
    min_rows = envs.SGLANG_DSV4_TOKEN_PARALLEL_MIN_TOKENS.get()
    context_len = tokenizer_manager.context_len
    if chunk <= 0:
        logger.info("dsv4 token-parallel warm-up skipped: chunked prefill is off")
        return
    tails = [
        t
        for t in _DSV4_TP_WARMUP_TAILS
        if min_rows <= t <= chunk and chunk + t < context_len
    ]
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for tail in tails:
        req = GenerateReqInput(
            input_ids=rng.integers(1000, 100000, size=chunk + tail).tolist(),
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info(
        "dsv4 token-parallel warm-up: %d prompts (first chunk %d + tails %s) in %.1f s",
        len(tails),
        chunk,
        tails,
        time.perf_counter() - t0,
    )
