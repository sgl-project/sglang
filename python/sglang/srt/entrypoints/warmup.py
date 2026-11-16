from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
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


@warmup("system_prefix_seed")
async def system_prefix_seed(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Seed the radix tree with the fixed system prefix's mamba boundary state.

    OPT-IN ONLY (--warmups system_prefix_seed); the server does NOT run this
    automatically. The seed renders with the template's DEFAULT kwargs,
    which need not match real traffic: a template may emit different
    system-block tokens per chat_template_kwargs (e.g. Qwen3.8 injects a
    reasoning-effort preamble unless thinking is disabled), and a mismatched
    seed pins a boundary state that no request ever hits while falsely
    disengaging the cold-start gate (observed in production: the whole
    workload ran cold at 3 mamba slots). Default operation is lazy: the
    first real request cold-seeds the tree on its own token path, serialized
    by the cold-start gate, which disengages on the first actual warm hit.
    Only use this warmup when the traffic is known to render with the
    template defaults.

    Runs one text-only chat through the fixed system prefix to completion:
    its finish-time donate is capped at the chunk-aligned boundary
    (cap_mamba_donate_len) and lands on the shared boundary node, which the
    insert path then permanently pins. Multimodal traffic shares the
    system block too (image hash pad tokens only diverge at the first
    image), so a text-only seed serves multimodal requests as well -- as long
    as the render kwargs match.
    """
    from sglang.srt.system_prompt.system_prompt import resolve_system_prompt
    from sglang.srt.system_prompt.token_seam import SystemPrefix

    server_args = tokenizer_manager.server_args
    system_text = resolve_system_prompt(server_args)
    if system_text is None:
        # Feature off (the startup probe may have disabled it after the
        # launch-time flag check) -- nothing to seed.
        return

    # Mirror serving_chat's encode-kwarg detection so the seam here produces
    # the same token stream as the serving path.
    try:
        auto_adds_specials = len(tokenizer_manager.tokenizer.encode("")) > 0
    except Exception:
        auto_adds_specials = True
    encode_kwargs = {"add_special_tokens": False} if auto_adds_specials else {}

    prefix = SystemPrefix(tokenizer_manager.tokenizer, system_text)
    prefix.prepare(server_args.mamba_cache_chunk_size, encode_kwargs)
    if not prefix.enabled:
        logger.warning(
            "system_prefix_seed warmup skipped: the seam probe rejected the "
            "fixed prefix; the first requests will cold-start the boundary "
            "state"
        )
        return

    input_ids = prefix.split_request(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "hi"},
        ],
        encode_kwargs,
    )
    if input_ids is None:
        logger.warning(
            "system_prefix_seed warmup skipped: split_request fell back to "
            "the full-render path; the first requests will cold-start the "
            "boundary state"
        )
        return

    generate_req_input = GenerateReqInput(
        # dp_size copies so every dp rank's tree can get seeded (cache-aware
        # routing may still co-locate them; any rank that misses out simply
        # cold-starts its first request, whose finish then pins it).
        input_ids=(
            input_ids if server_args.dp_size == 1 else [input_ids] * server_args.dp_size
        ),
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        system_prefix_boundary=prefix.boundary_len,
    )
    if disaggregation_mode != "null":
        generate_req_input.bootstrap_room = 0
        generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

    # Drain the generator so the request finishes server-side and the
    # finish-time donate + pin actually run.
    async for _ in tokenizer_manager.generate_request(generate_req_input, None):
        pass
    logger.info(
        "system_prefix_seed warmup done: %d-token shared prefix cached and "
        "pinned in the radix tree",
        prefix.boundary_len,
    )
