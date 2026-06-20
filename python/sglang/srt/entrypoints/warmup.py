from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

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


@warmup("kda_cache")
async def kda_cache(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Pre-populate MambaRadixCache with system prompts for KDA hybrid models.

    Sends each prompt through the full prefill pipeline so that KDA recurrent
    states and full-attention KV are cached in MambaRadixCache. Subsequent
    requests with the same prefix will hit the cache and skip recomputation.
    """
    import json

    server_args = tokenizer_manager.server_args
    prompts_file = server_args.warmup_prompts_file
    if prompts_file is None:
        logger.warning(
            "kda_cache warmup enabled but --warmup-prompts-file not specified, skipping"
        )
        return

    try:
        with open(prompts_file) as f:
            prompts = json.load(f)
    except FileNotFoundError:
        logger.error(f"Warmup prompts file not found: {prompts_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from warmup prompts file: {prompts_file}: {e}")
        return

    if not isinstance(prompts, list) or len(prompts) == 0:
        logger.warning("warmup prompts file is empty or not a list, skipping")
        return

    logger.info(f"KDA cache warmup: processing {len(prompts)} prompts")
    for i, prompt in enumerate(prompts):
        # Support both text strings and token ID lists
        if isinstance(prompt, str):
            req = GenerateReqInput(
                text=prompt,
                sampling_params={"max_new_tokens": 1, "temperature": 0},
            )
        elif isinstance(prompt, list):
            req = GenerateReqInput(
                input_ids=prompt,
                sampling_params={"max_new_tokens": 1, "temperature": 0},
            )
        else:
            logger.warning(
                f"Skipping warmup prompt {i}: unsupported type {type(prompt)}"
            )
            continue

        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST

        # Drain the generator to ensure full processing and caching
        async for _ in tokenizer_manager.generate_request(req, None):
            pass

        logger.info(f"KDA cache warmup: cached prompt {i + 1}/{len(prompts)}")

    logger.info("KDA cache warmup completed")


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
