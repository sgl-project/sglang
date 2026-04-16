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
    """Pre-compile the xgrammar FSM for Whisper auto-detect regex.

    The first request that uses the structured generation regex incurs a
    ~15-20s compilation cost.  Running it at startup avoids a latency
    spike on the first real request.
    """
    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        WHISPER_AUTODETECT_REGEX,
    )

    logger.info(
        "Compiling Whisper auto-detect regex FSM (one-time, ~15-20s)..."
    )
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono)
    import base64, io, struct

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    # Minimal WAV: RIFF header + fmt chunk + data chunk with silence
    data = struct.pack("<h", 0) * n
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    req = GenerateReqInput(
        text="",
        audio_data=f"data:audio/wav;base64,{audio_b64}",
        sampling_params={
            "max_new_tokens": 4,
            "temperature": 0,
            "regex": WHISPER_AUTODETECT_REGEX,
            "skip_special_tokens": False,
            "_detect_language": True,
        },
        modalities=["audio"],
    )
    await tokenizer_manager.generate_request(req, None).__anext__()
    logger.info("Whisper auto-detect regex FSM compiled.")


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
