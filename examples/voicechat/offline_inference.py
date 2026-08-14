#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct offline WAV inference for NVIDIA NemotronLabs VoiceChat."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from examples.voicechat.client import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    INPUT_RATE,
    OUTPUT_RATE,
    complete_frames,
    read_wav,
    resample_pcm16,
)
from examples.voicechat.online_server import (
    VoiceChatRuntime,
    add_runtime_arguments,
    validate_runtime_arguments,
)
from examples.voicechat.online_session import AsyncSGLangVoiceChatSession

PIPELINE_QUEUE_SIZE = 4

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


@dataclass
class OfflineVoiceChatResult:
    text: str
    function_text: str
    text_tokens: list[int]
    function_tokens: list[int]
    audio_pcm16: bytes
    frames: int


def _decode_tokens(tokenizer, tokens: list[int], pad_token_id: int) -> str:
    meaningful = [token for token in tokens if token != pad_token_id]
    return tokenizer.decode(meaningful, skip_special_tokens=True).strip()


def _model_input(input_pcm16: bytes, trailing_silence: float) -> bytes:
    if trailing_silence < 0:
        raise ValueError("trailing_silence cannot be negative")
    silence_frames = math.ceil(trailing_silence * INPUT_RATE / FRAME_SAMPLES)
    return complete_frames(input_pcm16) + bytes(silence_frames * FRAME_BYTES)


def _decode_sidecar_audio(decoded: dict) -> bytes:
    if decoded["sample_rate"] != OUTPUT_RATE:
        raise ValueError(
            "Audio sidecar returned sample rate "
            f"{decoded['sample_rate']}; expected {OUTPUT_RATE}."
        )
    pcm = base64.b64decode(decoded["pcm16"], validate=True)
    expected_bytes = int(decoded["samples"]) * 2
    if len(pcm) != expected_bytes:
        raise ValueError("Audio sidecar sample count does not match its PCM payload.")
    return pcm


async def async_run_offline_inference(
    runtime: VoiceChatRuntime,
    input_pcm16: bytes,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trailing_silence: float = 2.0,
    show_progress: bool = False,
) -> OfflineVoiceChatResult:
    """Pipeline one bounded file through direct SGLang engine sessions."""
    model_input = _model_input(input_pcm16, trailing_silence)
    frame_count = len(model_input) // FRAME_BYTES
    audio_session = None
    model_session = None
    tasks: list[asyncio.Task] = []
    text_tokens: list[int] = []
    function_tokens: list[int] = []
    output = bytearray()

    try:
        audio_session = await asyncio.to_thread(runtime.sidecar.start)
        model_session = await AsyncSGLangVoiceChatSession.create(
            runtime.duplex,
            runtime.eartts,
            capacity=runtime.session_capacity,
        )
        await model_session.start(
            runtime.prompt_ids(system_prompt),
            runtime.speaker,
            runtime.config.pad_token_id,
        )
        if frame_count > model_session.max_frames:
            raise ValueError(
                f"Offline input requires {frame_count} frames but the configured "
                f"context permits only {model_session.max_frames}."
            )

        stop = object()
        perception_queue = asyncio.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        duplex_queue = asyncio.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        eartts_queue = asyncio.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        codec_queue = asyncio.Queue(maxsize=PIPELINE_QUEUE_SIZE)

        async def input_worker() -> None:
            for index, offset in enumerate(range(0, len(model_input), FRAME_BYTES)):
                await perception_queue.put(
                    (index, model_input[offset : offset + FRAME_BYTES])
                )
            await perception_queue.put(stop)

        async def perception_worker() -> None:
            while True:
                item = await perception_queue.get()
                if item is stop:
                    await duplex_queue.put(stop)
                    return
                index, frame = item
                encoded = base64.b64encode(frame).decode()
                embedding = await asyncio.to_thread(
                    runtime.sidecar.encode, audio_session, encoded
                )
                await duplex_queue.put((index, embedding))

        async def duplex_worker() -> None:
            while True:
                item = await duplex_queue.get()
                if item is stop:
                    await eartts_queue.put(stop)
                    return
                index, embedding = item
                text_token, function_token, _ = await model_session.duplex_step(
                    embedding
                )
                await eartts_queue.put((index, text_token, function_token))

        async def eartts_worker() -> None:
            while True:
                item = await eartts_queue.get()
                if item is stop:
                    await codec_queue.put(stop)
                    return
                index, text_token, function_token = item
                codes, _ = await model_session.eartts_step(text_token)
                await codec_queue.put((index, text_token, function_token, codes))

        async def codec_worker() -> None:
            while True:
                item = await codec_queue.get()
                if item is stop:
                    return
                index, text_token, function_token, codes = item
                if index != len(text_tokens):
                    raise RuntimeError(
                        "Offline VoiceChat frames completed out of order."
                    )
                decoded = await asyncio.to_thread(
                    runtime.sidecar.decode, audio_session, codes
                )
                text_tokens.append(text_token)
                function_tokens.append(function_token)
                output.extend(_decode_sidecar_audio(decoded))
                completed = len(text_tokens)
                if show_progress and (completed % 10 == 0 or completed == frame_count):
                    print(f"processed {completed}/{frame_count} frames", flush=True)

        tasks = [
            asyncio.create_task(worker())
            for worker in (
                input_worker,
                perception_worker,
                duplex_worker,
                eartts_worker,
                codec_worker,
            )
        ]
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        try:
            if model_session is not None:
                await model_session.close()
        finally:
            if audio_session is not None:
                await asyncio.to_thread(runtime.sidecar.close, audio_session)

    pad_token_id = runtime.config.pad_token_id
    return OfflineVoiceChatResult(
        text=_decode_tokens(runtime.tokenizer, text_tokens, pad_token_id),
        function_text=_decode_tokens(runtime.tokenizer, function_tokens, pad_token_id),
        text_tokens=text_tokens,
        function_tokens=function_tokens,
        audio_pcm16=bytes(output),
        frames=frame_count,
    )


def run_offline_inference(
    runtime: VoiceChatRuntime,
    input_pcm16: bytes,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trailing_silence: float = 2.0,
    show_progress: bool = False,
) -> OfflineVoiceChatResult:
    """Synchronous entry point for direct offline inference."""
    return asyncio.run(
        async_run_offline_inference(
            runtime,
            input_pcm16,
            system_prompt=system_prompt,
            trailing_silence=trailing_silence,
            show_progress=show_progress,
        )
    )


def _write_wav(path: Path, pcm16: bytes, *, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(OUTPUT_RATE)
        destination.writeframes(pcm16)


def _combined_pcm16(input_pcm16: bytes, output_pcm16: bytes) -> bytes:
    user_pcm16 = resample_pcm16(input_pcm16, INPUT_RATE, OUTPUT_RATE)
    user = array("h")
    user.frombytes(user_pcm16)
    agent = array("h")
    agent.frombytes(output_pcm16)
    sample_count = max(len(user), len(agent))
    combined = array("h")
    for index in range(sample_count):
        combined.append(user[index] if index < len(user) else 0)
        combined.append(agent[index] if index < len(agent) else 0)
    return combined.tobytes()


def save_offline_outputs(
    result: OfflineVoiceChatResult,
    input_pcm16: bytes,
    output_dir: Path,
    input_wav: Path,
) -> dict[str, Path]:
    """Write NVIDIA-compatible text/audio outputs plus token metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_wav.stem
    paths = {
        "text": output_dir / f"{stem}_output.txt",
        "output": output_dir / f"{stem}_output.wav",
        "combined": output_dir / f"{stem}_combined.wav",
        "metadata": output_dir / f"{stem}_output.json",
    }
    paths["text"].write_text(result.text)
    _write_wav(paths["output"], result.audio_pcm16)
    _write_wav(
        paths["combined"],
        _combined_pcm16(input_pcm16, result.audio_pcm16),
        channels=2,
    )
    paths["metadata"].write_text(
        json.dumps(
            {
                "text": result.text,
                "function_text": result.function_text,
                "text_tokens": result.text_tokens,
                "function_tokens": result.function_tokens,
                "frames": result.frames,
                "input_sample_rate": INPUT_RATE,
                "output_sample_rate": OUTPUT_RATE,
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline NVIDIA NemotronLabs VoiceChat inference with SGLang"
    )
    add_runtime_arguments(parser, warmup_by_default=False)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_runtime_arguments(parser, args)
    if args.trailing_silence < 0:
        parser.error("--trailing-silence cannot be negative")

    runtime = VoiceChatRuntime(args)
    try:
        asyncio.run(runtime.warmup())
        input_pcm16 = read_wav(args.wav)
        result = run_offline_inference(
            runtime,
            input_pcm16,
            system_prompt=args.system_prompt,
            trailing_silence=args.trailing_silence,
            show_progress=True,
        )
        paths = save_offline_outputs(result, input_pcm16, args.output_dir, args.wav)
    finally:
        runtime.shutdown()

    print(f"Generated text: {result.text}")
    for name, path in paths.items():
        print(f"{name:8} -> {path}")


if __name__ == "__main__":
    main()
