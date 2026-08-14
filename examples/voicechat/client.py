#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Realtime microphone or WAV client for the SGLang VoiceChat endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import time
import wave
from array import array
from pathlib import Path

import websockets

INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate or not pcm:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    output_count = round(len(samples) * target_rate / source_rate)
    output = array("h")
    for index in range(output_count):
        position = index * source_rate / target_rate
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        output.append(
            round(samples[left] * (1.0 - fraction) + samples[right] * fraction)
        )
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        channels, width, source_rate = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
        )
        if (channels, width) != (1, 2):
            raise ValueError(
                "Input must be mono 16-bit PCM WAV; got "
                f"channels={channels}, sample_width={width}."
            )
        pcm = source.readframes(source.getnframes())
    output = resample_pcm16(pcm, source_rate, INPUT_RATE)
    if source_rate != INPUT_RATE:
        print(f"resampled input from {source_rate} Hz to {INPUT_RATE} Hz")
    return output


def _load_pyaudio():
    try:
        import pyaudio
    except ImportError as error:
        raise RuntimeError(
            "Microphone capture and playback require PyAudio. Install the "
            "PortAudio system package, then run `python -m pip install pyaudio`."
        ) from error
    return pyaudio


def list_audio_devices() -> None:
    pyaudio = _load_pyaudio()
    audio = pyaudio.PyAudio()
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            print(
                f"{index}: {info['name']} "
                f"(inputs={info['maxInputChannels']}, "
                f"outputs={info['maxOutputChannels']}, "
                f"default_rate={info['defaultSampleRate']})"
            )
    finally:
        audio.terminate()


class AudioDevices:
    def __init__(self, *, capture: bool, playback: bool, args):
        self.pyaudio = _load_pyaudio()
        self.audio = self.pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.input_rate = INPUT_RATE
        self.output_rate = OUTPUT_RATE
        self.input_frame_samples = FRAME_SAMPLES
        self.input_frames_read = 0
        try:
            if capture:
                if args.input_device_index is None:
                    info = self.audio.get_default_input_device_info()
                else:
                    info = self.audio.get_device_info_by_index(args.input_device_index)
                self.input_rate = round(info["defaultSampleRate"])
                self.input_frame_samples = round(self.input_rate * 0.08)
                kwargs = {
                    "format": self.pyaudio.paInt16,
                    "channels": 1,
                    "rate": self.input_rate,
                    "input": True,
                    "frames_per_buffer": self.input_frame_samples,
                }
                if args.input_device_index is not None:
                    kwargs["input_device_index"] = args.input_device_index
                self.input_stream = self.audio.open(**kwargs)
                print(
                    f"input device: {info['name']} at {self.input_rate} Hz "
                    f"(resampled to {INPUT_RATE} Hz)"
                )
            if playback:
                if args.output_device_index is None:
                    info = self.audio.get_default_output_device_info()
                else:
                    info = self.audio.get_device_info_by_index(args.output_device_index)
                self.output_rate = round(info["defaultSampleRate"])
                kwargs = {
                    "format": self.pyaudio.paInt16,
                    "channels": 1,
                    "rate": self.output_rate,
                    "output": True,
                    "frames_per_buffer": round(self.output_rate * 0.08),
                }
                if args.output_device_index is not None:
                    kwargs["output_device_index"] = args.output_device_index
                self.output_stream = self.audio.open(**kwargs)
                print(
                    f"output device: {info['name']} at {self.output_rate} Hz "
                    f"(resampled from {OUTPUT_RATE} Hz)"
                )
        except Exception:
            self.close()
            raise

    def read(self) -> bytes:
        pcm = self.input_stream.read(
            self.input_frame_samples, exception_on_overflow=False
        )
        samples = array("h")
        samples.frombytes(pcm)
        peak = max((abs(sample) for sample in samples), default=0)
        rms = math.sqrt(
            sum(sample * sample for sample in samples) / max(1, len(samples))
        )
        self.input_frames_read += 1
        if self.input_frames_read == 1 or self.input_frames_read % 12 == 0:
            print(f"microphone level: rms={rms:.0f} peak={peak}", flush=True)
        return resample_pcm16(pcm, self.input_rate, INPUT_RATE)

    def write(self, pcm: bytes) -> None:
        self.output_stream.write(resample_pcm16(pcm, OUTPUT_RATE, self.output_rate))

    def close(self) -> None:
        for stream in (self.input_stream, self.output_stream):
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
        self.input_stream = None
        self.output_stream = None
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None


def _complete_frames(pcm: bytes) -> bytes:
    remainder = len(pcm) % FRAME_BYTES
    return pcm if remainder == 0 else pcm + bytes(FRAME_BYTES - remainder)


async def _send_frame(socket, frame: bytes) -> None:
    await socket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }
        )
    )


async def _send_wav(socket, receiver, args) -> None:
    pcm = _complete_frames(read_wav(args.input_wav))
    next_frame_at = asyncio.get_running_loop().time()
    for offset in range(0, len(pcm), FRAME_BYTES):
        await _send_frame(socket, pcm[offset : offset + FRAME_BYTES])
        if receiver.done():
            await receiver
        if not args.no_realtime_pacing:
            next_frame_at += 0.08
            await asyncio.sleep(
                max(
                    0,
                    next_frame_at
                    - asyncio.get_running_loop().time()
                    - args.pacing_adjustment,
                )
            )


async def _send_microphone(socket, receiver, devices, args) -> None:
    print("Microphone streaming started.")
    if args.microphone_seconds is None:
        stop_waiter = asyncio.create_task(
            asyncio.to_thread(input, "Press Enter to stop the session.\n")
        )
        deadline = None
    else:
        stop_waiter = None
        deadline = time.monotonic() + args.microphone_seconds

    try:
        while True:
            if stop_waiter is not None and stop_waiter.done():
                await stop_waiter
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            frame = await asyncio.to_thread(devices.read)
            await _send_frame(socket, frame)
            if receiver.done():
                await receiver
    finally:
        if stop_waiter is not None and not stop_waiter.done():
            stop_waiter.cancel()


async def _send_trailing_silence(socket, receiver, args) -> None:
    seconds = args.trailing_silence
    count = math.ceil(seconds * INPUT_RATE / FRAME_SAMPLES)
    silence = bytes(FRAME_BYTES)
    next_frame_at = asyncio.get_running_loop().time()
    for _ in range(count):
        await _send_frame(socket, silence)
        if receiver.done():
            await receiver
        next_frame_at += 0.08
        if args.input_wav is None or not args.no_realtime_pacing:
            await asyncio.sleep(
                max(0, next_frame_at - asyncio.get_running_loop().time())
            )


async def run(args):
    capture = args.input_wav is None
    playback = not args.no_playback
    devices = (
        AudioDevices(capture=capture, playback=playback, args=args)
        if capture or playback
        else None
    )
    output = bytearray()
    try:
        async with websockets.connect(
            args.url, max_size=8 * 1024 * 1024, ping_timeout=60
        ) as socket:
            connected_at = time.perf_counter()
            created = json.loads(await socket.recv())
            print(json.dumps(created, indent=2))
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"instructions": args.instructions},
                    }
                )
            )
            print(await socket.recv())

            committed = asyncio.Event()
            received_frames = 0
            output_times = []

            async def receive_output():
                nonlocal received_frames
                async for raw_event in socket:
                    event = json.loads(raw_event)
                    if event.get("type") == "error":
                        raise RuntimeError(event["error"]["message"])
                    if event.get("type") == "input_audio_buffer.committed":
                        print(json.dumps(event))
                        committed.set()
                        continue
                    if event.get("type") == "response.output_audio.delta":
                        output_times.append(time.perf_counter())
                        pcm = base64.b64decode(event["delta"])
                        output.extend(pcm)
                        if devices is not None and devices.output_stream is not None:
                            await asyncio.to_thread(devices.write, pcm)
                        timing = event.get("timing_ms", {})
                        print(
                            f"frame {received_frames:04d}: "
                            f"text={event['text_token']} asr={event['asr_token']} "
                            f"audio_samples={event['samples']} "
                            f"server_ms={timing.get('total', 0):.1f} "
                            f"queue_ms={timing.get('queue', 0):.1f}"
                        )
                        received_frames += 1

            receiver = asyncio.create_task(receive_output())
            if args.input_wav is None:
                await _send_microphone(socket, receiver, devices, args)
            else:
                await _send_wav(socket, receiver, args)
            await _send_trailing_silence(socket, receiver, args)

            await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            commit_waiter = asyncio.create_task(committed.wait())
            done, _ = await asyncio.wait(
                {commit_waiter, receiver},
                timeout=args.drain_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                commit_waiter.cancel()
                raise TimeoutError("Timed out while draining queued audio frames.")
            if receiver in done:
                await receiver
            await commit_waiter
            await socket.send(json.dumps({"type": "session.close"}))
            await receiver

        if output_times:
            intervals = [
                (current - previous) * 1000
                for previous, current in zip(output_times, output_times[1:])
            ]
            print(
                "time to first audio: "
                f"{(output_times[0] - connected_at) * 1000:.1f}ms"
            )
            if intervals:
                ordered = sorted(intervals)
                p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
                print(
                    "client output interval: "
                    f"mean={sum(intervals) / len(intervals):.1f}ms "
                    f"p95={p95:.1f}ms frames={len(output_times)}"
                )
    finally:
        if devices is not None:
            devices.close()

    with wave.open(str(args.output_wav), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(OUTPUT_RATE)
        destination.writeframes(output)
    print(f"wrote {len(output) // 2} samples to {args.output_wav}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument(
        "--input-wav",
        type=Path,
        help="Stream a WAV file instead of the default microphone input.",
    )
    parser.add_argument("--output-wav", type=Path, default=Path("response.wav"))
    parser.add_argument(
        "--instructions", default="You are a helpful, concise voice assistant."
    )
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--input-device-index", type=int)
    parser.add_argument("--output-device-index", type=int)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--microphone-seconds",
        type=float,
        help="Stop microphone capture automatically after this many seconds.",
    )
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--pacing-adjustment", type=float, default=0.0)
    parser.add_argument("--drain-timeout", type=float, default=600.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_devices:
        list_audio_devices()
        return
    if args.url is None:
        parser.error("--url is required unless --list-devices is used")
    if args.trailing_silence < 0:
        parser.error("--trailing-silence cannot be negative")
    if args.microphone_seconds is not None and args.microphone_seconds <= 0:
        parser.error("--microphone-seconds must be positive")
    started = time.monotonic()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    print(f"completed in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
