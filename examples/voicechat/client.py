#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal laptop client for the SGLang VoiceChat WebSocket endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave
from array import array
from pathlib import Path

import websockets

RATE = 16_000
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2


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
    if source_rate == RATE:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    output_count = int(len(samples) * RATE / source_rate)
    output = array("h")
    for index in range(output_count):
        position = index * source_rate / RATE
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        output.append(
            round(samples[left] * (1.0 - fraction) + samples[right] * fraction)
        )
    if sys.byteorder != "little":
        output.byteswap()
    print(f"resampled input from {source_rate} Hz to {RATE} Hz")
    return output.tobytes()


async def run(args):
    pcm = read_wav(args.input_wav)
    remainder = len(pcm) % FRAME_BYTES
    if remainder:
        pcm += bytes(FRAME_BYTES - remainder)
    pcm += bytes(int(args.trailing_silence * RATE) * 2)
    remainder = len(pcm) % FRAME_BYTES
    if remainder:
        pcm += bytes(FRAME_BYTES - remainder)

    output = bytearray()
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
                    output.extend(base64.b64decode(event["delta"]))
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
        next_frame_at = asyncio.get_running_loop().time()

        for offset in range(0, len(pcm), FRAME_BYTES):
            frame = pcm[offset : offset + FRAME_BYTES]
            await socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame).decode(),
                    }
                )
            )
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
            "time to first audio: " f"{(output_times[0] - connected_at) * 1000:.1f}ms"
        )
        if intervals:
            ordered = sorted(intervals)
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            print(
                "client output interval: "
                f"mean={sum(intervals) / len(intervals):.1f}ms "
                f"p95={p95:.1f}ms frames={len(output_times)}"
            )

    with wave.open(str(args.output_wav), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(22_050)
        destination.writeframes(output)
    print(f"wrote {len(output) // 2} samples to {args.output_wav}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, default=Path("response.wav"))
    parser.add_argument(
        "--instructions", default="You are a helpful, concise voice assistant."
    )
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--pacing-adjustment", type=float, default=0.0)
    parser.add_argument("--drain-timeout", type=float, default=600.0)
    args = parser.parse_args()
    started = time.monotonic()
    asyncio.run(run(args))
    print(f"completed in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
