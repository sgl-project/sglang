# SPDX-License-Identifier: Apache-2.0
"""Drive a Nomas realtime session over the WebSocket endpoint.

Each ``--segment`` is one event: ``frames:caption``. The client sends the
init request, one ``segment`` event per extra event, then ``end``; the server
answers with raw RGB frame batches holding only frames no later window can
change, and the final flush sends the rest; each event contributes exactly its
``frames``. Frames are saved
as an ``.npy`` stack and, when imageio is installed, an MP4.

Calling the PE (VLM) to produce event JSON is the caller's job; pass its
output as the caption with ``--prompt-format json``.

    python examples/nomas/realtime_client.py --url ws://127.0.0.1:30000 \\
        --segment "17:A red car drives forward." --segment "32:The car turns left." \\
        --size 64x64 --steps 2 --output /tmp/nomas_realtime
"""

import argparse
import json
import pathlib
import sys
import time

import msgspec
import numpy as np
import websocket


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="ws://127.0.0.1:30000")
    parser.add_argument(
        "--segment", action="append", required=True, help="frames:caption"
    )
    parser.add_argument("--prompt-format", choices=("plain", "json"), default="plain")
    parser.add_argument("--size", default="640x352")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def parse_segment(text):
    frames, caption = text.split(":", 1)
    return {"frames": int(frames), "caption": caption}


def main():
    args = parse_args()
    segments = [parse_segment(s) for s in args.segment]
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ws = websocket.create_connection(
        f"{args.url}/v1/realtime_video/generate", timeout=args.timeout
    )
    ws.send_binary(
        msgspec.msgpack.encode(
            {
                "type": "init",
                "prompt": segments[0]["caption"],
                "num_frames": segments[0]["frames"],
                "size": args.size,
                "fps": args.fps,
                "num_inference_steps": args.steps,
                "guidance_scale": 1,
                "seed": args.seed,
                "realtime_output_format": "raw",
                "condition_inputs": {
                    "prompt_format": args.prompt_format,
                    "segments": segments[:1],
                },
            }
        )
    )
    for index, segment in enumerate(segments[1:], start=1):
        ws.send_binary(
            msgspec.msgpack.encode(
                {
                    "type": "event",
                    "kind": "segment",
                    "event_id": index,
                    "payload": segment,
                }
            )
        )
    ws.send_binary(msgspec.msgpack.encode({"type": "event", "kind": "end"}))

    frames, chunks, started = [], [], time.time()
    while True:
        try:
            raw = ws.recv()
        except websocket.WebSocketConnectionClosedException:
            break
        if isinstance(raw, str):
            if not raw:  # closing frame
                break
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                print(f"ignoring text message: {raw[:80]}", file=sys.stderr)
                continue
        else:
            message = msgspec.msgpack.decode(raw)
        kind = message.get("type")
        if kind in ("frame_batch", "frame_batch_header"):
            # Small batches inline the payload; large ones send it as the next binary message.
            payload = message["payload"] if kind == "frame_batch" else ws.recv()
            count = message["num_frames"]
            if count:
                height, width, channels = (
                    message["height"],
                    message["width"],
                    message["channels"],
                )
                frames.append(
                    np.frombuffer(payload, dtype=np.uint8).reshape(
                        count, height, width, channels
                    )
                )
            print(f"chunk {message['chunk_index']}: {count} frames", flush=True)
        elif kind == "chunk_stats":
            chunks.append(message)
        elif kind == "error":
            print(f"server error: {message}", file=sys.stderr)
            ws.close()
            return 1
        if time.time() - started > args.timeout:
            print("timed out waiting for frames", file=sys.stderr)
            return 1
    ws.close()

    video = (
        np.concatenate(frames, axis=0)
        if frames
        else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    )
    np.save(output / "frames.npy", video)
    expected = sum(s["frames"] for s in segments)
    summary = {
        "segments": segments,
        "received_frames": int(video.shape[0]),
        "expected_frames": expected,
        "frame_shape": list(video.shape[1:]),
        "chunks": len(chunks),
        "chunk_total_ms": [c.get("chunk_total_ms") for c in chunks],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import imageio.v3 as iio

        iio.imwrite(output / "session.mp4", video, fps=args.fps)
    except Exception as exc:  # imageio or its ffmpeg plugin is optional
        summary["mp4"] = f"skipped: {exc}"
    print(json.dumps(summary, indent=2))
    return 0 if video.shape[0] == expected else 2


if __name__ == "__main__":
    sys.exit(main())
