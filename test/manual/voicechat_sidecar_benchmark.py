"""Measure real VoiceChat perception and codec latency through the sidecar."""

import argparse
import base64
import json
import time
import urllib.request

FRAME_SAMPLES = 1_280


def request(url, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            url.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        ),
        timeout=30,
    ) as response:
        return json.load(response)


def summary(values):
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def main(url: str, frames: int, warmup: int):
    session_id = request(url, "POST", "/session")["session_id"]
    pcm16 = base64.b64encode(bytes(FRAME_SAMPLES * 2)).decode()
    encode_ms = []
    decode_ms = []
    try:
        for index in range(frames + warmup):
            started = time.perf_counter()
            request(
                url,
                "POST",
                f"/session/{session_id}/encode",
                {"pcm16": pcm16},
            )
            encoded = time.perf_counter()
            request(
                url,
                "POST",
                f"/session/{session_id}/decode",
                {"codes": [0] * 31},
            )
            decoded = time.perf_counter()
            if index >= warmup:
                encode_ms.append((encoded - started) * 1000)
                decode_ms.append((decoded - encoded) * 1000)
    finally:
        request(url, "DELETE", f"/session/{session_id}")

    print(
        json.dumps(
            {
                "frames": frames,
                "warmup_frames": warmup,
                "perception_http_ms": summary(encode_ms),
                "codec_http_ms": summary(decode_ms),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    main(args.url, args.frames, args.warmup)
