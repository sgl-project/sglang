# SPDX-License-Identifier: Apache-2.0
"""Latency of a running FLUX 3 Action server, as reported in the cookbook.

Protocol (keep the cookbook numbers comparable across GPUs):

- one GPU, the default ``sglang serve`` command of the variant, eager mode;
- the OpenPI WebSocket (``/openpi/policy``) with msgpack numpy payloads: three
  uint8 360x640 cameras, an 8-dim float32 state and a fixed prompt, so the
  caption context is cached after the first request;
- one request at a time; the first ``--warmup`` requests are discarded
  (CUDA / JIT / FlexAttention compilation and the caption cache miss);
- the reported latency is the median (and p90) of the server-side
  ``server_timing.infer_ms`` over the next ``--requests`` requests, which
  excludes network and client serialization.

Usage::

    sglang serve black-forest-labs/flux-3-action-droid --model-type diffusion --port 30000
    python -m sglang.multimodal_gen.benchmarks.bench_flux3_action --url ws://127.0.0.1:30000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import numpy as np
import websockets

from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    pack_msgpack,
    unpack_msgpack,
)

CAMERAS = ("wrist", "left", "right")


def _observation(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observation = {
        f"observation.images.{name}": rng.integers(
            0, 256, (360, 640, 3), dtype=np.uint8
        )
        for name in CAMERAS
    }
    observation["observation.state"] = rng.uniform(-1, 1, 8).astype(np.float32)
    observation["prompt"] = "put the marker in the cup"
    return observation


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


async def _run(url: str, warmup: int, requests: int, seed: int) -> dict:
    observation = pack_msgpack(_observation(seed))
    async with websockets.connect(f"{url}/openpi/policy", max_size=None) as ws:
        metadata = unpack_msgpack(await ws.recv())
        infer, stages, round_trip = [], {}, []
        for i in range(warmup + requests):
            start = time.perf_counter()
            await ws.send(observation)
            response = unpack_msgpack(await ws.recv())
            elapsed = (time.perf_counter() - start) * 1000
            if i < warmup:
                continue
            round_trip.append(elapsed)
            infer.append(float(response["server_timing"]["infer_ms"]))
            for name, value in response["timings"].items():
                stages.setdefault(name, []).append(float(value))
    return {
        "model": metadata.get("model"),
        "variant": metadata.get("defaults", {}).get("variant"),
        "warmup": warmup,
        "requests": requests,
        "infer_ms_median": statistics.median(infer),
        "infer_ms_p90": _percentile(infer, 90),
        "round_trip_ms_median": statistics.median(round_trip),
        "stage_ms_median": {k: statistics.median(v) for k, v in stages.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="ws://127.0.0.1:30000")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print the raw result")
    args = parser.parse_args()
    result = asyncio.run(_run(args.url, args.warmup, args.requests, args.seed))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    stages = "  ".join(f"{k}={v:.1f}" for k, v in result["stage_ms_median"].items())
    print(
        f"{result['model']} [{result['variant']}]: "
        f"infer {result['infer_ms_median']:.0f} ms median, "
        f"{result['infer_ms_p90']:.0f} ms p90 "
        f"(round trip {result['round_trip_ms_median']:.0f} ms; "
        f"{result['requests']} requests after {result['warmup']} warmup)\n  {stages}"
    )


if __name__ == "__main__":
    main()
