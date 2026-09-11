"""Offline HTTP smoke and throughput checks; no dataset downloads required."""

import argparse
import base64
import concurrent.futures
import io
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image

PROMPTS = [
    "A red apple on a wooden table, studio photography.",
    "A cinematic mountain lake at sunrise with snow covered peaks reflected in "
    "calm water, a small wooden cabin beside pine trees, soft golden light, "
    "realistic photography with detailed textures and a wide composition.",
    "A blue ceramic teapot on a white background.",
    "A quiet street in an ancient Chinese town after rain, stone pavement, "
    "warm lanterns reflected in puddles, traditional wooden buildings, "
    "evening light, a detailed watercolor painting.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if (
        min(args.requests, args.concurrency, args.size, args.steps) < 1
        or args.warmup < 0
    ):
        parser.error(
            "counts, size and steps must be positive; warmup must be nonnegative"
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    def request(index):
        payload = dict(
            model=args.model,
            prompt=PROMPTS[index % len(PROMPTS)],
            seed=1000 + index,
            n=1,
            size=f"{args.size}x{args.size}",
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            response_format="b64_json",
            output_format="png",
            generator_device="npu",
        )
        start = time.perf_counter()
        try:
            req = urllib.request.Request(
                args.url + "/v1/images/generations",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3600) as response:
                body = json.load(response)
            if len(body.get("data", [])) != 1:
                raise ValueError("expected exactly one output")
            raw = base64.b64decode(body["data"][0]["b64_json"], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                if im.size != (args.size, args.size):
                    raise ValueError(f"unexpected image dimensions: {im.size}")
            result = dict(
                index=index,
                seed=payload["seed"],
                prompt=payload["prompt"],
                success=True,
                latency_s=time.perf_counter() - start,
                peak_memory_mb=body.get("peak_memory_mb"),
            )
            return result, raw
        except Exception as exc:
            return dict(
                index=index,
                success=False,
                error=str(exc),
                latency_s=time.perf_counter() - start,
            ), None

    def run(count):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            return list(pool.map(request, range(count)))

    warmup = run(args.warmup)
    if any(not item[0]["success"] for item in warmup):
        (out / "warmup_errors.json").write_text(
            json.dumps([x[0] for x in warmup], indent=2)
        )
        raise SystemExit("Warmup failed; inspect warmup_errors.json")
    del warmup
    start = time.perf_counter()
    results = run(args.requests)
    duration = time.perf_counter() - start
    records = [item[0] for item in results]
    latencies = sorted(r["latency_s"] for r in records if r["success"])
    summary = dict(
        config=vars(args),
        successful=len(latencies),
        failed=args.requests - len(latencies),
        duration_s=duration,
        outputs_per_s=len(latencies) / duration,
        mean_latency_s=statistics.mean(latencies) if latencies else None,
        p95_latency_s=latencies[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else None,
        requests=records,
    )
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_images:
        for record, raw in results:
            if raw is not None:
                (out / f"{record['index']:03d}.png").write_bytes(raw)
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
