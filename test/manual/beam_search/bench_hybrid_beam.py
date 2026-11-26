"""Compare ordinary and shared FlashInfer decode before measuring hybrid beams.

Run once against the ordinary server to write a baseline, then restart on the
same dedicated GPU with SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION=1 and pass
that baseline with --baseline. Use the same model, dtypes, and server options
in both runs; use --enable-metrics, --page-size 1 and --disable-radix-cache.
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from pathlib import Path

import aiohttp

WIDTHS = (256, 5120, 1024, 2048)
PROMPT_LEN = 1024
MAX_RETURNED_BEAMS = 4096  # The /generate n limit does not cap search width.


def prompt(seed):
    rng = random.Random(seed)
    return [rng.randrange(100, 10000) for _ in range(PROMPT_LEN)]


async def generate(session, width, input_ids, output_len):
    returned_beams = min(width, MAX_RETURNED_BEAMS)
    start = time.perf_counter()
    async with session.post(
        "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "beam_width": width,
                "n": returned_beams,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
        },
    ) as response:
        if not response.ok:
            raise RuntimeError(
                f"Beam width {width}: HTTP {response.status}: {await response.text()}"
            )
        payload = await response.json()
    elapsed = time.perf_counter() - start
    beams = [
        (tuple(beam["output_ids"]), beam["meta_info"]["sequence_score"])
        for beam in payload["meta_info"]["beam_results"]
    ]
    if len(beams) != returned_beams or len(dict(beams)) != returned_beams:
        raise RuntimeError(
            f"Expected {returned_beams} distinct beams, got {len(dict(beams))}"
        )
    if any(
        len(tokens) != output_len or not math.isfinite(score) for tokens, score in beams
    ):
        raise RuntimeError("Invalid beam length or score")
    return beams, elapsed


def compare(actual, expected, *, min_overlap, score_atol, score_p99_atol=None):
    actual_by_tokens = {tuple(tokens): score for tokens, score in actual}
    expected_by_tokens = {tuple(tokens): score for tokens, score in expected}
    shared = actual_by_tokens.keys() & expected_by_tokens.keys()
    overlap = len(shared) / len(expected)
    errors = sorted(
        abs(actual_by_tokens[tokens] - expected_by_tokens[tokens]) for tokens in shared
    )
    error = errors[-1] if errors else math.inf
    p99_error = errors[math.ceil(0.99 * len(errors)) - 1] if errors else math.inf
    top_match = tuple(actual[0][0]) == tuple(expected[0][0])
    exact_order = list(actual_by_tokens) == list(expected_by_tokens)
    if (
        not top_match
        or overlap < min_overlap
        or error > score_atol
        or (score_p99_atol is not None and p99_error > score_p99_atol)
    ):
        raise RuntimeError(
            f"Parity failed: {top_match=}, {overlap=:.4f}, "
            f"{error=:.5f}, {p99_error=:.5f}, {exact_order=}"
        )
    return {
        "overlap": overlap,
        "max_score_error": error,
        "p99_score_error": p99_error,
        "top_score_error": abs(actual[0][1] - expected[0][1]),
        "exact_order": exact_order,
    }


async def sample_metrics(session, stop, gpu, samples):
    from pynvml import nvmlDeviceGetUtilizationRates

    while not stop.is_set():
        async with session.get("/metrics") as response:
            response.raise_for_status()
            metrics = await response.text()
        queued = [
            float(line.rsplit(maxsplit=1)[-1])
            for line in metrics.splitlines()
            if line.startswith(("sglang:num_queue_reqs{", "sglang:num_queue_reqs "))
        ]
        if not queued:
            raise RuntimeError("Queue depth unavailable; launch with --enable-metrics")
        samples.append((nvmlDeviceGetUtilizationRates(gpu).gpu, max(queued)))
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass


async def measure(session, args, width, concurrency, gpu):
    requests = iter(range(args.requests))
    inputs = [prompt(100 + i) for i in range(args.requests)]

    async def worker():
        return [
            (await generate(session, width, inputs[i], args.output_len))[1]
            for i in requests
        ]

    await asyncio.gather(
        *(
            generate(session, width, prompt(i), args.output_len)
            for i in range(concurrency)
        )
    )
    samples = []
    stop = asyncio.Event()
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(sample_metrics(session, stop, gpu, samples))
        start = time.perf_counter()
        try:
            workers = await asyncio.gather(*(worker() for _ in range(concurrency)))
            duration = time.perf_counter() - start
        finally:
            stop.set()
    latency_ms = sorted(t * 1000 for group in workers for t in group)
    gpu_util, queued = zip(*samples, strict=True)
    return {
        "beam_width": width,
        "returned_beams": min(width, MAX_RETURNED_BEAMS),
        "concurrency": concurrency,
        "requests": len(latency_ms),
        "requests_per_s": len(latency_ms) / duration,
        "beam_tokens_per_s": len(latency_ms) * width * args.output_len / duration,
        "latency_mean_ms": statistics.mean(latency_ms),
        **{
            f"latency_p{p}_ms": latency_ms[math.ceil(p * len(latency_ms) / 100) - 1]
            for p in (50, 95, 99, 100)
        },
        "gpu_util_mean_pct": statistics.mean(gpu_util),
        "max_sampled_queued": max(queued),
        "metric_samples": len(samples),
    }


async def run(args):
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    async with aiohttp.ClientSession(
        args.url, timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        async with session.get("/server_info") as response:
            response.raise_for_status()
            info = await response.json()
        config = {
            key: info[key]
            for key in (
                "model_path",
                "max_running_requests",
                "mamba_ssm_dtype",
                "dtype",
            )
        }
        config.update(
            prompt_len=PROMPT_LEN,
            output_len=args.output_len,
            parity_prompts=args.parity_prompts,
            max_returned_beams=MAX_RETURNED_BEAMS,
        )
        result = {"config": config}
        if baseline is not None and baseline["config"] != config:
            raise RuntimeError("Baseline and shared server configurations differ")

        cases = {}
        for width in WIDTHS:
            width_cases = []
            for seed in range(args.parity_prompts):
                beams, _ = await generate(session, width, prompt(seed), args.output_len)
                if baseline is None:
                    width_cases.append(beams)
                    continue
                try:
                    width_cases.append(
                        compare(
                            beams,
                            baseline["cases"][str(width)][seed],
                            min_overlap=args.min_overlap,
                            score_atol=args.score_atol,
                            score_p99_atol=args.score_p99_atol,
                        )
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Beam width {width}, prompt {seed}: {exc}"
                    ) from exc
            cases[str(width)] = width_cases
        result["cases"] = cases
        if baseline is not None:
            from pynvml import (
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetName,
                nvmlInit,
                nvmlShutdown,
            )

            nvmlInit()
            try:
                gpu = nvmlDeviceGetHandleByIndex(args.gpu)
                result["gpu"] = nvmlDeviceGetName(gpu)
                result["gpu_index"] = args.gpu
                result["parity_gate"] = {
                    "min_overlap": args.min_overlap,
                    "score_atol": args.score_atol,
                    "score_p99_atol": args.score_p99_atol,
                }
                points = args.point or [
                    (width, c)
                    for width in WIDTHS
                    for c in args.concurrency
                    if width * c <= config["max_running_requests"]
                ]
                result.update(status="parity_passed", points=[])
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                for width, concurrency in points:
                    if width * concurrency > config["max_running_requests"]:
                        raise ValueError(
                            f"Point {width}:{concurrency} exceeds the row pool"
                        )
                    point = await measure(session, args, width, concurrency, gpu)
                    result["points"].append(point)
                    print(json.dumps(point), flush=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                result["status"] = "completed"
            finally:
                nvmlShutdown()
        args.output.write_text(json.dumps(result, indent=2) + "\n")


def parse_point(text):
    try:
        width, concurrency = map(int, text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("Expected WIDTH:CONCURRENCY") from None
    if width not in WIDTHS or concurrency < 1:
        raise argparse.ArgumentTypeError(
            "Expected a matrix width and positive concurrency"
        )
    return width, concurrency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gpu", type=int, help="Physical GPU index for NVML sampling")
    parser.add_argument("--output-len", type=int, default=3)
    parser.add_argument("--parity-prompts", type=int, default=3)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--point",
        type=parse_point,
        action="append",
        help="WIDTH:CONCURRENCY; repeat to select specific load points",
    )
    parser.add_argument("--min-overlap", type=float, default=1.0)
    parser.add_argument("--score-atol", type=float, default=0.05)
    parser.add_argument("--score-p99-atol", type=float)
    args = parser.parse_args()
    if args.baseline and (args.gpu is None or args.gpu < 0):
        parser.error("--baseline requires a nonnegative physical --gpu index")
    if (
        min(args.requests, args.parity_prompts, *args.concurrency) < 1
        or args.output_len < 2
    ):
        parser.error(
            "Require positive request counts and concurrency and output length >= 2"
        )
    if (
        not 0 < args.min_overlap <= 1
        or not math.isfinite(args.score_atol)
        or args.score_atol < 0
        or (
            args.score_p99_atol is not None
            and (not math.isfinite(args.score_p99_atol) or args.score_p99_atol < 0)
        )
    ):
        parser.error("Invalid parity thresholds")
    asyncio.run(run(args))
