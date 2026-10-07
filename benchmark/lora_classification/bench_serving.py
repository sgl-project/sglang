"""Compare classification, generation, and mixed load on an existing server.

Prompts reuse a fixed token-ID sequence, so prefix caching can affect results.
Generation requests fix min/max output tokens and ignore EOS. ITL is measured
only between adjacent single-token stream chunks; batched chunks are reported
separately, never assigned fabricated per-token timestamps. CPU/RSS cover the
selected process and its recursive children. Summed RSS can double-count shared
pages; CPU can exceed 100% across cores. GPU memory describes the whole device.
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        return ordered[lower] + (
            ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
        ) * (index - lower)

    return {"p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


@dataclass
class Sample:
    kind: str
    latency_ms: float = 0
    error: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    ttft_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    chunk_intervals_ms: list[float] = field(default_factory=list)
    batched_token_chunks: int = 0


async def request(client, args, kind):
    sample = Sample(kind)
    prompt = [args.token_id] * args.input_tokens
    started = time.perf_counter()
    try:
        if kind == "classification":
            response = await client.post(
                "/v1/classify", json={"model": args.classifier_model, "input": prompt}
            )
            response.raise_for_status()
            body = response.json()
            if body.get("error") or len(body.get("data", [])) != 1:
                raise ValueError("Invalid classification response")
            sample.prompt_tokens = body.get("usage", {}).get("prompt_tokens")
        else:
            payload = {
                "model": args.base_model,
                "prompt": prompt,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
                "temperature": 0,
                "min_tokens": args.output_tokens,
                "max_tokens": args.output_tokens,
                "ignore_eos": True,
            }
            last_time, usage_tokens = None, None
            last_size = token_count = 0
            done = False
            async with client.stream(
                "POST", "/v1/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise ValueError(str(event["error"]))
                    usage = event.get("usage") or {}
                    if "completion_tokens" in usage:
                        usage_tokens = usage["completion_tokens"]
                    if "prompt_tokens" in usage:
                        sample.prompt_tokens = usage["prompt_tokens"]
                    for choice in event.get("choices", []):
                        count = len(choice.get("token_ids") or [])
                        if not count and not choice.get("text"):
                            continue
                        now = time.perf_counter()
                        if sample.ttft_ms is None:
                            sample.ttft_ms = (now - started) * 1000
                        if last_time is not None:
                            gap = (now - last_time) * 1000
                            sample.chunk_intervals_ms.append(gap)
                            if count == last_size == 1:
                                sample.itl_ms.append(gap)
                        sample.batched_token_chunks += count > 1
                        token_count += count
                        last_time, last_size = now, count
            sample.output_tokens = (
                usage_tokens if usage_tokens is not None else token_count
            )
            if not done or sample.output_tokens != args.output_tokens:
                raise ValueError(
                    f"Incomplete generation: done={done}, tokens={sample.output_tokens}/{args.output_tokens}"
                )
        if (
            sample.prompt_tokens is not None
            and sample.prompt_tokens != args.input_tokens
        ):
            raise ValueError(
                f"Prompt length changed: {sample.prompt_tokens}/{args.input_tokens}"
            )
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
        sample.error = f"{type(error).__name__}: {error}"
    sample.latency_ms = (time.perf_counter() - started) * 1000
    return sample


async def resources(args, stop):
    values = {
        "cpu_percent_server_tree": [],
        "rss_mib_server_tree": [],
        "gpu_memory_mib_device": [],
    }
    errors = []
    process = None
    known = {}
    if args.server_pid is not None:
        try:
            import psutil

            process = psutil.Process(args.server_pid)
            process.cpu_percent()
            known[process.pid] = process
        except Exception as error:
            errors.append(f"Process monitor: {error}")
    gpu = args.gpu_index is not None
    while True:
        try:
            await asyncio.wait_for(stop.wait(), args.sample_interval)
        except asyncio.TimeoutError:
            pass
        if process is not None:
            try:
                cpu, rss, current = [], [], {}
                for candidate in [process, *process.children(recursive=True)]:
                    try:
                        cached = known.get(candidate.pid)
                        if cached == candidate:
                            candidate = cached
                            cpu.append(candidate.cpu_percent())
                        else:
                            # A new process needs a baseline before its CPU is sampled.
                            candidate.cpu_percent()
                        rss.append(candidate.memory_info().rss / 2**20)
                        current[candidate.pid] = candidate
                    except psutil.NoSuchProcess:
                        continue
                known = current
                if cpu:
                    values["cpu_percent_server_tree"].append(sum(cpu))
                if rss:
                    values["rss_mib_server_tree"].append(sum(rss))
            except psutil.NoSuchProcess:
                process = None
            except Exception as error:
                errors.append(f"Process monitor: {error}")
                process = None
        if gpu:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "nvidia-smi",
                        f"--id={args.gpu_index}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )
                values["gpu_memory_mib_device"].append(float(result.stdout.strip()))
            except Exception as error:
                errors.append(f"GPU monitor: {error}")
                gpu = False
        if stop.is_set():
            break
    for name, data in values.items():
        values[name] = (
            dict(samples=len(data), mean=statistics.mean(data), peak=max(data))
            if data
            else None
        )
    return dict(values, errors=errors)


def summarize(samples, elapsed):
    result = {}
    for kind in sorted({sample.kind for sample in samples}):
        selected = [sample for sample in samples if sample.kind == kind]
        good = [sample for sample in selected if sample.error is None]
        metrics = dict(
            requests=len(selected),
            successful=len(good),
            errors=dict(Counter(sample.error for sample in selected if sample.error)),
            requests_per_second=len(good) / elapsed,
        )
        fields = ["latency_ms"]
        if kind == "generation":
            fields += ["ttft_ms", "itl_ms", "chunk_intervals_ms"]
            tokens = sum(sample.output_tokens for sample in good)
            metrics.update(
                output_tokens=tokens,
                output_tokens_per_second=tokens / elapsed,
                batched_token_chunks=sum(
                    sample.batched_token_chunks for sample in good
                ),
            )
        for name in fields:
            values = []
            for sample in good:
                value = getattr(sample, name)
                if value is not None:
                    values.extend(value if isinstance(value, list) else [value])
            metrics[name] = distribution(values)
        result[kind] = metrics
    return result


async def phase(client, args, name):
    def kind(index):
        if name == "mixed":
            return "classification" if index % 2 == 0 else "generation"
        return name

    for index in range(args.warmup):
        warmup = await request(client, args, kind(index))
        if warmup.error:
            raise RuntimeError(f"{name} warmup failed: {warmup.error}")
    samples = []
    pending = iter(range(args.requests))

    async def worker():
        for index in pending:
            samples.append(await request(client, args, kind(index)))

    stop = asyncio.Event()
    monitor = asyncio.create_task(resources(args, stop))
    started = time.perf_counter()
    try:
        await asyncio.gather(
            *(worker() for _ in range(min(args.concurrency, args.requests)))
        )
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
    measurements = await monitor
    return {
        "phase": name,
        "elapsed_seconds": elapsed,
        "metrics": summarize(samples, elapsed),
        "resources": measurements,
    }


async def run(args):
    report = {"configuration": vars(args), "runs": []}
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"), timeout=args.timeout, limits=limits
    ) as client:
        for repeat in range(args.repeat):
            phases = (
                ("classification", "generation", "mixed")
                if args.phase == "all"
                else (args.phase,)
            )
            for name in phases:
                print(f"Run {repeat + 1}: {name}", file=sys.stderr)
                result = await phase(client, args, name)
                result["repeat"] = repeat + 1
                report["runs"].append(result)
    print(json.dumps(report, indent=2))
    return int(
        any(
            metrics["errors"]
            for result in report["runs"]
            for metrics in result["metrics"].values()
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--classifier-model",
        help="Model name including :adapter suffix; required except for generation-only",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "classification", "generation", "mixed"),
        default="all",
    )
    for name, default in (
        ("input-tokens", 128),
        ("output-tokens", 32),
        ("token-id", 42),
        ("concurrency", 8),
        ("requests", 100),
        ("repeat", 1),
        ("warmup", 2),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--server-pid",
        type=int,
        help="Sample this process and recursive children using optional psutil",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        help="Sample device-wide memory using optional nvidia-smi",
    )
    parser.add_argument("--sample-interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.phase != "generation" and not args.classifier_model:
        parser.error("--classifier-model is required when classification is measured")
    positive = (
        "input_tokens output_tokens concurrency requests repeat timeout sample_interval"
    )
    if any(getattr(args, key) <= 0 for key in positive.split()):
        parser.error(
            "Token counts, concurrency, requests, repeat and timeouts must be positive"
        )
    if args.warmup < 0 or args.token_id < 0:
        parser.error("Warmup and token ID must be nonnegative")
    try:
        return asyncio.run(run(args))
    except (RuntimeError, KeyboardInterrupt) as error:
        parser.exit(1, f"Benchmark stopped: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
