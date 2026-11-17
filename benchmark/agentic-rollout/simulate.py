# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp", "transformers"]
# ///
"""Synthetic multi-turn HTTP workload. See README.md for examples."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
import sys
import time
import uuid
from pathlib import Path

import aiohttp


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record(file, value):
    file.write(json.dumps(value, sort_keys=True) + "\n")
    file.flush()


def synthetic_tokens(tokenizer, seed, conversation, turn, length):
    text = (
        f"Session {seed}/{conversation}. Tool result {turn}: "
        "The local search returned a synthetic measurement. "
        "Read the result and continue the investigation. "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Tokenizer produced no input tokens")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


async def sse_events(content):
    # aiohttp iterates complete lines, even when TCP splits a UTF-8 character.
    data = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        raise ValueError("Truncated SSE frame")


async def check_response(response):
    if response.status >= 400:
        detail = (await response.text())[:2000]
        raise ValueError(f"HTTP {response.status} {response.url}: {detail}")


async def json_request(http, method, url, **kwargs):
    async with http.request(method, url, **kwargs) as response:
        await check_response(response)
        return await response.json()


async def generate(http, url, payload, row, incremental, dp_size=None):
    start = time.perf_counter()
    row.update(submitted_at=time.time(), events=[])
    count, ids, meta, done = 0, [], {}, False
    async with http.post(url + "/generate", json=payload) as response:
        await check_response(response)
        async for event in sse_events(response.content):
            if event == "[DONE]":
                done = True
                break
            message = json.loads(event)
            if "error" in message:
                raise ValueError(str(message["error"]))
            meta = message["meta_info"]
            current = meta["completion_tokens"]
            if current < count:
                raise ValueError("Completion count moved backwards")
            if current > count:
                row["events"].append([time.perf_counter() - start, current])
                if not count:
                    row["first_token_at"] = time.time()
                    row["ttft_s"] = row["events"][-1][0]
                count = current
            if "output_ids" in message:
                if incremental:
                    ids.extend(message["output_ids"])
                else:
                    ids = message["output_ids"]
            row["meta_info"] = meta
    row.update(completed_at=time.time(), latency_s=time.perf_counter() - start)
    if (
        not done
        or not count
        or (meta.get("finish_reason") or {}).get("type") != "length"
    ):
        raise ValueError(
            f"Incomplete or aborted generation: {meta.get('finish_reason')}"
        )
    expected = payload["sampling_params"]["max_new_tokens"]
    if count != expected or len(ids) != expected:
        raise ValueError(
            f"Expected {expected} tokens, got count={count}, IDs={len(ids)}"
        )
    if meta["prompt_tokens"] != row["context_tokens"]:
        raise ValueError(f"Wrong context length: {meta['prompt_tokens']}")
    if (
        row["rank"] is not None
        and meta.get("dp_rank") != row["rank"]
        and not (dp_size == 1 and row["rank"] == 0 and meta.get("dp_rank") is None)
    ):
        raise ValueError(f"Wrong DP rank: {meta.get('dp_rank')}")
    events = row["events"]
    row["avg_token_time_s"] = (
        (events[-1][0] - events[0][0]) / (count - events[0][1])
        if count > events[0][1]
        else None
    )
    row["output_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return meta["id"], ids


async def scrape_metrics(http, url, file, stopped):
    while not stopped.is_set():
        started = time.monotonic()
        row = {"timestamp": time.time(), "url": url}
        try:
            async with http.get(url) as response:
                response.raise_for_status()
                row["text"] = await response.text()
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
            row["error"] = str(exc)
        row["duration_s"] = time.monotonic() - started
        record(file, row)
        try:
            await asyncio.wait_for(stopped.wait(), max(0.01, 1 - row["duration_s"]))
        except asyncio.TimeoutError:
            pass


def max_context_tokens(args):
    return (
        args.initial_tokens
        + (args.turns - 1) * (args.tool_tokens + args.output_tokens)
        + args.output_tokens
    )


async def conversation(
    http, tokenizer, args, info, slot, semaphore, requests, sessions
):
    rank = slot % info["dp_size"] if not args.disable_dp_sticky_routing else None
    rng = random.Random(f"{args.seed}/{slot}")
    await asyncio.sleep(args.start_spread * slot / args.conversations)
    sid, rid, history = None, None, []
    identity = {"conversation": slot, "rank": rank}
    try:
        if args.mode != "full-history":
            sid = uuid.uuid4().hex
            opened = await json_request(
                http,
                "POST",
                args.base_url + "/open_session",
                json={
                    "session_id": sid,
                    "streaming": args.mode == "streaming",
                    # Required by older servers, but unused by session execution.
                    "capacity_of_str_len": 1000,
                },
            )
            if opened != sid:
                raise ValueError(f"Unexpected open response: {opened}")
            record(
                sessions,
                {**identity, "event": "open", "id": sid, "timestamp": time.time()},
            )
        for turn in range(args.turns):
            delay = rng.uniform(*args.tool_delay) if turn else 0
            tool_started_at = time.time()
            delay_start = time.perf_counter()
            await asyncio.sleep(delay)
            actual_delay = time.perf_counter() - delay_start
            tool_completed_at = time.time()
            length = args.tool_tokens if turn else args.initial_tokens
            chunk = synthetic_tokens(tokenizer, args.seed, slot, turn, length)
            history.extend(chunk)
            row = {
                **identity,
                "turn": turn,
                "context_tokens": len(history),
                "tool_delay_s": delay,
                "tool_started_at": tool_started_at if turn else None,
                "tool_completed_at": tool_completed_at if turn else None,
                "actual_tool_delay_s": actual_delay,
                "input_tokens": length,
            }
            payload = {
                "input_ids": history if args.mode == "full-history" else chunk,
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "ignore_eos": True,
                    "max_new_tokens": args.output_tokens,
                },
            }
            if sid is not None:
                payload["session_params"] = {"id": sid, "rid": rid}
            if rank is not None:
                payload["routed_dp_rank"] = rank
            row["client_wait_started_at"] = time.time()
            waiting = time.perf_counter()
            try:
                async with semaphore:
                    row["client_wait_s"] = time.perf_counter() - waiting
                    rid, output = await generate(
                        http,
                        args.base_url,
                        payload,
                        row,
                        info.get("incremental_streaming_output", False),
                        dp_size=info.get("dp_size"),
                    )
                history.extend(output)
            except BaseException as exc:
                row["failed_at"] = time.time()
                row["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record(requests, row)
    finally:
        if sid is not None:
            event = {
                **identity,
                "event": "close",
                "id": sid,
                "timestamp": time.time(),
            }
            try:
                async with http.post(
                    args.base_url + "/close_session",
                    json={"session_id": sid},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    response.raise_for_status()
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
                event["error"] = str(exc)
                raise
            finally:
                record(sessions, event)


async def run(args, tokenizer):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "started_at": time.time(),
        "status": "running",
        "python": sys.version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {
            n: importlib.metadata.version(n) for n in ("aiohttp", "transformers")
        },
    }
    stopped = asyncio.Event()
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=args.timeout),
                connector=aiohttp.TCPConnector(limit=0),
            ) as http,
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as metrics_http,
        ):
            info = await json_request(http, "GET", args.base_url + "/server_info")
            manifest["server_info"] = info
            context = max_context_tokens(args)
            if info.get("context_length") and context > info["context_length"]:
                raise ValueError(
                    f"Workload requires {context} tokens; server context is too small"
                )
            if not args.disable_dp_sticky_routing and info.get("dp_size", 0) < 1:
                raise ValueError("DP sticky routing requires server_info.dp_size")
            write_json(manifest_path, manifest)
            semaphore = asyncio.Semaphore(args.concurrency)
            with (
                (args.output_dir / "requests.jsonl").open("w") as requests,
                (args.output_dir / "sessions.jsonl").open("w") as sessions,
                (args.output_dir / "metrics.jsonl").open("w") as metrics,
            ):
                scrapers = [
                    asyncio.create_task(
                        scrape_metrics(metrics_http, url, metrics, stopped)
                    )
                    for url in (args.metrics_url or [args.base_url + "/metrics"])
                ]
                try:
                    async with asyncio.TaskGroup() as group:
                        for slot in range(args.conversations):
                            group.create_task(
                                conversation(
                                    http,
                                    tokenizer,
                                    args,
                                    info,
                                    slot,
                                    semaphore,
                                    requests,
                                    sessions,
                                )
                            )
                finally:
                    stopped.set()
                    await asyncio.gather(*scrapers)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["finished_at"] = time.time()
        write_json(manifest_path, manifest)
        print(f"{manifest['status']}: {args.output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30000", help="SGLang server URL"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Model ID or tokenizer path matching the server",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom tokenizer code"
    )
    parser.add_argument(
        "--mode",
        choices=("full-history", "ordinary", "streaming"),
        default="ordinary",
        help="Send full history or use ordinary/streaming sessions",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=8,
        help="Number of independent conversations",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum in-flight generation requests",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=8,
        help="Generation turns per conversation",
    )
    parser.add_argument(
        "--initial-tokens",
        type=int,
        default=2048,
        help="Tokens in each initial prompt",
    )
    parser.add_argument(
        "--tool-tokens",
        type=int,
        default=256,
        help="Synthetic tool-result tokens appended each turn",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
        help="Exact number of tokens generated per turn",
    )
    parser.add_argument(
        "--tool-delay",
        type=float,
        nargs=2,
        default=[1, 3],
        metavar=("MIN", "MAX"),
        help="Range of simulated tool waits between turns, in seconds",
    )
    parser.add_argument(
        "--start-spread",
        type=float,
        default=5,
        help="Seconds over which conversation starts are staggered",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for synthetic inputs and tool delays"
    )
    parser.add_argument(
        "--disable-dp-sticky-routing",
        action="store_true",
        help="Disable default routing of conversation i to data-parallel rank i %% dp_size",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--metrics-url",
        action="append",
        help="Metrics endpoint; repeat for distinct exporters (default: BASE_URL/metrics)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for request records, metrics, and run settings",
    )
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    return args


if __name__ == "__main__":
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    asyncio.run(run(args, tokenizer))
