"""Replay metadata from https://github.com/uw-syfi/TraceLab.

The public corpus does not contain prompt text. Token lengths and session order
are observed data; generated prompt content is synthetic, not a coding-quality
or speculative-acceptance benchmark.

Download the public trace, choose any sufficiently large UTF-8 code/text corpus,
then run the replay::

    curl -L --fail -o syfi_coding_trace.jsonl.gz \
      https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.jsonl.gz
    curl -L --fail -o enwik9.zip http://mattmahoney.net/dc/enwik9.zip
    unzip enwik9.zip

    python -m sglang.benchmark.tracelab \
        --dataset-path syfi_coding_trace.jsonl.gz \
        --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
        --text-file enwik9 \
        --min-input-len 131072 --max-input-len 262144 \
        --min-output-len 128 --max-output-len 4096 \
        --num-requests 128 --concurrency 8 --request-rate 0.5 \
        --output-file tracelab-results.jsonl

Length bounds select observed rounds without truncation. Requests within each
selected session execute in round order, replaying canonical CSV arrival and
tool-wait fields. The global request rate additionally controls start spacing.
Different sessions run concurrently.
Recorded token counts come from the source model's tokenizer and are used as
target token counts, not retokenizations of unavailable original text. Prefix
reuse retains preceding prompts and actual generated IDs. Missing history is
filled from a supplied corpus, as in TraceLab's replay/src/tokens.rs. Cold first
requests and skipped rounds are therefore distinguishable
from trace cache counts in the output. Dataset attribution: UW SyFI TraceLab,
CC BY 4.0, https://github.com/uw-syfi/TraceLab.
"""

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class TraceRound:
    session_id: str
    round_id: str
    round_index: int
    input_tokens: int
    output_tokens: int
    prefix_tokens: int
    arrival_time_ms: float = 0
    tool_wait_after_ms: float = 0


@dataclass(frozen=True)
class LengthBounds:
    min_input: int = 1
    max_input: Optional[int] = None
    min_output: int = 1
    max_output: Optional[int] = None

    def __post_init__(self):
        for lower, upper in (
            (self.min_input, self.max_input),
            (self.min_output, self.max_output),
        ):
            if lower < 1 or (upper is not None and upper < lower):
                raise ValueError("Token bounds must be positive and ordered")

    def accepts(self, row: TraceRound) -> bool:
        return (
            row.input_tokens >= self.min_input
            and (self.max_input is None or row.input_tokens <= self.max_input)
            and row.output_tokens >= self.min_output
            and (self.max_output is None or row.output_tokens <= self.max_output)
        )


def _integer(record: dict, key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def parse_round(record: dict) -> TraceRound:
    """Validate the released TraceLab token accounting without adding cache twice."""
    session = record.get("session_id")
    round_id = record.get("round_id")
    if not isinstance(session, str) or not session:
        raise ValueError("session_id must be a nonempty string")
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("round_id must be a nonempty string")
    total = _integer(record, "input_tokens_total")
    prefix = _integer(record, "prefix_tokens")
    appended = _integer(record, "newly_append_tokens")
    if total != prefix + appended:
        raise ValueError(
            "input_tokens_total must equal prefix_tokens + newly_append_tokens"
        )
    timing = {}
    for key in ("arrival_time_ms", "tool_wait_after_ms"):
        value = record.get(key, 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
        timing[key] = value
    return TraceRound(
        session_id=session,
        round_id=round_id,
        round_index=_integer(record, "round_index"),
        input_tokens=total,
        output_tokens=_integer(record, "output_tokens"),
        prefix_tokens=prefix,
        **timing,
    )


def iter_rounds(
    path: str,
    bounds: Optional[LengthBounds] = None,
    limit: Optional[int] = None,
    max_rounds_per_session: Optional[int] = None,
) -> Iterator[TraceRound]:
    """Filter, never truncate, observed lengths; report malformed rows explicitly."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if max_rounds_per_session is not None and max_rounds_per_session < 1:
        raise ValueError("max_rounds_per_session must be positive")
    bounds = bounds or LengthBounds()
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    accepted = 0
    per_session = defaultdict(int)
    with opener(source, "rt", encoding="utf-8") as stream:
        is_csv = source.name.removesuffix(".gz").endswith(".csv")
        records = csv.DictReader(stream) if is_csv else stream
        for line_number, line in enumerate(records, 1):
            if not is_csv and not line.strip():
                continue
            try:
                record = parse_csv_record(line) if is_csv else json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                row = parse_round(record)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            if bounds.accepts(row):
                if (
                    max_rounds_per_session is not None
                    and per_session[row.session_id] >= max_rounds_per_session
                ):
                    continue
                yield row
                per_session[row.session_id] += 1
                accepted += 1
                if limit is not None and accepted >= limit:
                    return


def parse_csv_record(record):
    """Accept TraceLab's session-runner and canonical simulator CSV schemas."""
    session = record.get("session_id") or record.get("id")
    prefix, appended = int(record["prefix_len"]), int(record["input_len"])
    return {
        "session_id": session,
        "round_id": f"{session}:{record['round_idx']}",
        "round_index": int(record["round_idx"]),
        "prefix_tokens": prefix,
        "newly_append_tokens": appended,
        "input_tokens_total": prefix + appended,
        "output_tokens": int(record["output_len"]),
        "arrival_time_ms": float(record.get("arrival_time") or 0),
        "tool_wait_after_ms": float(record.get("tool_wait_after_ms") or 0),
    }


async def replay_sessions(rows, send, *, concurrency=1, request_rate=math.inf):
    """Serialize each session, with a global start-rate cap across active sessions.

    ``send`` is an async callable receiving a TraceRound. Its returned result is
    paired with that row. Request exceptions cancel the replay rather than
    silently fabricating a successful measurement. Rate waiting occurs before
    the transport starts its service-latency clock.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if math.isnan(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be positive")
    sessions = defaultdict(list)
    identities = set()
    for row in rows:
        identity = (row.session_id, row.round_index)
        if identity in identities:
            raise ValueError(f"Duplicate session round: {identity}")
        identities.add(identity)
        sessions[row.session_id].append(row)
    for rounds in sessions.values():
        rounds.sort(key=lambda row: row.round_index)

    session_queue = asyncio.Queue()
    for rounds in sorted(sessions.values(), key=lambda rows: rows[0].arrival_time_ms):
        session_queue.put_nowait(rounds)
    interval = 0.0 if math.isinf(request_rate) else 1.0 / request_rate
    rate_lock = asyncio.Lock()
    next_start = 0.0
    results = []
    replay_started = asyncio.get_running_loop().time()

    async def consume():
        nonlocal next_start
        while True:
            try:
                rounds = session_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            delay = (
                replay_started
                + rounds[0].arrival_time_ms / 1000
                - asyncio.get_running_loop().time()
            )
            if delay > 0:
                await asyncio.sleep(delay)
            for row in rounds:
                async with rate_lock:
                    now = asyncio.get_running_loop().time()
                    if next_start > now:
                        await asyncio.sleep(next_start - now)
                    next_start = asyncio.get_running_loop().time() + interval
                result = await send(row)
                results.append((row, result))
                if row.tool_wait_after_ms:
                    await asyncio.sleep(row.tool_wait_after_ms / 1000)
            session_queue.task_done()

    tasks = [
        asyncio.create_task(consume()) for _ in range(min(concurrency, len(sessions)))
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


@dataclass
class StreamMetrics:
    """Client-observed timing; multi-token chunk TBT is an estimate, not a trace."""

    started: float
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    output_tokens: int = 0
    chunk_gaps: list = field(default_factory=list)
    chunk_token_counts: list = field(default_factory=list)

    def observe(self, completion_tokens: int, arrived: float):
        if isinstance(completion_tokens, bool) or not isinstance(
            completion_tokens, int
        ):
            raise ValueError("completion_tokens must be an integer")
        if completion_tokens < self.output_tokens:
            raise ValueError("Cumulative completion token count regressed")
        if arrived < (self.last_token_at or self.started):
            raise ValueError("Stream timestamps regressed")
        delta = completion_tokens - self.output_tokens
        if not delta:
            return
        if self.first_token_at is None:
            self.first_token_at = arrived
        else:
            self.chunk_gaps.append(arrived - self.last_token_at)
            self.chunk_token_counts.append(delta)
        self.last_token_at = arrived
        self.output_tokens = completion_tokens

    def result(self):
        if self.first_token_at is None:
            raise ValueError("Stream ended without output tokens")
        measured_tokens = sum(self.chunk_token_counts)
        return {
            "output_tokens": self.output_tokens,
            "ttft_s": self.first_token_at - self.started,
            "generation_latency_s": self.last_token_at - self.started,
            "mean_tbt_s": (
                sum(self.chunk_gaps) / measured_tokens if measured_tokens else None
            ),
            "tbt_method": "client_chunk_gap_divided_by_new_tokens",
            "chunk_gaps_s": self.chunk_gaps,
            "chunk_token_counts": self.chunk_token_counts,
        }


async def iter_sse_data(content):
    """Decode SSE events across transport chunks, including CRLF and comments."""
    data = []
    async for raw_line in content:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


async def send_generate(client, base_url, row, input_ids):
    """Measure native SGLang /generate streaming without retokenizing prompts."""
    if len(input_ids) != row.input_tokens:
        raise ValueError("Prompt token count differs from the selected trace round")
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": row.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    clock = asyncio.get_running_loop().time
    metrics = StreamMetrics(clock())
    complete = False
    cached_tokens = None
    output_ids = []
    async with client.post(
        base_url.rstrip("/") + "/generate", json=payload
    ) as response:
        response.raise_for_status()
        async for data in iter_sse_data(response.content):
            if data == "[DONE]":
                complete = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"Server stream error: {event['error']}")
            meta = event.get("meta_info", {})
            if "output_ids" in event:
                ids = event["output_ids"]
                total = meta.get("completion_tokens")
                if not isinstance(ids, list) or any(
                    type(token) is not int for token in ids
                ):
                    raise ValueError("Invalid generated token IDs")
                if len(ids) == total:
                    output_ids = ids
                elif len(output_ids) + len(ids) == total:
                    output_ids.extend(ids)
                else:
                    raise ValueError(
                        "Generated token IDs disagree with completion count"
                    )
            if "completion_tokens" in meta:
                metrics.observe(meta["completion_tokens"], clock())
            cached_tokens = meta.get("cached_tokens", cached_tokens)
    if not complete:
        raise RuntimeError("SSE stream ended without [DONE]")
    result = metrics.result()
    result.update(
        request_latency_s=clock() - metrics.started,
        cached_tokens=cached_tokens,
        requested_output_tokens=row.output_tokens,
        output_length_matched=metrics.output_tokens == row.output_tokens,
        output_ids=output_ids,
    )
    return result


class SyntheticPrompts:
    """Use vocabulary IDs for length-only probes, retaining committed history."""

    def __init__(self, token_ids, seed=0):
        if not token_ids:
            raise ValueError("Tokenizer has no usable non-special tokens")
        self.token_ids = sorted(token_ids)
        self.seed = seed
        self.previous = {}

    def build(self, row):
        previous = self.previous.get(row.session_id, [])
        reused = min(row.prefix_tokens, len(previous), row.input_tokens)
        identity = f"{self.seed}:{row.session_id}:{row.round_id}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest(), "big"))
        prompt = previous[:reused] + rng.choices(
            self.token_ids, k=row.input_tokens - reused
        )
        return prompt, reused

    def commit_output(self, row, prompt, output_ids):
        if len(output_ids) != row.output_tokens:
            raise ValueError(
                "Exact generated token IDs are required for session replay"
            )
        self.previous[row.session_id] = prompt + output_ids


class CorpusPrompts(SyntheticPrompts):
    """TraceLab replay/src/tokens.rs: retained context plus new corpus tokens."""

    def __init__(self, token_ids, seed=0):
        super().__init__(token_ids, seed)
        self.token_ids = list(token_ids)
        self.cursors = {}

    def build(self, row):
        previous = self.previous.get(row.session_id, [])
        reused = min(row.prefix_tokens, len(previous))
        if row.session_id not in self.cursors:
            digest = hashlib.sha256(f"{self.seed}:{row.session_id}".encode()).digest()
            self.cursors[row.session_id] = int.from_bytes(digest, "big") % len(
                self.token_ids
            )
        cursor = self.cursors[row.session_id]
        count = row.input_tokens - reused
        fresh = [
            self.token_ids[(cursor + i) % len(self.token_ids)] for i in range(count)
        ]
        self.cursors[row.session_id] = (cursor + count) % len(self.token_ids)
        return previous[:reused] + fresh, reused


def summarize_results(results, elapsed):
    """Aggregate successful requests; preserve failed counts in the denominator."""
    successful = [result for _, result in results if result["success"]]

    def distribution(values):
        values = sorted(values)
        if not values:
            return None
        return {
            "mean": statistics.mean(values),
            **{
                f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
                for p in (50, 90, 99)
            },
        }

    tbt = [
        gap / count
        for result in successful
        for gap, count in zip(result["chunk_gaps_s"], result["chunk_token_counts"])
        for _ in range(count)
    ]
    output_tokens = sum(result["output_tokens"] for result in successful)
    return {
        "type": "summary",
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "duration_s": elapsed,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / elapsed if elapsed > 0 else None,
        "ttft_s": distribution([result["ttft_s"] for result in successful]),
        "estimated_tbt_s": distribution(tbt),
        "percentile_method": "nearest_rank",
    }


async def run_trace(args):
    import aiohttp
    from transformers import AutoTokenizer

    bounds = LengthBounds(
        args.min_input_len, args.max_input_len, args.min_output_len, args.max_output_len
    )
    rows = list(
        iter_rounds(
            args.dataset_path, bounds, args.num_requests, args.max_rounds_per_session
        )
    )
    if not rows:
        raise ValueError("No trace rows satisfy the length bounds")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    excluded = set(tokenizer.all_special_ids)
    prompts = SyntheticPrompts(
        [value for value in tokenizer.get_vocab().values() if value not in excluded],
        args.seed,
    )
    if args.text_file:
        token_ids = []
        with open(args.text_file, encoding="utf-8") as corpus:
            for line in corpus:
                token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
                if len(token_ids) >= args.token_pool_limit:
                    token_ids = token_ids[: args.token_pool_limit]
                    break
        if len(token_ids) < max(row.input_tokens for row in rows):
            raise ValueError(
                "Text corpus must cover the longest selected prompt without repetition"
            )
        prompts = CorpusPrompts(token_ids, args.seed)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    # Open exclusively before sending requests, so an output-path failure cannot
    # discard an expensive completed serving run or overwrite earlier evidence.
    with open(args.output_file, "x", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "type": "configuration",
                    "dataset_path": str(Path(args.dataset_path).resolve()),
                    "bounds": asdict(bounds),
                    "request_rate": (
                        args.request_rate if math.isfinite(args.request_rate) else "inf"
                    ),
                    "concurrency": args.concurrency,
                    "selected_rounds": len(rows),
                    "selected_sessions": len({row.session_id for row in rows}),
                    "max_rounds_per_session": args.max_rounds_per_session,
                    "seed": args.seed,
                    "content": (
                        "corpus_tokens"
                        if args.text_file
                        else "synthetic_vocabulary_tokens"
                    ),
                    "text_file": args.text_file,
                    "prefix_policy": "retain_prior_prompt_and_actual_generated_tokens",
                    "speculative_acceptance_representative": False,
                }
            )
            + "\n"
        )
        output.flush()
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as client:

            async def send(row):
                input_ids, reused = prompts.build(row)
                try:
                    result = await send_generate(client, args.base_url, row, input_ids)
                    result["success"] = result["output_length_matched"]
                    if result["success"]:
                        prompts.commit_output(row, input_ids, result.pop("output_ids"))
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    ValueError,
                    RuntimeError,
                ) as error:
                    result = {"success": False, "error": str(error)}
                output.write(
                    json.dumps(
                        {
                            "type": "request",
                            **asdict(row),
                            **result,
                            "synthetic_reused_input_tokens": reused,
                        }
                    )
                    + "\n"
                )
                output.flush()
                return result

            started = asyncio.get_running_loop().time()
            results = await replay_sessions(
                rows, send, concurrency=args.concurrency, request_rate=args.request_rate
            )
            elapsed = asyncio.get_running_loop().time() - started
        summary = summarize_results(results, elapsed)
        output.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
        return 1 if summary["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--text-file", help="UTF-8 corpus for TraceLab-style token reconstruction"
    )
    parser.add_argument("--token-pool-limit", type=int, default=100_000_000)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-rounds-per-session", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=math.inf)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--min-input-len", type=int, default=1)
    parser.add_argument("--max-input-len", type=int)
    parser.add_argument("--min-output-len", type=int, default=1)
    parser.add_argument("--max-output-len", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.request_timeout <= 0 or not math.isfinite(args.request_timeout):
        parser.error("request-timeout must be finite and positive")
    if args.token_pool_limit < 1:
        parser.error("token-pool-limit must be positive")
    raise SystemExit(asyncio.run(run_trace(args)))


if __name__ == "__main__":
    main()
