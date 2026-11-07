"""Capture full-batch decode on Ascend; profiled requests are not latency results.

Prepare fixtures once, reuse them across storage modes, and use a fresh tag for
each capture. The streaming barrier requires every request to have decoded
--warmup-tokens before profiling starts. This avoids timing chunked prefill or a
partially populated batch. MTP is deliberately not assumed by this client.
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import logging
import statistics
import threading
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def http(args, route, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        args.url + route, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        content_type = response.headers.get_content_type()
    if not raw:
        return None
    return json.loads(raw) if content_type == "application/json" else raw.decode()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def measure_full_batch(states, start):
    """Count tokens only while all requests remain active after the barrier."""
    end = min(s["events"][-1][0] for s in states)
    generated = 0
    intervals = []
    for state in states:
        before = max((n for t, n in state["events"] if t <= start), default=0)
        until = max((n for t, n in state["events"] if t < end), default=0)
        generated += until - before
        window = [(t, n) for t, n in state["events"] if start <= t < end]
        intervals.extend(
            (b[0] - a[0]) / (b[1] - a[1])
            for a, b in itertools.pairwise(window)
            if b[1] > a[1]
        )
    if end <= start or generated <= 0 or not intervals:
        raise RuntimeError("Empty common full-batch decode window")
    return {
        "start_unix": start,
        "end_unix": end,
        "output_tokens_in_window": generated,
        "output_tokens_per_second": generated / (end - start),
        "median_stream_token_interval_ms": 1000 * statistics.median(intervals),
        "scope": "Client streaming, all requests active after warmup; excludes prefill and batch drain",
    }


def prepare(args):
    from transformers import AutoTokenizer

    if args.fixtures.exists():
        raise RuntimeError("Fixtures already exist; preserve the original inputs")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    filler = tokenizer.encode(
        "Archived note: the blue river flows past the quiet village. "
        * max(12000, max(args.input_lengths) // 8 + 1),
        add_special_tokens=False,
    )
    rows = []
    for length in args.input_lengths:
        for variant in range(args.variants):
            needle = f"LIM{7319 + variant}"
            head = tokenizer.encode(
                f"Archive number {variant}. Secret verification code: {needle}.\n",
                add_special_tokens=False,
            )
            # Preserve the native GLM assistant suffix, rather than hand writing
            # chat markers. Only the user message's filler is expanded.
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": "PLACEHOLDER"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            before, after = prefix.split("PLACEHOLDER")
            first = tokenizer.encode(before, add_special_tokens=False) + head
            last = tokenizer.encode(
                "\nReturn only the secret verification code from the beginning."
                + after,
                add_special_tokens=False,
            )
            count = length - len(first) - len(last)
            if not 0 <= count <= len(filler):
                raise RuntimeError("Filler cannot satisfy the requested token length")
            ids = first + filler[:count] + last
            assert len(ids) == length
            rows.append(
                {
                    "name": f"L{length}-v{variant}",
                    "length": length,
                    "variant": variant,
                    "needle": needle,
                    "input_ids": ids,
                    "input_sha256": hashlib.sha256(
                        json.dumps(ids).encode()
                    ).hexdigest(),
                }
            )
    save(args.fixtures, rows)
    print("Prepared", len(rows), "fixed token-ID fixtures", flush=True)


def smoke(args):
    rows = json.loads(args.fixtures.read_text())
    results = []
    lengths = sorted({r["length"] for r in rows})
    for length in lengths[:2]:
        row = next(r for r in rows if r["length"] == length and r["variant"] == 0)
        response = http(
            args,
            "/generate",
            {
                "input_ids": row["input_ids"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=900,
        )
        passed = row["needle"] in response.get("text", "")
        results.append({"length": length, "passed": passed, "response": response})
        save(args.workdir / f"{args.tag}-smoke.json", results)
        if not passed:
            raise RuntimeError(f"Needle smoke failed at {length}: {response}")
    print("Needle smoke checks passed", flush=True)


def capture(args):
    output = args.workdir / f"{args.tag}-capture.json"
    if output.exists():
        raise RuntimeError("Capture tag already exists; choose a fresh tag")
    rows = [
        row
        for row in json.loads(args.fixtures.read_text())
        if row["length"] == args.length
    ][: args.batch]
    if len(rows) != args.batch:
        raise RuntimeError("Insufficient fixed fixtures for this batch")
    lock = threading.Lock()
    states = [
        {"rid": f"{args.tag}-{i}", "count": 0, "done": False, "events": []}
        for i in range(args.batch)
    ]
    record = {
        "tag": args.tag,
        "diagnostic_only": not args.no_profile,
        "started_unix": time.time(),
        "input_length": args.length,
        "batch": args.batch,
        "requested_output_tokens": args.tokens,
        "requested_profile_steps": args.steps,
        "warmup_tokens": args.warmup_tokens,
        "input_sha256": [r["input_sha256"] for r in rows],
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "server_info": http(args, "/get_server_info"),
        "states": states,
    }

    def stream(index):
        state = states[index]
        payload = {
            "rid": state["rid"],
            "input_ids": rows[index]["input_ids"],
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.tokens,
                "ignore_eos": True,
            },
        }
        request = urllib.request.Request(
            args.url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                for raw in response:
                    if not raw.startswith(b"data: ") or raw[6:].strip() == b"[DONE]":
                        continue
                    event = json.loads(raw[6:])
                    meta = event.get("meta_info", {})
                    if "error" in event:
                        raise RuntimeError(str(event))
                    with lock:
                        state["count"] = meta.get("completion_tokens", 0)
                        state["events"].append([time.time(), state["count"]])
                        state["last_response"] = event
            with lock:
                state["done"] = True
        except Exception as error:
            with lock:
                state["error"] = repr(error)
                state["done"] = True
            raise

    failure = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch) as pool:
        futures = [pool.submit(stream, i) for i in range(args.batch)]
        try:
            deadline = time.monotonic() + args.barrier_timeout
            while True:
                with lock:
                    snapshot = [dict(s) for s in states]
                if any(s["done"] for s in snapshot):
                    raise RuntimeError("A request ended before the full-batch barrier")
                if all(s["count"] >= args.warmup_tokens for s in snapshot):
                    record["arm_states"] = [
                        {k: s[k] for k in ("rid", "count", "done")} for s in snapshot
                    ]
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Full-batch decode barrier timed out")
                time.sleep(0.1)
            record["full_batch_barrier_unix"] = time.time()
            profile = {
                "output_dir": str(args.workdir / "profiles" / args.tag / "decode"),
                "num_steps": args.steps,
                "activities": ["CPU", "GPU"],
                "with_stack": args.with_stack,
                "record_shapes": False,
                "detailed_annotations": True,
                "profile_id": args.tag + "-decode",
                "merge_profiles": False,
            }
            if not args.no_profile:
                record["profile_payload"] = profile
                record["arm_unix"] = time.time()
                record["profile_response"] = http(
                    args, "/start_profile", profile, timeout=300
                )
                print("PROFILE_ARMED", record["arm_states"], flush=True)
            else:
                print("FULL_BATCH_DECODE_BARRIER", record["arm_states"], flush=True)
            for future in futures:
                future.result(timeout=3600)
            if any(s["count"] != args.tokens for s in states):
                raise RuntimeError("Incomplete token generation; capture is invalid")
            record["status"] = (
                "unprofiled_requests_complete"
                if args.no_profile
                else "requests_complete_trace_requires_validation"
            )
            if args.no_profile:
                # Restrict measurement to the common active decode interval.
                # Prefix prefill and the batch-draining tail are excluded. The
                # measured count uses cumulative token counters, never ITL/3.
                record["full_batch_decode_measurement"] = measure_full_batch(
                    states, record["full_batch_barrier_unix"]
                )
        except Exception as error:
            logger.exception("Capture failed; aborting only this client's requests")
            failure = error
            record["status"] = "invalid"
            record["error"] = repr(error)
            for state in states:
                try:
                    http(args, "/abort_request", {"rid": state["rid"]}, timeout=10)
                except Exception as abort_error:
                    logger.exception("Could not abort request %s", state["rid"])
                    record.setdefault("abort_errors", []).append(repr(abort_error))
        finally:
            record["finished_unix"] = time.time()
            save(output, record)
    print(record["status"], flush=True)
    if failure is not None:
        raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "capture"))
    parser.add_argument("--url", default="http://127.0.0.1:18732")
    parser.add_argument("--model", default="/models/GLM-5.2-w4a8")
    parser.add_argument("--workdir", type=Path, default=Path("/task/results"))
    parser.add_argument(
        "--fixtures", type=Path, default=Path("/task/results/profile-fixtures.json")
    )
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--length", type=int, default=64000)
    parser.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=[512, 8192, 16000, 64000],
        help="Token lengths to freeze during prepare",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=16,
        help="Distinct token-ID fixtures per prepared length",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-tokens", type=int, default=10)
    parser.add_argument("--barrier-timeout", type=int, default=2400)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Measure a separate full-batch decode request without enabling the profiler",
    )
    parser.add_argument(
        "--followup-rounds",
        type=int,
        default=0,
        help="Independent unprofiled repeats after a diagnostic capture",
    )
    args = parser.parse_args()
    if (
        args.length <= 0
        or args.variants <= 0
        or any(n <= 0 for n in args.input_lengths)
    ):
        parser.error("Lengths and fixture count must be positive")
    if not 1 <= args.batch <= 16 or args.tokens <= args.warmup_tokens + args.steps + 2:
        parser.error(
            "Batch must be 1..16 and output must leave room for warmup and profiling"
        )
    args.workdir.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "smoke": smoke, "capture": capture}[args.phase](args)
    if args.phase == "capture" and not args.no_profile:
        for index in range(args.followup_rounds):
            followup = argparse.Namespace(**vars(args))
            followup.no_profile = True
            followup.tag = f"{args.tag}-unprofiled-r{index + 1}"
            capture(followup)


if __name__ == "__main__":
    main()
