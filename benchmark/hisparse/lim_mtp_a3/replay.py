"""Measure MTP streaming requests using frozen text-tokenized input.

SSE arrival gaps are client observations, not device step times. Native path
and fixed-batch scheduler timing must be established independently. The paper's
three-token assumption is never used to calculate measured TPOT or throughput.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def request_json(url, payload=None, timeout=30):
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def run_one(base_url, tokens, output_tokens, timeout, index, barrier):
    rid = f"lim-mtp-a3-{time.time_ns()}-{index}"
    request = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(
            {
                "rid": rid,
                "input_ids": tokens,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                },
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait(timeout=30)
    start = time.perf_counter()
    start_unix = time.time()
    events, raw_meta = [], []
    final_meta, text, error = {}, "", None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                final_meta = data.get("meta_info", {})
                now = time.perf_counter()
                raw_meta.append({"elapsed_s": now - start, "meta": final_meta})
                count = int(final_meta.get("completion_tokens", 0))
                if count and (not events or count > events[-1]["tokens"]):
                    events.append(
                        {
                            "elapsed_s": now - start,
                            "monotonic_s": now,
                            "tokens": count,
                            "verify_ct": final_meta.get("spec_verify_ct"),
                        }
                    )
                text = data.get("text", text)
                finish = final_meta.get("finish_reason")
                if isinstance(finish, dict) and finish.get("type") in (
                    "abort",
                    "error",
                ):
                    raise RuntimeError(str(finish))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.perf_counter() - start
    count = events[-1]["tokens"] if events else 0
    if error is None and count != output_tokens:
        error = f"expected {output_tokens} output tokens, received {count}"
    gaps = [
        {
            "seconds": current["elapsed_s"] - previous["elapsed_s"],
            "new_tokens": current["tokens"] - previous["tokens"],
            "verify_increment": (
                current["verify_ct"] - previous["verify_ct"]
                if current["verify_ct"] is not None
                and previous["verify_ct"] is not None
                else None
            ),
        }
        for previous, current in zip(events, events[1:])
    ]
    ttft = events[0]["elapsed_s"] if events else None
    return {
        "index": index,
        "rid": rid,
        "start_unix": start_unix,
        "start_monotonic_s": start,
        "input_tokens": len(tokens),
        "input_sha256": digest(tokens),
        "output_tokens": count,
        "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_text": text,
        "ttft_s": ttft,
        "e2e_s": duration,
        "conventional_tpot_s": (duration - ttft) / (count - 1)
        if events and count > 1
        else None,
        "after_first_chunk_tpot_s": (
            (events[-1]["elapsed_s"] - ttft) / (count - events[0]["tokens"])
            if events and count > events[0]["tokens"]
            else None
        ),
        "events": events,
        "chunk_gaps": gaps,
        "final_meta": final_meta,
        "raw_meta": raw_meta,
        "error": error,
    }


def summarize(rows, duration):
    completed = [row for row in rows if row["error"] is None]
    tokens = sum(row["output_tokens"] for row in rows)
    verify = sum(row["final_meta"].get("spec_verify_ct", 0) for row in completed)
    accepted = sum(
        row["final_meta"].get("spec_num_correct_drafts", 0) for row in completed
    )
    proposed = sum(
        row["final_meta"].get("spec_num_proposed_drafts", 0) for row in completed
    )
    have_spec = bool(completed) and all(
        "spec_verify_ct" in row["final_meta"] for row in completed
    )
    gaps = [gap["seconds"] for row in rows for gap in row["chunk_gaps"]]
    overlap = None
    if len(completed) == len(rows) and all(row["events"] for row in rows):
        start = max(row["events"][0]["monotonic_s"] for row in rows)
        end = min(row["events"][-1]["monotonic_s"] for row in rows)
        if end > start:
            observed_tokens = 0
            for row in rows:
                for previous, current in zip(row["events"], row["events"][1:]):
                    if start <= previous["monotonic_s"] < current["monotonic_s"] <= end:
                        observed_tokens += current["tokens"] - previous["tokens"]
            overlap = {
                "duration_s": end - start,
                "observed_tokens": observed_tokens,
                "observed_tps": observed_tokens / (end - start),
                "note": "Client overlap only; requires scheduler evidence before labeling fixed-BS decode.",
            }
    tpots = [
        row["conventional_tpot_s"]
        for row in completed
        if row["conventional_tpot_s"] is not None
    ]
    chunk_tpots = [
        row["after_first_chunk_tpot_s"]
        for row in completed
        if row["after_first_chunk_tpot_s"] is not None
    ]
    successful_tokens = sum(row["output_tokens"] for row in completed)
    step_histogram = {}
    single_verify_events = coalesced_verify_steps = observed_verify_steps = 0
    for row in completed:
        for gap in row["chunk_gaps"]:
            increment = gap.get("verify_increment")
            if increment is None or increment <= 0:
                continue
            observed_verify_steps += increment
            if increment == 1:
                emitted = str(gap["new_tokens"])
                step_histogram[emitted] = step_histogram.get(emitted, 0) + 1
                single_verify_events += 1
            else:
                coalesced_verify_steps += increment
    return {
        "requests": len(rows),
        "completed": len(completed),
        "duration_s": duration,
        "output_tokens": tokens,
        "successful_output_tokens": successful_tokens,
        "partial_failed_output_tokens": tokens - successful_tokens,
        "whole_request_output_tps": tokens / duration,
        "successful_request_output_tps": successful_tokens / duration,
        "mean_conventional_tpot_ms": statistics.mean(tpots) * 1000 if tpots else None,
        "mean_after_first_chunk_tpot_ms": (
            statistics.mean(chunk_tpots) * 1000 if chunk_tpots else None
        ),
        "mean_ttft_ms": statistics.mean(row["ttft_s"] for row in completed) * 1000
        if completed
        else None,
        "median_sse_chunk_gap_ms": statistics.median(gaps) * 1000 if gaps else None,
        "verify_count_request_sum": verify if have_spec else None,
        "actual_tokens_per_verify": sum(row["output_tokens"] for row in completed)
        / verify
        if have_spec and verify
        else None,
        "accepted_drafts": accepted if have_spec else None,
        "proposed_drafts": proposed if have_spec else None,
        "draft_accept_rate": accepted / proposed if have_spec and proposed else None,
        "observed_single_verify_output_tokens_histogram": step_histogram,
        "observed_single_verify_events": single_verify_events,
        "coalesced_verify_steps": coalesced_verify_steps,
        "verify_steps_without_a_counted_output_gap": (
            max(0, verify - observed_verify_steps) if have_spec else None
        ),
        "client_overlap": overlap,
        "device_step_ms": None,
        "paper_assumption_tpot_ms": None,
    }


def validate_server(server, algorithm):
    config = server.get("server_args", server)
    expected = {
        "tp_size": 16,
        "ep_size": 16,
        "dp_size": 1,
        "attn_dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "moe_dp_size": 1,
        "nnodes": 1,
        "node_rank": 0,
    }
    actual_algorithm = config.get("speculative_algorithm")
    # NEXTN is resolved to EAGLE for the checkpoint's embedded MTP layer.
    if actual_algorithm not in (
        ("NEXTN", "EAGLE") if algorithm == "NEXTN" else (None,)
    ):
        raise RuntimeError(f"unexpected speculative algorithm: {actual_algorithm}")
    if algorithm == "NEXTN":
        expected.update(
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"resolved configuration mismatch: {mismatches}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18898")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument(
        "--input-sha256",
        default="ecf2ddbf180fc07b82dd670ab53a54dc4d4f00c2028d79cc81164850c06b1383",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-tokens", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--mtp-off-control", action="store_true")
    args = parser.parse_args()
    if min(args.input_tokens, args.batch_size, args.output_tokens) <= 0:
        parser.error("lengths and batch size must be positive")
    if args.output.exists():
        parser.error("output exists; preserve prior measurements")
    actual_hash = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    if actual_hash != args.input_sha256:
        parser.error(f"frozen input file SHA mismatch: {actual_hash}")
    opener = gzip.open if args.requests.suffix == ".gz" else open
    inputs = []
    with opener(args.requests, "rt", encoding="utf-8") as stream:
        for line in stream:
            if len(inputs) >= args.batch_size:
                break
            tokens = json.loads(line)["token_ids"][: args.input_tokens]
            if len(tokens) != args.input_tokens:
                parser.error("frozen request is shorter than the requested prefix")
            inputs.append(tokens)
    if len(inputs) != args.batch_size:
        parser.error("not enough frozen requests")
    before = request_json(args.base_url + "/server_info")
    validate_server(before, None if args.mtp_off_control else "NEXTN")
    barrier = threading.Barrier(args.batch_size)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as pool:
        futures = [
            pool.submit(
                run_one,
                args.base_url,
                tokens,
                args.output_tokens,
                args.timeout,
                index,
                barrier,
            )
            for index, tokens in enumerate(inputs)
        ]
        rows = [future.result() for future in futures]
    duration = time.perf_counter() - start
    summary = summarize(rows, duration)
    after = request_json(args.base_url + "/server_info")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "input_file_sha256": actual_hash,
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "summary": summary,
                "server_before": before,
                "server_after": after,
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["completed"] != len(rows):
        raise SystemExit(1)
    if not args.mtp_off_control and summary["verify_count_request_sum"] is None:
        raise SystemExit("MTP acceptance metadata missing; metrics are incomplete")


if __name__ == "__main__":
    main()
