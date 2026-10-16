"""Validation and summary of one evalscope multi-turn run, and the results table.

usage: summarize_rung.py validate --evalscope-dir DIR --dataset-stats FILE --history-tokenizer DIR --offset N
                                  --number N --turns N --osl N --context-length N [--loads FILE --cc N] --out FILE
       summarize_rung.py row      --rung-dir DIR --cc N --status S --expected-requests N --n-gpus N --out FILE
       summarize_rung.py table    --results-dir DIR

`validate` reads evalscope's raw per-request database and exits 1 unless all of the following hold:
* every expected conversation ran all its turns exactly once, and every request succeeded;
* each request produced exactly --osl completion tokens, as the server's usage reports;
* turn 1 of each conversation has the prompt-token count that the builder recorded;
* each later turn carries the previous turn's streamed text as its assistant history, minus special-token
  spellings (see evalscope_client.py); an empty stream replays as an empty message, which is counted, not failed;
* no prompt outgrows the context.

`row` prints the rung's final status and exits 1 when it cannot confirm OK: validation failed, or evalscope's
summary or percentiles are missing, incomplete or without the headline metrics.

The databases hold pickled stream chunks (evalscope's format). Only read databases that this package's own client
wrote; never point this at downloaded files.
"""

import argparse
import base64
import glob
import hashlib
import itertools
import json
import math
import os
import pickle
import sqlite3
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_tool import special_token_pattern  # noqa: E402

RECORD_KEY = "openhands_replay"  # evalscope_client.py's per-request usage record
N_PERCENTILES = ("50%", "90%", "99%")
REQUIRED_METRICS = (
    "out_tok_s",
    "total_tok_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "tpot_p90_ms",
)
COLUMNS = (
    "cc",
    "status",
    "conversations",
    "requests",
    "failed",
    "out_tok_s",
    "out_tok_s_per_gpu",
    "total_tok_s",
    "total_tok_s_per_gpu",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p99_ms",
    "interactivity_p50",
    "interactivity_p90",
    "interactivity_at_tpot_p99",
    "actual_cache_hit_pct",
    "evalscope_kv_hit_pct",
    "evalscope_kv_hit_basis",
    "avg_input_tokens",
    "avg_output_tokens",
    "duration_s",
    "max_server_reqs",
)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def find_run_dir(evalscope_dir):
    """evalscope writes <outputs-dir>/<name>/parallel_<p>_number_<n>/; exactly one is expected."""
    dirs = glob.glob(os.path.join(evalscope_dir, "*", "parallel_*"))
    return dirs[0] if len(dirs) == 1 else None


def decode_stream(response_messages_b64):
    """(streamed text, content, reasoning, usage record) of one request.

    The streamed text is what evalscope accumulates per chunk: delta.content + delta.reasoning_content.
    """
    chunks = pickle.loads(base64.b64decode(response_messages_b64))
    content, reasoning, record = [], [], None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if RECORD_KEY in chunk:
            record = chunk[RECORD_KEY]
        choices = chunk.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content.append(delta.get("content") or "")
        reasoning.append(delta.get("reasoning_content") or "")
    streamed = "".join(c + r for c, r in zip(content, reasoning, strict=True))
    return streamed, "".join(content), "".join(reasoning), record


def read_requests(db_path):
    """One record per request row: conversation key, turn, token counts, carried history and streamed text."""
    records = []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT request, success, response_messages, prompt_tokens, completion_tokens FROM result"
        ).fetchall()
    finally:
        con.close()
    for request, success, response_b64, prompt_tokens, completion_tokens in rows:
        messages = json.loads(request)["messages"]
        streamed, content, reasoning, usage = (
            decode_stream(response_b64) if success else ("", "", "", None)
        )
        records.append(
            {
                "conv": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
                "turn": sum(m["role"] == "user" for m in messages),
                "success": bool(success),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "assistant_history": [
                    m["content"] for m in messages if m["role"] == "assistant"
                ],
                "streamed": streamed,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "usage": usage,
            }
        )
    return records


def _expected_conversations(stats, offset, number):
    # evalscope's swe_smith loader rotates the list by --dataset-offset and takes the first --number.
    n = len(stats)
    return [stats[(offset + i) % n] for i in range(number)]


def _check_counts(records, by_sha, number, turns, osl):
    failures = []
    if len(by_sha) != number:
        failures.append(
            f"the dataset window holds {len(by_sha)} distinct conversations, expected {number}"
        )
    if len(records) != number * turns:
        failures.append(f"{len(records)} requests recorded, expected {number * turns}")
    failed = sum(not r["success"] for r in records)
    if failed:
        failures.append(f"{failed} requests failed")
    bad_osl = [
        r["completion_tokens"]
        for r in records
        if r["success"] and r["completion_tokens"] != osl
    ]
    if bad_osl:
        failures.append(
            f"{len(bad_osl)} requests did not produce {osl} completion tokens (e.g. {bad_osl[:5]})"
        )
    no_record = sum(r["success"] and r["usage"] is None for r in records)
    if no_record:
        failures.append(
            f"{no_record} requests lack the replay plugin's usage record (not run through evalscope_client.py)"
        )
    return failures


def _conversation_turns(records, by_sha, turns):
    """({conversation: [turn 1..N records]} for complete known conversations, failures)."""
    convs = {}
    for r in records:
        convs.setdefault(r["conv"], {}).setdefault(r["turn"], []).append(r)
    failures = []
    unknown = set(convs) - set(by_sha)
    if unknown:
        failures.append(
            f"{len(unknown)} conversations are not in the expected dataset window"
        )
    missing = set(by_sha) - set(convs)
    if missing:
        failures.append(f"{len(missing)} expected conversations never ran")
    complete = {
        k: [t[i][0] for i in range(1, turns + 1)]
        for k, t in convs.items()
        if k not in unknown
        and sorted(t) == list(range(1, turns + 1))
        and all(len(v) == 1 for v in t.values())
    }
    if len(complete) != len(convs) - len(unknown):
        failures.append(
            f"{len(convs) - len(unknown) - len(complete)} conversations do not have turns 1..{turns} exactly once"
        )
    return complete, len(convs), failures


def _spread(values):
    return (
        {"min": min(values), "median": statistics.median(values), "max": max(values)}
        if values
        else None
    )


def _check_sequences(complete, by_sha, osl, context_length, special):
    failures = []
    (
        turn1_mismatch,
        history_bad,
        history_empty,
        non_increasing,
        growth,
        history_chars,
        max_prompt,
    ) = [], 0, 0, 0, [], [], 0
    for key, seq in complete.items():
        if seq[0]["prompt_tokens"] != by_sha[key]["first_turn_prompt_tokens"]:
            turn1_mismatch.append(
                (seq[0]["prompt_tokens"], by_sha[key]["first_turn_prompt_tokens"])
            )
        for prev, cur in itertools.pairwise(seq):
            if not cur["success"] or not prev["success"]:
                continue
            hist = cur["assistant_history"]
            if len(hist) != cur["turn"] - 1 or hist[-1] != special.sub(
                "", prev["streamed"]
            ):
                history_bad += 1
            history_empty += not hist[-1]
            history_chars.append(len(hist[-1]))
            if cur["prompt_tokens"] <= prev["prompt_tokens"]:
                non_increasing += 1
            growth.append(cur["prompt_tokens"] - prev["prompt_tokens"])
        max_prompt = max([max_prompt] + [r["prompt_tokens"] or 0 for r in seq])
    if turn1_mismatch:
        failures.append(
            f"{len(turn1_mismatch)} conversations: server turn-1 prompt tokens differ from the builder's "
            f"(server, builder) e.g. {turn1_mismatch[:3]}; tokenizer or chat_template_kwargs differ"
        )
    if history_bad:
        failures.append(
            f"{history_bad} turns do not carry the previous turn's streamed text as history"
        )
    if non_increasing:
        failures.append(
            f"{non_increasing} turns have no more prompt tokens than the turn before"
        )
    if max_prompt + osl > context_length:
        failures.append(
            f"max prompt {max_prompt} + {osl} exceeds context {context_length}"
        )
    details = {
        "turn1_prompt_tokens_checked": len(complete),
        "turn1_prompt_tokens_mismatched": len(turn1_mismatch),
        "max_prompt_tokens": max_prompt,
        "prompt_growth_per_turn": _spread(growth),
        "replayed_history_chars": _spread(history_chars),
        # The server sometimes streams no text for a whole turn; evalscope then replays an empty message.
        "empty_replayed_history_turns": history_empty,
    }
    return failures, details


def server_cache_report(records):
    """Cached-token totals from the per-request usage records; a ratio only if every request reported them."""
    ok = [r for r in records if r["success"] and r["usage"] is not None]
    reported = [r for r in ok if r["usage"].get("cached_tokens") is not None]
    cached = sum(r["usage"]["cached_tokens"] for r in reported)
    prompt = sum(r["usage"]["prompt_tokens"] or 0 for r in reported)
    complete = bool(ok) and len(reported) == len(ok) and prompt > 0
    return {
        "requests": len(ok),
        "requests_reporting_cached_tokens": len(reported),
        "cached_tokens": cached,
        "prompt_tokens": prompt,
        "hit_pct": round(100 * cached / prompt, 2) if complete else None,
    }


def validate(records, stats, offset, number, turns, osl, context_length, special):
    """(failures, details) for one run."""
    by_sha = {
        s["system_sha256"]: s for s in _expected_conversations(stats, offset, number)
    }
    failures = _check_counts(records, by_sha, number, turns, osl)
    complete, n_convs, conv_failures = _conversation_turns(records, by_sha, turns)
    seq_failures, seq_details = _check_sequences(
        complete, by_sha, osl, context_length, special
    )
    ok = [r for r in records if r["success"]]
    details = {
        "requests": len(records),
        "failed": len(records) - len(ok),
        "conversations": n_convs,
        "expected_conversations": number,
        "turns": turns,
        **seq_details,
        "responses_with_special_token_text": sum(
            bool(special.search(r["streamed"])) for r in ok
        ),
        "server_cache": server_cache_report(records),
        "output_mix": {
            "content_and_reasoning": sum(
                r["content_chars"] > 0 and r["reasoning_chars"] > 0 for r in ok
            ),
            "reasoning_only": sum(
                r["content_chars"] == 0 and r["reasoning_chars"] > 0 for r in ok
            ),
            "content_only": sum(
                r["content_chars"] > 0 and r["reasoning_chars"] == 0 for r in ok
            ),
            "empty_text": sum(
                r["content_chars"] == 0 and r["reasoning_chars"] == 0 for r in ok
            ),
        },
    }
    return failures + conv_failures + seq_failures, details


def load_peak(loads_path):
    """Peak running+queued requests the server held while the ladder ran, from sweep_tool.py sample-loads."""
    peak, samples = 0, 0
    with open(loads_path) as f:
        for line in f:
            row = json.loads(line)
            if "running" in row:
                samples += 1
                peak = max(peak, row["running"] + row["waiting"])
    return peak, samples


def cmd_validate(args):
    run_dir = find_run_dir(args.evalscope_dir)
    if run_dir is None or not os.path.exists(
        os.path.join(run_dir, "benchmark_data.db")
    ):
        _write_json(
            args.out,
            {
                "ok": False,
                "failures": ["no evalscope run directory with benchmark_data.db"],
            },
        )
        print("validate: no evalscope output", file=sys.stderr)
        return 1
    failures, details = validate(
        records=read_requests(os.path.join(run_dir, "benchmark_data.db")),
        stats=_load_json(args.dataset_stats)["conversation_stats"],
        offset=args.offset,
        number=args.number,
        turns=args.turns,
        osl=args.osl,
        context_length=args.context_length,
        special=special_token_pattern(args.history_tokenizer),
    )
    if args.loads and os.path.exists(args.loads):
        peak, samples = load_peak(args.loads)
        details["server_peak_requests"] = peak
        details["load_samples"] = samples
        if peak > args.cc:
            failures.append(
                f"the server held {peak} requests at once, more than the client's {args.cc}: foreign load"
            )
    result = {"ok": not failures, "failures": failures, "run_dir": run_dir, **details}
    _write_json(args.out, result)
    for f in failures:
        print(f"validate: {f}", file=sys.stderr)
    print(
        f"validate: {'OK' if not failures else 'INVALID'} "
        f"({details['requests']} requests, {details['conversations']} conversations)"
    )
    return 0 if not failures else 1


def _percentile(rows, which, key):
    for row in rows:
        if row.get("Percentiles") == which:
            return row.get(key)
    return None


def _per(value, n):
    return round(value / n, 1) if isinstance(value, (int, float)) else None


def rung_metrics(summary, percentiles, validation, n_gpus):
    """Headline metrics of one rung from evalscope's summary and percentile files and the raw validation."""
    tpot = {p: _percentile(percentiles, p, "TPOT (ms)") for p in N_PERCENTILES}
    ttft = {p: _percentile(percentiles, p, "TTFT (ms)") for p in N_PERCENTILES}
    out_tok_s = summary.get("Output Throughput (tok/s)")
    total_tok_s = summary.get("Total Throughput (tok/s)")
    cache = validation.get("server_cache") or {}
    measured = cache.get("hit_pct") is not None
    return {
        "requests": summary.get("Total Requests"),
        "failed": summary.get("Failed Requests"),
        "out_tok_s": out_tok_s,
        "out_tok_s_per_gpu": _per(out_tok_s, n_gpus),
        "total_tok_s": total_tok_s,
        "total_tok_s_per_gpu": _per(total_tok_s, n_gpus),
        "ttft_p50_ms": ttft["50%"],
        "ttft_p90_ms": ttft["90%"],
        "ttft_p99_ms": ttft["99%"],
        "tpot_p50_ms": tpot["50%"],
        "tpot_p90_ms": tpot["90%"],
        "tpot_p99_ms": tpot["99%"],
        "interactivity_p50": round(1000 / tpot["50%"], 2) if tpot["50%"] else None,
        "interactivity_p90": round(1000 / tpot["90%"], 2) if tpot["90%"] else None,
        "interactivity_at_tpot_p99": round(1000 / tpot["99%"], 2)
        if tpot["99%"]
        else None,
        "actual_cache_hit_pct": cache["hit_pct"] if measured else "unavailable",
        # evalscope substitutes previous prompt + completion when the server reports no cached tokens.
        "evalscope_kv_hit_pct": summary.get("KV Cache Hit Rate (%)"),
        "evalscope_kv_hit_basis": "server cached_tokens"
        if measured
        else "client estimate, not a cache measurement",
        "avg_input_tokens": summary.get("Avg Input Tokens"),
        "avg_output_tokens": summary.get("Avg Output Tokens"),
        "duration_s": summary.get("Test Duration (s)"),
    }


def confirm_ok(metrics, validation, summary_found, expected_requests):
    """Reasons a rung that ran to the end still cannot be reported as OK."""
    reasons = []
    if not validation.get("ok"):
        reasons.append("raw validation did not pass")
    if not summary_found:
        reasons.append("evalscope summary or percentiles missing")
    if metrics["requests"] != expected_requests or metrics["failed"] != 0:
        reasons.append(
            f"evalscope summary has {metrics['requests']} requests / {metrics['failed']} failed, "
            f"expected {expected_requests} / 0"
        )
    missing = [
        k
        for k in REQUIRED_METRICS
        if not isinstance(metrics[k], (int, float))
        or not math.isfinite(metrics[k])
        or metrics[k] <= 0
    ]
    if missing:
        reasons.append(f"missing or non-positive metrics: {missing}")
    return reasons


def cmd_row(args):
    if args.n_gpus < 1:
        print(f"row: --n-gpus must be positive, got {args.n_gpus}", file=sys.stderr)
        return 2
    run_dir = find_run_dir(os.path.join(args.rung_dir, "ladder"))
    summary_path = os.path.join(run_dir, "benchmark_summary.json") if run_dir else ""
    pct_path = os.path.join(run_dir, "benchmark_percentile.json") if run_dir else ""
    summary_found = os.path.exists(summary_path) and os.path.exists(pct_path)
    summary = _load_json(summary_path) if os.path.exists(summary_path) else {}
    percentiles = _load_json(pct_path) if os.path.exists(pct_path) else []
    validation_path = os.path.join(args.rung_dir, "validation.json")
    validation = _load_json(validation_path) if os.path.exists(validation_path) else {}
    metrics = rung_metrics(
        summary=summary,
        percentiles=percentiles,
        validation=validation,
        n_gpus=args.n_gpus,
    )
    status, reasons = args.status, []
    if status == "OK":
        reasons = confirm_ok(metrics, validation, summary_found, args.expected_requests)
        if reasons:
            status = "SUMMARY_FAIL"
    row = {
        "cc": args.cc,
        "status": status,
        "status_reasons": reasons,
        "conversations": validation.get("conversations"),
        "max_server_reqs": validation.get("server_peak_requests"),
        **metrics,
        "validation_failures": validation.get("failures", []),
        "output_mix": validation.get("output_mix"),
        "responses_with_special_token_text": validation.get(
            "responses_with_special_token_text"
        ),
        "server_cache": validation.get("server_cache"),
        "prompt_growth_per_turn": validation.get("prompt_growth_per_turn"),
        "replayed_history_chars": validation.get("replayed_history_chars"),
        "empty_replayed_history_turns": validation.get("empty_replayed_history_turns"),
        "n_gpus": args.n_gpus,
    }
    _write_json(args.out, row)
    for r in reasons:
        print(f"row: c={args.cc} not OK: {r}", file=sys.stderr)
    print(status)
    return 0 if status == args.status else 1


def cmd_table(args):
    rows = [
        _load_json(path)
        for path in glob.glob(os.path.join(args.results_dir, "c*", "rung.json"))
    ]
    rows.sort(key=lambda r: -r["cc"])
    tsv = os.path.join(args.results_dir, "results.tsv")
    with open(f"{tsv}.tmp", "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write(
                "\t".join("" if r.get(c) is None else str(r.get(c)) for c in COLUMNS)
                + "\n"
            )
    os.replace(f"{tsv}.tmp", tsv)
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print(
            "| "
            + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in COLUMNS)
            + " |"
        )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--evalscope-dir", required=True)
    p.add_argument("--dataset-stats", required=True)
    p.add_argument("--history-tokenizer", required=True)
    p.add_argument("--offset", type=int, required=True)
    p.add_argument("--number", type=int, required=True)
    p.add_argument("--turns", type=int, required=True)
    p.add_argument("--osl", type=int, required=True)
    p.add_argument("--context-length", type=int, required=True)
    p.add_argument("--loads", default=None)
    p.add_argument("--cc", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("row")
    p.add_argument("--rung-dir", required=True)
    p.add_argument("--cc", type=int, required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--expected-requests", type=int, required=True)
    p.add_argument("--n-gpus", type=int, required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("table")
    p.add_argument("--results-dir", required=True)

    args = parser.parse_args()
    sys.exit(
        {"validate": cmd_validate, "row": cmd_row, "table": cmd_table}[args.cmd](args)
    )


if __name__ == "__main__":
    main()
