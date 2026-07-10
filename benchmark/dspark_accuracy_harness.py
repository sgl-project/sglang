"""Accuracy harness for DSpark speculative decoding.

This script is intentionally server-agnostic. Run it in three phases:

1. prepare: build a fixed prompt JSONL.
2. collect: query one running SGLang server and dump outputs/meta.
3. compare: compare a target-only run against a DSpark run.

Example:
  python3 benchmark/dspark_accuracy_harness.py prepare \
    --model-path zai-org/GLM-5.2-FP8 \
    --output /tmp/glm52_prompts_256.jsonl \
    --num-samples 256

  python3 benchmark/dspark_accuracy_harness.py collect \
    --base-url http://127.0.0.1:30099 \
    --prompts /tmp/glm52_prompts_256.jsonl \
    --output /tmp/glm52_dspark_256.jsonl

  python3 benchmark/dspark_accuracy_harness.py compare \
    --target /tmp/glm52_target_256.jsonl \
    --spec /tmp/glm52_dspark_256.jsonl \
    --summary-output /tmp/glm52_compare_256.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "mgoin/GLM-5.2-FP8-magpie-ultrachat"
DEFAULT_MODEL = "zai-org/GLM-5.2-FP8"


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_text(stable_json(data))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def to_messages(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for turn in conversations:
        role = turn.get("from")
        content = turn.get("value", "")
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            break
    return messages


def load_ultrachat_prompts(args) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    rows = []
    for row in ds:
        messages = to_messages(row["conversations"])
        if not messages:
            continue
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if args.disable_thinking:
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
        rows.append(
            {
                "idx": len(rows),
                "id": row.get("id"),
                "source": row.get("source"),
                "dataset": args.dataset,
                "prompt_mode": "chat_no_thinking"
                if args.disable_thinking
                else "chat_default",
                "prompt_tokens_offline": len(tokenizer.encode(text)),
                "text": text,
            }
        )
        if len(rows) >= args.num_samples:
            break
    return rows


def command_prepare(args) -> None:
    rows = load_ultrachat_prompts(args)
    if len(rows) < args.num_samples:
        raise RuntimeError(
            f"Only prepared {len(rows)} prompts, requested {args.num_samples}."
        )
    write_jsonl(args.output, rows)
    summary = {
        "output": args.output,
        "dataset": args.dataset,
        "num_prompts": len(rows),
        "min_prompt_tokens": min(r["prompt_tokens_offline"] for r in rows),
        "max_prompt_tokens": max(r["prompt_tokens_offline"] for r in rows),
        "mean_prompt_tokens": statistics.fmean(
            r["prompt_tokens_offline"] for r in rows
        ),
    }
    print(json.dumps(summary, sort_keys=True))


def post_generate(base_url: str, text: str, args) -> dict[str, Any]:
    body = make_generate_payload(text, args)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_generate_payload(text: str, args) -> dict[str, Any]:
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }
    if args.sampling_seed is not None:
        sampling_params["sampling_seed"] = args.sampling_seed
    return {
        "text": text,
        "sampling_params": sampling_params,
        "return_meta_info": True,
        "return_logprob": args.return_logprob,
        "return_prompt_token_ids": args.return_prompt_token_ids,
    }


def request_signature(row: dict[str, Any], args) -> dict[str, Any]:
    payload = make_generate_payload(row["text"], args)
    return {
        "prompt_sha256": sha256_text(row["text"]),
        "prompt_token_count_offline": row.get("prompt_tokens_offline"),
        "payload_sha256": sha256_json(payload),
        "payload": payload,
    }


def command_collect(args) -> None:
    if (
        args.temperature != 0.0
        and args.sampling_seed is None
        and not args.allow_nondeterministic_sampling
    ):
        raise ValueError(
            "Nonzero temperature needs --sampling-seed for exact-match accuracy runs."
        )
    prompts = read_jsonl(args.prompts)
    if args.start_idx is not None:
        prompts = [row for row in prompts if row["idx"] >= args.start_idx]
    if args.end_idx is not None:
        prompts = [row for row in prompts if row["idx"] < args.end_idx]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    output_path = Path(args.output)
    if output_path.exists() and not args.resume:
        output_path.unlink()

    done = set()
    if output_path.exists() and args.resume:
        done = {r["idx"] for r in read_jsonl(output_path) if r.get("ok") and "idx" in r}

    started = time.perf_counter()
    pending_prompts = [row for row in prompts if row["idx"] not in done]
    if args.concurrency <= 1:
        for row in pending_prompts:
            record = collect_one(row, args)
            append_jsonl(output_path, record)
            print_collect_record(record, args)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(collect_one, row, args) for row in pending_prompts
            ]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                append_jsonl(output_path, record)
                print_collect_record(record, args)

    rows = read_jsonl(output_path)
    summary = summarize_run(rows)
    summary.update(
        {
            "run_label": args.run_label,
            "output": str(output_path),
            "elapsed_s": time.perf_counter() - started,
        }
    )
    print(json.dumps(summary, sort_keys=True))


def collect_one(row: dict[str, Any], args) -> dict[str, Any]:
    idx = row["idx"]
    request_started = time.perf_counter()
    record = {
        "run_label": args.run_label,
        "idx": idx,
        "id": row.get("id"),
        "source": row.get("source"),
        "dataset": row.get("dataset"),
        "prompt_mode": row.get("prompt_mode"),
        "prompt_tokens_offline": row.get("prompt_tokens_offline"),
        "request": request_signature(row, args),
    }
    for attempt in range(args.retries + 1):
        try:
            obj = post_generate(args.base_url, row["text"], args)
            meta = obj.get("meta_info") or {}
            output_ids = obj.get("output_ids")
            prompt_token_ids = obj.get("prompt_token_ids")
            record.update(
                {
                    "ok": True,
                    "elapsed_s": time.perf_counter() - request_started,
                    "text": obj.get("text"),
                    "output_ids": output_ids,
                    "output_id_count": len(output_ids)
                    if isinstance(output_ids, list)
                    else None,
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_token_id_count": len(prompt_token_ids)
                    if isinstance(prompt_token_ids, list)
                    else None,
                    "prompt_token_ids_sha256": sha256_json(prompt_token_ids)
                    if isinstance(prompt_token_ids, list)
                    else None,
                    "meta_info": meta,
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens"),
                }
            )
            break
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            record.update(
                {
                    "ok": False,
                    "elapsed_s": time.perf_counter() - request_started,
                    "error": repr(e),
                    "attempt": attempt,
                }
            )
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep_s)
    return record


def print_collect_record(record: dict[str, Any], args) -> None:
    idx = record["idx"]
    if args.print_records:
        print(json.dumps(record, sort_keys=True), flush=True)
    else:
        print(
            json.dumps(
                {
                    "idx": idx,
                    "ok": record.get("ok"),
                    "elapsed_s": record.get("elapsed_s"),
                    "completion_tokens": record.get("completion_tokens"),
                    "spec_accept_length": (record.get("meta_info") or {}).get(
                        "spec_accept_length"
                    ),
                    "spec_accept_rate": (record.get("meta_info") or {}).get(
                        "spec_accept_rate"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    metas = [r.get("meta_info") or {} for r in ok]
    completion = sum(m.get("completion_tokens") or 0 for m in metas)
    spec_metas = [m for m in metas if (m.get("spec_verify_ct") or 0) > 0]
    spec_completion = sum(m.get("completion_tokens") or 0 for m in spec_metas)
    verify = sum(m.get("spec_verify_ct") or 0 for m in spec_metas)
    correct = sum(m.get("spec_num_correct_drafts") or 0 for m in spec_metas)
    proposed = sum(m.get("spec_num_proposed_drafts") or 0 for m in spec_metas)
    accept_lengths = [
        m["spec_accept_length"] for m in metas if m.get("spec_accept_length") is not None
    ]
    accept_rates = [
        m["spec_accept_rate"] for m in metas if m.get("spec_accept_rate") is not None
    ]
    summary = {
        "requests": len(rows),
        "ok_requests": len(ok),
        "error_requests": len(rows) - len(ok),
        "completion_tokens": completion,
        "spec_metric_rows": len(spec_metas),
        "spec_completion_tokens": spec_completion,
    }
    if verify:
        summary["aggregate_accept_length"] = spec_completion / verify
    if proposed:
        summary["aggregate_accept_rate"] = correct / proposed
    if accept_lengths:
        summary["mean_accept_length"] = statistics.fmean(accept_lengths)
        summary["min_accept_length"] = min(accept_lengths)
        summary["max_accept_length"] = max(accept_lengths)
        summary["p10_accept_length"] = quantile(accept_lengths, 10)
        summary["p50_accept_length"] = quantile(accept_lengths, 50)
        summary["p90_accept_length"] = quantile(accept_lengths, 90)
    if accept_rates:
        summary["mean_accept_rate"] = statistics.fmean(accept_rates)
        summary["min_accept_rate"] = min(accept_rates)
        summary["max_accept_rate"] = max(accept_rates)
        summary["p10_accept_rate"] = quantile(accept_rates, 10)
        summary["p50_accept_rate"] = quantile(accept_rates, 50)
        summary["p90_accept_rate"] = quantile(accept_rates, 90)
        summary["accept_rate_buckets"] = accept_rate_buckets(accept_rates)
    return summary


def quantile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    pos = (len(ordered) - 1) * pct / 100
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    upper_weight = pos - lower
    return ordered[lower] * (1 - upper_weight) + ordered[upper] * upper_weight


def accept_rate_buckets(values: list[float]) -> dict[str, int]:
    buckets = {"<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0, ">=0.85": 0}
    for value in values:
        if value < 0.3:
            buckets["<0.3"] += 1
        elif value < 0.5:
            buckets["0.3-0.5"] += 1
        elif value < 0.7:
            buckets["0.5-0.7"] += 1
        elif value < 0.85:
            buckets["0.7-0.85"] += 1
        else:
            buckets[">=0.85"] += 1
    return buckets


def ranked_spec_examples(
    rows: list[dict[str, Any]], *, high: bool, limit: int
) -> list[dict[str, Any]]:
    ok = [
        row
        for row in rows
        if row.get("ok")
        and (row.get("meta_info") or {}).get("spec_accept_rate") is not None
    ]
    ranked = sorted(
        ok,
        key=lambda row: (row.get("meta_info") or {}).get("spec_accept_rate"),
        reverse=high,
    )
    examples = []
    for row in ranked[:limit]:
        meta = row.get("meta_info") or {}
        examples.append(
            {
                "idx": row.get("idx"),
                "id": row.get("id"),
                "source": row.get("source"),
                "prompt_tokens": row.get("prompt_tokens_offline"),
                "completion_tokens": row.get("completion_tokens"),
                "elapsed_s": row.get("elapsed_s"),
                "spec_accept_length": meta.get("spec_accept_length"),
                "spec_accept_rate": meta.get("spec_accept_rate"),
                "spec_verify_ct": meta.get("spec_verify_ct"),
            }
        )
    return examples


def command_run_summary(args) -> None:
    rows = read_jsonl(args.input)
    summary = summarize_run(rows)
    summary.update(
        {
            "input": args.input,
            "low_accept_examples": ranked_spec_examples(
                rows, high=False, limit=args.max_examples
            ),
            "high_accept_examples": ranked_spec_examples(
                rows, high=True, limit=args.max_examples
            ),
        }
    )
    failures = []
    if args.min_ok_requests is not None and summary["ok_requests"] < args.min_ok_requests:
        failures.append(
            f"ok_requests {summary['ok_requests']} is below {args.min_ok_requests}"
        )
    if args.min_aggregate_accept_rate is not None and (
        summary.get("aggregate_accept_rate") is None
        or summary["aggregate_accept_rate"] < args.min_aggregate_accept_rate
    ):
        failures.append(
            "aggregate_accept_rate "
            f"{summary.get('aggregate_accept_rate')} is below "
            f"{args.min_aggregate_accept_rate}"
        )
    if args.min_aggregate_accept_length is not None and (
        summary.get("aggregate_accept_length") is None
        or summary["aggregate_accept_length"] < args.min_aggregate_accept_length
    ):
        failures.append(
            "aggregate_accept_length "
            f"{summary.get('aggregate_accept_length')} is below "
            f"{args.min_aggregate_accept_length}"
        )
    summary["verdict"] = {"passed": not failures, "failures": failures}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    if args.fail_on_verdict and failures:
        raise SystemExit(1)


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def matched_prefix_from_comparison(item: dict[str, Any]) -> int | None:
    if item["token_exact"] is None:
        return None
    if item["first_mismatch"] is not None:
        return item["first_mismatch"]
    lengths = [v for v in (item["target_len"], item["spec_len"]) if v is not None]
    return min(lengths) if lengths else None


def command_compare(args) -> None:
    target_rows = {r["idx"]: r for r in read_jsonl(args.target)}
    spec_rows = {r["idx"]: r for r in read_jsonl(args.spec)}
    common = sorted(set(target_rows) & set(spec_rows))
    comparisons = []
    mismatches = []
    skipped = []
    signature_mismatches = []
    prompt_token_mismatches = []
    for idx in common:
        target = target_rows[idx]
        spec = spec_rows[idx]
        if not target.get("ok") or not spec.get("ok"):
            skipped.append(
                {
                    "idx": idx,
                    "id": target.get("id") or spec.get("id"),
                    "target_ok": bool(target.get("ok")),
                    "spec_ok": bool(spec.get("ok")),
                    "target_error": target.get("error"),
                    "spec_error": spec.get("error"),
                }
            )
            continue
        target_request = target.get("request") or {}
        spec_request = spec.get("request") or {}
        signature_comparable = bool(target_request) and bool(spec_request)
        signature_exact = (
            target_request.get("payload_sha256") == spec_request.get("payload_sha256")
            and target_request.get("prompt_sha256") == spec_request.get("prompt_sha256")
            if signature_comparable
            else None
        )
        if signature_exact is False:
            signature_mismatches.append(
                {
                    "idx": idx,
                    "target_request": target_request,
                    "spec_request": spec_request,
                }
            )
        target_ids = target.get("output_ids")
        spec_ids = spec.get("output_ids")
        can_compare_ids = isinstance(target_ids, list) and isinstance(spec_ids, list)
        token_exact = target_ids == spec_ids if can_compare_ids else None
        text_exact = target.get("text") == spec.get("text")
        mismatch_at = (
            first_mismatch(target_ids, spec_ids) if can_compare_ids else None
        )
        target_prompt_hash = target.get("prompt_token_ids_sha256")
        spec_prompt_hash = spec.get("prompt_token_ids_sha256")
        prompt_tokens_comparable = target_prompt_hash is not None and spec_prompt_hash is not None
        prompt_tokens_exact = (
            target_prompt_hash == spec_prompt_hash if prompt_tokens_comparable else None
        )
        if prompt_tokens_exact is False:
            prompt_token_mismatches.append(
                {
                    "idx": idx,
                    "target_prompt_token_id_count": target.get("prompt_token_id_count"),
                    "spec_prompt_token_id_count": spec.get("prompt_token_id_count"),
                }
            )
        item = {
            "idx": idx,
            "id": target.get("id"),
            "signature_exact": signature_exact,
            "prompt_tokens_exact": prompt_tokens_exact,
            "token_exact": token_exact,
            "text_exact": text_exact,
            "target_len": len(target_ids) if isinstance(target_ids, list) else None,
            "spec_len": len(spec_ids) if isinstance(spec_ids, list) else None,
            "first_mismatch": mismatch_at,
            "spec_accept_length": (spec.get("meta_info") or {}).get(
                "spec_accept_length"
            ),
            "spec_accept_rate": (spec.get("meta_info") or {}).get("spec_accept_rate"),
        }
        comparisons.append(item)
        if token_exact is False or (token_exact is None and not text_exact):
            mismatches.append(item)

    token_comparable = [c for c in comparisons if c["token_exact"] is not None]
    token_matches = [c for c in token_comparable if c["token_exact"]]
    matched_prefixes = [
        prefix
        for prefix in (matched_prefix_from_comparison(c) for c in token_comparable)
        if prefix is not None
    ]
    text_matches = [c for c in comparisons if c["text_exact"]]
    signature_comparable = [c for c in comparisons if c["signature_exact"] is not None]
    signature_matches = [c for c in signature_comparable if c["signature_exact"]]
    prompt_tokens_comparable = [
        c for c in comparisons if c["prompt_tokens_exact"] is not None
    ]
    prompt_token_matches = [c for c in prompt_tokens_comparable if c["prompt_tokens_exact"]]
    token_exact_match_rate = (
        len(token_matches) / len(token_comparable) if token_comparable else None
    )
    text_exact_match_rate = len(text_matches) / len(comparisons) if comparisons else None
    matched_prefix_mean = (
        statistics.fmean(matched_prefixes) if matched_prefixes else None
    )
    matched_prefix_median = (
        statistics.median(matched_prefixes) if matched_prefixes else None
    )
    failures = []
    if args.require_request_signature_match:
        if len(signature_comparable) != len(comparisons):
            failures.append(
                "request signatures are not available for all comparable rows"
            )
        elif len(signature_matches) != len(signature_comparable):
            failures.append("request signatures do not match")
    if args.require_prompt_token_match:
        if len(prompt_tokens_comparable) != len(comparisons):
            failures.append("prompt token hashes are not available for all rows")
        elif len(prompt_token_matches) != len(prompt_tokens_comparable):
            failures.append("prompt token hashes do not match")
    if args.min_token_exact_rate is not None and (
        token_exact_match_rate is None
        or token_exact_match_rate < args.min_token_exact_rate
    ):
        failures.append(
            f"token exact match rate {token_exact_match_rate} is below "
            f"{args.min_token_exact_rate}"
        )
    if args.min_matched_prefix_mean is not None and (
        matched_prefix_mean is None
        or matched_prefix_mean < args.min_matched_prefix_mean
    ):
        failures.append(
            f"matched prefix mean {matched_prefix_mean} is below "
            f"{args.min_matched_prefix_mean}"
        )
    summary = {
        "target": args.target,
        "spec": args.spec,
        "target_requests": len(target_rows),
        "spec_requests": len(spec_rows),
        "common_requests": len(common),
        "ok_common_requests": len(comparisons),
        "skipped_non_ok_common_requests": len(skipped),
        "skipped_non_ok_examples": skipped[: args.max_mismatch_examples],
        "target_only_missing": sorted(set(spec_rows) - set(target_rows))[:20],
        "spec_missing": sorted(set(target_rows) - set(spec_rows))[:20],
        "target_only_missing_count": len(set(spec_rows) - set(target_rows)),
        "spec_missing_count": len(set(target_rows) - set(spec_rows)),
        "request_signature_comparable_requests": len(signature_comparable),
        "request_signature_matches": len(signature_matches),
        "request_signature_mismatches": len(signature_comparable)
        - len(signature_matches),
        "request_signature_mismatch_examples": signature_mismatches[
            : args.max_mismatch_examples
        ],
        "prompt_token_comparable_requests": len(prompt_tokens_comparable),
        "prompt_token_exact_matches": len(prompt_token_matches),
        "prompt_token_mismatches": len(prompt_tokens_comparable)
        - len(prompt_token_matches),
        "prompt_token_mismatch_examples": prompt_token_mismatches[
            : args.max_mismatch_examples
        ],
        "token_comparable_requests": len(token_comparable),
        "token_exact_matches": len(token_matches),
        "token_mismatches": len(token_comparable) - len(token_matches),
        "token_exact_match_rate": token_exact_match_rate,
        "matched_prefix_mean": matched_prefix_mean,
        "matched_prefix_median": matched_prefix_median,
        "matched_prefix_min": min(matched_prefixes) if matched_prefixes else None,
        "matched_prefix_max": max(matched_prefixes) if matched_prefixes else None,
        "text_exact_matches": len(text_matches),
        "text_mismatches": len(comparisons) - len(text_matches),
        "text_exact_match_rate": text_exact_match_rate,
        "mismatch_examples": mismatches[: args.max_mismatch_examples],
        "target_run": summarize_run(list(target_rows.values())),
        "spec_run": summarize_run(list(spec_rows.values())),
        "verdict": {
            "passed": not failures,
            "failures": failures,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    if args.fail_on_verdict and failures:
        raise SystemExit(1)


def command_trace_summary(args) -> None:
    rows = read_jsonl(args.trace)
    failures = []
    pass_count = 0
    skipped_count = 0
    skipped_examples = []
    compact_count = 0
    folded_count = 0
    non_uniform_count = 0
    non_greedy_count = 0
    seeded_count = 0
    sampling_backends = {}
    simulated_count = 0
    padded_graph_count = 0
    request_rows = 0
    max_bs = 0
    verify_lens_sums = []
    graph_token_counts = []
    graph_padding_tokens = []
    for i, row in enumerate(rows):
        verdict = row.get("verdict") or {}
        row_failures = verdict.get("failures") or []
        row_skipped = verdict.get("skipped") or []
        if verdict.get("passed"):
            pass_count += 1
        if row_skipped:
            skipped_count += 1
            skipped_examples.append(
                {
                    "record": i,
                    "forward_ct": row.get("forward_ct"),
                    "skipped": row_skipped,
                }
            )
        if row_failures:
            failures.append(
                {
                    "record": i,
                    "forward_ct": row.get("forward_ct"),
                    "failures": row_failures,
                }
            )
        if row.get("run_compact"):
            compact_count += 1
        if row.get("folded_accept"):
            folded_count += 1
        if row.get("non_uniform_verify_lens"):
            non_uniform_count += 1
        if not row.get("all_greedy", True):
            non_greedy_count += 1
        sampling = row.get("sampling") or {}
        if sampling.get("seed_present"):
            seeded_count += 1
        backend = sampling.get("backend")
        if backend is not None:
            sampling_backends[backend] = sampling_backends.get(backend, 0) + 1
        if row.get("simulated_accept"):
            simulated_count += 1
        req_count = len(row.get("reqs") or [])
        request_rows += req_count
        max_bs = max(max_bs, req_count)
        if row.get("verify_lens_sum") is not None:
            verify_lens_sum = int(row["verify_lens_sum"])
            verify_lens_sums.append(verify_lens_sum)
            layout_graph_num_tokens = row.get("layout_graph_num_tokens")
            if layout_graph_num_tokens is not None:
                graph_num_tokens = int(layout_graph_num_tokens)
                graph_token_counts.append(graph_num_tokens)
                padding_tokens = graph_num_tokens - verify_lens_sum
                graph_padding_tokens.append(padding_tokens)
                if padding_tokens > 0:
                    padded_graph_count += 1

    gate_failures = []
    if args.require_records_min is not None and len(rows) < args.require_records_min:
        gate_failures.append(
            f"records {len(rows)} is below {args.require_records_min}"
        )
    if args.require_compact and compact_count == 0:
        gate_failures.append("no compact records were observed")
    if args.require_non_uniform_verify_lens and non_uniform_count == 0:
        gate_failures.append("no non-uniform verify_lens records were observed")
    if args.require_folded_accept and folded_count == 0:
        gate_failures.append("no folded accept records were observed")
    if args.require_non_greedy and non_greedy_count == 0:
        gate_failures.append("no non-greedy records were observed")
    if args.require_seeded_sampling and seeded_count == 0:
        gate_failures.append("no seeded sampling records were observed")
    if args.require_no_skipped and skipped_count:
        gate_failures.append(f"{skipped_count} skipped records were observed")
    if args.require_padded_graph and padded_graph_count == 0:
        gate_failures.append("no padded graph records were observed")

    summary = {
        "trace": args.trace,
        "records": len(rows),
        "request_rows": request_rows,
        "passed_records": pass_count,
        "failed_records": len(failures),
        "skipped_records": skipped_count,
        "compact_records": compact_count,
        "folded_accept_records": folded_count,
        "non_uniform_verify_lens_records": non_uniform_count,
        "non_greedy_records": non_greedy_count,
        "seeded_sampling_records": seeded_count,
        "sampling_backends": sampling_backends,
        "simulated_accept_records": simulated_count,
        "padded_graph_records": padded_graph_count,
        "max_bs": max_bs,
        "min_verify_lens_sum": min(verify_lens_sums) if verify_lens_sums else None,
        "max_verify_lens_sum": max(verify_lens_sums) if verify_lens_sums else None,
        "min_layout_graph_num_tokens": min(graph_token_counts)
        if graph_token_counts
        else None,
        "max_layout_graph_num_tokens": max(graph_token_counts)
        if graph_token_counts
        else None,
        "max_graph_padding_tokens": max(graph_padding_tokens)
        if graph_padding_tokens
        else None,
        "failure_examples": failures[: args.max_failure_examples],
        "skipped_examples": skipped_examples[: args.max_failure_examples],
        "verdict": {
            "passed": not failures and not gate_failures,
            "failures": [f["failures"] for f in failures[: args.max_failure_examples]],
            "gate_failures": gate_failures,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    if args.fail_on_verdict and (failures or gate_failures):
        raise SystemExit(1)


def add_prepare(subparsers) -> None:
    parser = subparsers.add_parser("prepare")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.set_defaults(func=command_prepare)


def add_collect(subparsers) -> None:
    parser = subparsers.add_parser("collect")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-label", default="server")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--allow-nondeterministic-sampling", action="store_true")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-logprob", action="store_true")
    parser.add_argument("--return-prompt-token-ids", action="store_true")
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--print-records", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=command_collect)


def add_compare(subparsers) -> None:
    parser = subparsers.add_parser("compare")
    parser.add_argument("--target", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--max-mismatch-examples", type=int, default=10)
    parser.add_argument("--require-request-signature-match", action="store_true")
    parser.add_argument("--require-prompt-token-match", action="store_true")
    parser.add_argument("--min-token-exact-rate", type=float)
    parser.add_argument("--min-matched-prefix-mean", type=float)
    parser.add_argument("--fail-on-verdict", action="store_true")
    parser.set_defaults(func=command_compare)


def add_run_summary(subparsers) -> None:
    parser = subparsers.add_parser("run-summary")
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--min-ok-requests", type=int)
    parser.add_argument("--min-aggregate-accept-rate", type=float)
    parser.add_argument("--min-aggregate-accept-length", type=float)
    parser.add_argument("--fail-on-verdict", action="store_true")
    parser.set_defaults(func=command_run_summary)


def add_trace_summary(subparsers) -> None:
    parser = subparsers.add_parser("trace-summary")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--max-failure-examples", type=int, default=10)
    parser.add_argument("--require-records-min", type=int)
    parser.add_argument("--require-compact", action="store_true")
    parser.add_argument("--require-non-uniform-verify-lens", action="store_true")
    parser.add_argument("--require-folded-accept", action="store_true")
    parser.add_argument("--require-non-greedy", action="store_true")
    parser.add_argument("--require-seeded-sampling", action="store_true")
    parser.add_argument("--require-padded-graph", action="store_true")
    parser.add_argument("--require-no-skipped", action="store_true")
    parser.add_argument("--fail-on-verdict", action="store_true")
    parser.set_defaults(func=command_trace_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_prepare(subparsers)
    add_collect(subparsers)
    add_compare(subparsers)
    add_run_summary(subparsers)
    add_trace_summary(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
