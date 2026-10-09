# SPDX-License-Identifier: Apache-2.0
"""Auditable served GSM8K evaluation; no local model execution or judge model."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

from run_inferencex import load_encoder

HASHES = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
COUNTS = {"train": 7473, "test": 1319}
PROTOCOL = "dsv41-gsm8k-train5-cot-completions-v1"
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def number(text):
    try:
        value = Decimal(text.replace(",", ""))
        return (
            ("0" if value == 0 else format(value.normalize(), "f"))
            if value.is_finite()
            else None
        )
    except InvalidOperation:
        return None


def strict_answer(text):
    matches = re.findall(r"^\s*####\s*\$?(" + NUMBER + r")\s*$", text, re.MULTILINE)
    return number(matches[-1]) if matches else None


def load_dataset(directory):
    data = {}
    for split in ("train", "test"):
        path = Path(directory) / f"{split}.jsonl"
        if digest(path) != HASHES[split]:
            raise ValueError(f"{split} dataset SHA256 mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != COUNTS[split] or any(
            not isinstance(r.get("question"), str)
            or not isinstance(r.get("answer"), str)
            or strict_answer(r["answer"]) is None
            for r in rows
        ):
            raise ValueError(f"Invalid {split} dataset rows")
        data[split] = rows
    if {r["question"] for r in data["train"][:5]} & {
        r["question"] for r in data["test"]
    }:
        raise ValueError("Few-shot/test question overlap")
    return data


def selected_indices(mode, seed):
    return (
        list(range(1319))
        if mode == "full"
        else sorted(random.Random(seed).sample(range(1319), 32))
    )


def make_prompt(question, train, encode, thinking, effort):
    examples = "\n\n".join(
        "Question: "
        + row["question"]
        + "\nAnswer: "
        + re.sub(r"<<.*?>>", "", row["answer"])
        for row in train[:5]
    )
    content = (
        "Solve the math problem. Show your calculation and finish with a line "
        "exactly in the form #### <number>.\n\nHere are five examples:\n\n"
        + examples
        + "\n\nQuestion: "
        + question
        + "\nAnswer:"
    )
    return encode(
        [{"role": "user", "content": content}],
        thinking_mode=thinking,
        reasoning_effort=effort,
    )


def grade(text, expected, thinking, finish_reason, error=None):
    # Never extract an intermediate number from an unfinished thinking block.
    if thinking == "thinking":
        final = text.rsplit("</think>", 1)[1] if "</think>" in text else ""
    else:
        final = text.rsplit("</think>", 1)[-1]
    final = final.replace("<｜end▁of▁sentence｜>", "").strip()
    strict = strict_answer(final)
    numbers = re.findall(NUMBER, final)
    flexible = (
        strict if strict is not None else (number(numbers[-1]) if numbers else None)
    )
    usable = not error and finish_reason == "stop"
    return dict(
        final_content=final,
        prediction=strict,
        flexible_prediction=flexible,
        correct=bool(usable and strict is not None and strict == expected),
        flexible_correct=bool(usable and flexible is not None and flexible == expected),
        truncated=finish_reason == "length",
        missing_final=not bool(final),
    )


def request_one(entry, args):
    started = time.monotonic()
    response, error = {}, None
    try:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(entry["request"]).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=args.request_timeout) as handle:
            response = json.loads(handle.read())
        choice = response["choices"][0]
        if not isinstance(choice["text"], str) or choice.get("finish_reason") not in (
            "stop",
            "length",
        ):
            raise ValueError(
                "Malformed completion response or unexpected finish_reason"
            )
        text, finish = choice["text"], choice["finish_reason"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        text, finish = "", None
    return dict(
        index=entry["index"],
        response=response,
        response_text=text,
        finish_reason=finish,
        error=error,
        latency_s=time.monotonic() - started,
    )


def score_records(records, indices, test, thinking):
    if len(records) != len(indices) or sorted(r["index"] for r in records) != sorted(
        indices
    ):
        raise ValueError("Missing, duplicate or unexpected prediction indices")
    scored = []
    for record in sorted(records, key=lambda r: r["index"]):
        row = test[record["index"]]
        expected = strict_answer(row["answer"])
        scored.append(
            dict(
                record,
                ground_truth=expected,
                ground_truth_text=row["answer"],
                **grade(
                    record["response_text"],
                    expected,
                    thinking,
                    record["finish_reason"],
                    record["error"],
                ),
            )
        )
    n = len(scored)
    return scored, dict(
        num_examples=n,
        correct=sum(r["correct"] for r in scored),
        accuracy=sum(r["correct"] for r in scored) / n,
        flexible_accuracy=sum(r["flexible_correct"] for r in scored) / n,
        errors=sum(bool(r["error"]) for r in scored),
        truncated=sum(r["truncated"] for r in scored),
        missing_final=sum(r["missing_final"] for r in scored),
        missing_answer_marker=sum(r["prediction"] is None for r in scored),
    )


def rescore(folder, dataset):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    data = load_dataset(dataset)
    if (folder / "summary.json").exists():
        summary = json.loads((folder / "summary.json").read_text())
        if digest(folder / "responses.jsonl") != summary["responses_sha256"]:
            raise ValueError("Raw response SHA256 differs from saved summary")
    records = [
        json.loads(line)
        for line in (folder / "responses.jsonl").read_text().splitlines()
    ]
    scored, stats = score_records(
        records, manifest["indices"], data["test"], manifest["thinking"]
    )
    return scored, stats


def smoke_label(mode, passed):
    # A full-set result is not another smoke gate, even when it has capped outputs.
    return ("passed" if passed else "failed") if mode == "smoke" else "not_applicable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--mode", choices=["smoke", "full", "diagnostic"], default="smoke"
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Diagnostic-only test indices; never a qualification score",
    )
    parser.add_argument("--thinking", choices=["thinking", "chat"], default="thinking")
    parser.add_argument("--reasoning-effort", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rescore", type=Path)
    args = parser.parse_args(argv)
    if (args.mode == "diagnostic") != bool(args.indices):
        parser.error(
            "--indices is required for diagnostic mode and forbidden for scored modes"
        )
    if args.indices and (
        len(set(args.indices)) != len(args.indices)
        or any(i < 0 or i >= 1319 for i in args.indices)
    ):
        parser.error("Diagnostic indices must be unique test indices in [0, 1319)")
    data = load_dataset(args.dataset)
    if args.check_only:
        print(
            json.dumps(
                dict(dataset=str(args.dataset), rows=COUNTS, sha256=HASHES), indent=2
            )
        )
        return 0
    if args.rescore:
        _, stats = rescore(args.rescore, args.dataset)
        old = json.loads((args.rescore / "summary.json").read_text())
        if any(old[k] != v for k, v in stats.items()):
            raise ValueError("Rescoring does not match saved summary")
        print(json.dumps(dict(rescore="passed", **stats), indent=2))
        return 0
    if (
        not args.output_dir
        or not 1 <= args.reasoning_effort <= 100
        or min(
            args.concurrency, args.max_tokens, args.context_length, args.request_timeout
        )
        <= 0
    ):
        parser.error(
            "Require output-dir, positive resource limits and reasoning-effort 1..100"
        )
    if args.output_dir.exists():
        parser.error(
            "output-dir already exists: choose a fresh path (no overwrite/resume)"
        )
    # Tokenization only; all neural inference runs in the serving process.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    encode = load_encoder(args.model_path)
    indices = (
        sorted(args.indices)
        if args.mode == "diagnostic"
        else selected_indices(args.mode, args.seed)
    )
    entries = []
    for index in indices:
        prompt = make_prompt(
            data["test"][index]["question"],
            data["train"],
            encode,
            args.thinking,
            args.reasoning_effort,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + args.max_tokens + 64 > args.context_length:
            raise ValueError(
                f"Example {index}: prompt + output budget exceeds context; no silent truncation"
            )
        entries.append(
            dict(
                index=index,
                prompt=prompt,
                prompt_tokens=len(ids),
                request=dict(
                    model=args.model,
                    prompt=ids,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    stream=False,
                    skip_special_tokens=False,
                ),
            )
        )
    args.output_dir.mkdir(parents=True)
    write_json(
        args.output_dir / "manifest.json",
        dict(
            protocol=PROTOCOL,
            mode=args.mode,
            indices=indices,
            fewshot_split="train",
            fewshot_indices=list(range(5)),
            dataset=str(args.dataset),
            dataset_sha256=HASHES,
            dataset_rows=COUNTS,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            context_length=args.context_length,
            concurrency=args.concurrency,
            seed=args.seed,
            temperature=0.0,
            top_p=1.0,
            endpoint=args.base_url + "/v1/completions",
            model=args.model,
            model_path=str(args.model_path),
            encoder_sha256=digest(args.model_path / "encoding/encoding.py"),
            evaluator_sha256=digest(__file__),
            prompt_tokens_min=min(e["prompt_tokens"] for e in entries),
            prompt_tokens_max=max(e["prompt_tokens"] for e in entries),
            retry_policy="no retries",
            accuracy_definition="strict final #### numeric exact match; length/error/missing final counted incorrect",
        ),
    )
    with (args.output_dir / "requests.jsonl").open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    started, records = time.monotonic(), []
    with (
        (args.output_dir / "responses.jsonl").open("w") as handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        pending = [pool.submit(request_one, entry, args) for entry in entries]
        for future in concurrent.futures.as_completed(pending):
            record = future.result()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            print(
                f"Completed {len(records)}/{len(indices)} index={record['index']} finish={record['finish_reason']} error={record['error']}",
                flush=True,
            )
    scored, stats = score_records(records, indices, data["test"], args.thinking)
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    complete = stats["errors"] == 0
    smoke_passed = (
        complete
        and stats["truncated"] == 0
        and stats["accuracy"] > 0.75
        and stats["missing_final"] == 0
    )
    summary = dict(
        stats,
        protocol=PROTOCOL,
        mode=args.mode,
        elapsed_s=time.monotonic() - started,
        status="completed" if complete else "failed_transport",
        smoke_gate=smoke_label(args.mode, smoke_passed),
        quality_judgement=(
            "NOT JUDGED: diagnostic subset, not qualification"
            if args.mode == "diagnostic"
            else "NOT JUDGED: smoke only"
            if args.mode == "smoke"
            else "MEASURED: no matched reference threshold established"
        ),
        responses_sha256=digest(args.output_dir / "responses.jsonl"),
    )
    write_json(args.output_dir / "summary.json", summary)
    _, rescored = rescore(args.output_dir, args.dataset)
    if stats != rescored:
        raise ValueError("Independent artifact rescore mismatch")
    write_json(args.output_dir / "rescore.json", dict(status="passed", **rescored))
    print(json.dumps(summary, indent=2), flush=True)
    return (
        0 if complete and (args.mode in ("full", "diagnostic") or smoke_passed) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
