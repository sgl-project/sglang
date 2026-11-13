#!/usr/bin/env python3
"""Offline, small-sample GSM8K check using vLLM's five-shot prompt format."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def answer_value(text: str) -> int | None:
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[-1]) if numbers else None


def build_cases(num_questions: int) -> list[tuple[str, int]]:
    train = json.loads((DATA_DIR / "gsm8k_train_5.json").read_text())
    test = json.loads((DATA_DIR / "gsm8k_test_40.json").read_text())
    if len(train) != 5 or len(test) != 40:
        raise ValueError("Expected five training examples and 40 test questions.")
    if not 1 <= num_questions <= len(test):
        raise ValueError("--num-questions must be between 1 and 40.")
    examples = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train
    )
    cases = []
    for row in test[:num_questions]:
        question = row["conversations"][0]["value"]
        reference = row["conversations"][1]["value"]
        expected = answer_value(reference)
        if expected is None:
            raise ValueError(f"Missing numeric answer for {question!r}")
        cases.append((examples + f"Question: {question}\nAnswer:", expected))
    return cases


def generate(
    url: str, prompt: str, max_new_tokens: int, timeout: int
) -> tuple[str, dict]:
    payload = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": False,
            },
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    text = result.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Unexpected SGLang response: {result!r}")
    return text, result.get("meta_info") or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("xpress", "dflash", "dspark", "target"), required=True
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "30000"))
    )
    parser.add_argument("--num-questions", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    cases = build_cases(args.num_questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/generate"
    correct = 0
    verify_count = 0
    correct_drafts = 0
    proposed_drafts = 0
    verified_output_tokens = 0
    with args.output.open("x", encoding="utf-8") as output:
        for index, (prompt, expected) in enumerate(cases, start=1):
            text, meta_info = generate(url, prompt, args.max_new_tokens, args.timeout)
            predicted = answer_value(text)
            is_correct = predicted == expected
            correct += is_correct
            verify_count += int(meta_info.get("spec_verify_ct") or 0)
            correct_drafts += int(meta_info.get("spec_num_correct_drafts") or 0)
            proposed_drafts += int(meta_info.get("spec_num_proposed_drafts") or 0)
            if meta_info.get("spec_verify_ct"):
                verified_output_tokens += int(meta_info.get("completion_tokens") or 0)
            output.write(
                json.dumps(
                    {
                        "mode": args.mode,
                        "index": index,
                        "expected": expected,
                        "predicted": predicted,
                        "correct": is_correct,
                        "output": text,
                        "spec_verify_ct": meta_info.get("spec_verify_ct"),
                        "spec_accept_length": meta_info.get("spec_accept_length"),
                        "spec_accept_rate": meta_info.get("spec_accept_rate"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()
            print(
                f"{args.mode} {index}/{len(cases)} accuracy={correct / index:.3f}",
                flush=True,
            )
    print(
        f"{args.mode} GSM8K 5-shot accuracy: {correct}/{len(cases)} = {correct / len(cases):.3f}"
    )
    if verify_count:
        print(
            f"{args.mode} acceptance length: {verified_output_tokens / verify_count:.3f}"
        )
    if proposed_drafts:
        print(f"{args.mode} acceptance rate: {correct_drafts / proposed_drafts:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
