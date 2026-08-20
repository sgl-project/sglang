#!/usr/bin/env python3
"""Build a deterministic, category-balanced long-context LongBench-v2 slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from sglang.test.simple_eval_longbench_v2 import (
    TASK_CATEGORIES,
    format_longbench_v2_question,
)


DOMAIN_TO_CATEGORY = {
    "Single-Document QA": "single_document_qa",
    "Multi-Document QA": "multi_document_qa",
    "Long In-context Learning": "long_in_context_learning",
    "Long-dialogue History Understanding": "long_dialogue_history",
    "Code Repository Understanding": "code_repo_understanding",
    "Long Structured Data Understanding": "long_structured_data",
}


def canonical_category(example: dict) -> str | None:
    value = example.get("category", example.get("domain"))
    if value in TASK_CATEGORIES:
        return value
    return DOMAIN_TO_CATEGORY.get(value)


def stable_key(example: dict) -> str:
    identity = json.dumps(
        {
            "context": example.get("context", ""),
            "question": example.get("question", ""),
            "answer": example.get("answer", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="THUDM/LongBench-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=32768)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [dict(row) for row in load_dataset(args.dataset, split=args.split)]
    eligible: dict[str, list[dict]] = defaultdict(list)
    token_lengths: dict[str, int] = {}
    for row in rows:
        category = canonical_category(row)
        if category is None:
            continue
        key = stable_key(row)
        num_tokens = len(
            tokenizer.encode(
                format_longbench_v2_question(row), add_special_tokens=False
            )
        )
        if num_tokens >= args.min_tokens:
            eligible[category].append(row)
            token_lengths[key] = num_tokens

    categories = sorted(TASK_CATEGORIES)
    base, remainder = divmod(args.num_examples, len(categories))
    selected: list[dict] = []
    selected_keys: set[str] = set()
    for index, category in enumerate(categories):
        quota = base + (index < remainder)
        ranked = sorted(eligible[category], key=stable_key)
        for row in ranked[:quota]:
            selected.append(row)
            selected_keys.add(stable_key(row))

    if len(selected) < args.num_examples:
        remainder_rows = sorted(
            (
                row
                for category in categories
                for row in eligible[category]
                if stable_key(row) not in selected_keys
            ),
            key=stable_key,
        )
        selected.extend(remainder_rows[: args.num_examples - len(selected)])
    if len(selected) != args.num_examples:
        raise RuntimeError(
            f"only {len(selected)} examples have at least {args.min_tokens} tokens"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    lengths = [token_lengths[stable_key(row)] for row in selected]
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "minimum_tokens": args.min_tokens,
        "num_examples": len(selected),
        "category_counts": Counter(canonical_category(row) for row in selected),
        "domain_to_category": DOMAIN_TO_CATEGORY,
        "minimum_observed_tokens": min(lengths),
        "maximum_observed_tokens": max(lengths),
        "subset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
