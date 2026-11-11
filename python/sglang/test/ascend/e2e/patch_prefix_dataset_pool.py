"""Rebuild the per-run aisbench prefix toolkit's dataset generator so the
shared prefix is a natural-text pool instead of one repeated GSM8K question.

The upstream toolkit pads a prefix by picking a single GSM8K row and
repeating it until input_len. At 128k tokens that is one question repeated
~1500 times; DSpark then keeps emitting EOS and message separators, the
verify accept length collapses (~2.8) and TPOT sits around 37 ms.
Concatenating distinct rows keeps the prompt tail natural text.

The cached toolkit on the shared PVC is a different revision from the public
repo, so the body text cannot be matched reliably. The rewrite therefore
locates ``create_dataset`` with ``ast`` and replaces the whole function by
line span, reusing the original signature plus the original
``AutoTokenizer.from_pretrained(...)`` / ``DataPicker(...)`` expressions.

run_aisbench.sh copies the toolkit to /tmp for every attempt and runs it from
there, so only that copy is modified; the cached toolkit on the shared PVC is
never written back.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys

CORPUS_NAME = "GSM8K.jsonl"
FUNC_NAME = "create_dataset"
REQUIRED_PARAMS = ("tokenizer_path", "input_len", "number", "prefix_flag")
# A 128k prefix needs roughly input_len / 60 = ~2200 distinct GSM8K rows.
MIN_CORPUS_ROWS = 4000

NEW_BODY = """\
    logging.info(f"加载tokenizer: {tokenizer_path}")
    tokenizer = __TOKENIZER_INIT__
    output_samples = []
    attempts = 0
    max_attempts = number * 10
    while len(output_samples) < number and attempts < max_attempts:
        attempts += 1
        picker = __PICKER_INIT__
        raw_text = picker.pick_one()
        if raw_text is None:
            break
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        if len(tokens) == 0:
            break
        if len(tokens) >= input_len:
            adjusted_tokens = tokens[:input_len]
        elif prefix_flag == 1:
            pool_tokens = list(tokens)
            while len(pool_tokens) < input_len:
                more_text = picker.pick_one()
                if more_text is None:
                    break
                more_tokens = tokenizer.encode(more_text, add_special_tokens=False)
                if not more_tokens:
                    break
                pool_tokens.extend(more_tokens)
            adjusted_tokens = pool_tokens[:input_len]
        else:
            repeat_times = (input_len + len(tokens) - 1) // len(tokens)
            adjusted_tokens = (tokens * repeat_times)[:input_len]
        adjusted_text = tokenizer.decode(adjusted_tokens, skip_special_tokens=True)
        final_len = len(tokenizer.encode(adjusted_text, add_special_tokens=False))
        if final_len != input_len:
            corrected_tokens = tokenizer.encode(adjusted_text, add_special_tokens=False)
            if len(corrected_tokens) >= input_len:
                corrected_tokens = corrected_tokens[:input_len]
            elif prefix_flag == 1:
                gap = input_len - len(corrected_tokens)
                corrected_tokens = corrected_tokens + corrected_tokens[-gap:]
            else:
                corrected_tokens = (
                    corrected_tokens * ((input_len // len(corrected_tokens)) + 1)
                )[:input_len]
            adjusted_text = tokenizer.decode(corrected_tokens, skip_special_tokens=True)
        output_samples.append(adjusted_text)
    if len(output_samples) < number:
        return None
    return output_samples
"""


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _rewrite_create_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    target = _find_function(tree, FUNC_NAME)
    if target is None:
        names = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        raise SystemExit(f"{FUNC_NAME} not found in {path}; functions={names}")

    params = [a.arg for a in target.args.args]
    missing = [p for p in REQUIRED_PARAMS if p not in params]
    if missing:
        raise SystemExit(f"{FUNC_NAME}{tuple(params)} lacks {missing} in {path}")

    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: target.lineno - 1])
    end = sum(len(line) for line in lines[: target.end_lineno])
    orig = text[start:end]
    sig = "".join(lines[target.lineno - 1 : target.body[0].lineno - 1]).rstrip()
    if not sig.endswith(":"):
        raise SystemExit(f"cannot locate the {FUNC_NAME} signature: {sig[:120]!r}")

    picker_match = re.search(r"DataPicker\((?:[^()]|\([^()]*\))*\)", orig)
    picker_expr = (
        picker_match.group(0)
        if picker_match
        else 'DataPicker("./GSM8K.jsonl", "./picked_ids.txt", prefix_flag)'
    )
    tokenizer_match = re.search(r"AutoTokenizer\.from_pretrained\([^()]*\)", orig)
    tokenizer_expr = (
        tokenizer_match.group(0)
        if tokenizer_match
        else "AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)"
    )

    body = NEW_BODY.replace("__TOKENIZER_INIT__", tokenizer_expr).replace(
        "__PICKER_INIT__", picker_expr
    )
    new_text = text[:start] + sig + "\n" + body + text[end:]
    ast.parse(new_text)
    if new_text.count(f"def {FUNC_NAME}") != 1 or "pool_tokens" not in new_text:
        raise SystemExit(f"rewrite of {path} did not produce {FUNC_NAME} as expected")

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)

    print(
        f"rewrote {FUNC_NAME} in {path}: {len(orig.splitlines())} -> "
        f"{len((sig + chr(10) + body).splitlines())} lines, "
        f"sha256(src)={hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
    )
    print(f"  signature: {sig.splitlines()[0].strip()}")
    print(f"  tokenizer init: {tokenizer_expr}")
    print(f"  picker init: {picker_expr}")


def _count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _top_up_corpus(corpus_path, train_path, min_rows):
    rows = _count_rows(corpus_path)
    if rows >= min_rows or not train_path or not os.path.isfile(train_path):
        return rows
    with open(train_path, "r", encoding="utf-8") as src:
        first = src.readline()
        try:
            sample = json.loads(first)
        except ValueError:
            raise SystemExit(f"{train_path} is not a GSM8K jsonl file")
        if not sample.get("question"):
            raise SystemExit(f"{train_path} has no 'question' field")
        src.seek(0)
        with open(corpus_path, "a", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(line if line.endswith("\n") else line + "\n")
    return _count_rows(corpus_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit-dir", required=True)
    parser.add_argument("--train-jsonl", default="")
    parser.add_argument("--min-corpus-rows", type=int, default=MIN_CORPUS_ROWS)
    args = parser.parse_args()

    target = os.path.join(args.toolkit_dir, "generate_dataset.py")
    corpus = os.path.join(args.toolkit_dir, CORPUS_NAME)
    if not os.path.isfile(target):
        raise SystemExit(f"{target} not found; is --toolkit-dir the copied toolkit?")
    if not os.path.isfile(corpus):
        raise SystemExit(f"{corpus} not found; the toolkit corpus layout changed")

    _rewrite_create_dataset(target)
    rows = _top_up_corpus(corpus, args.train_jsonl, args.min_corpus_rows)
    print(f"natural-text prefix pool enabled: corpus rows={rows}")
    if rows < args.min_corpus_rows:
        print(
            f"WARNING: corpus has {rows} rows (< {args.min_corpus_rows}); a 128k "
            "prefix may run out of distinct rows and fall back to tail padding",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
