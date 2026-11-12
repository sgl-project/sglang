"""Rebuild the per-run aisbench prefix toolkit's dataset generator.

Two rewrites, both applied inside the disposable /tmp copy of the toolkit:

1. ``create_dataset`` builds the shared prefix by concatenating distinct
   GSM8K rows instead of repeating one question until input_len. At 128k the
   upstream padding turns the prompt into one question repeated ~1500 times;
   DSpark then keeps emitting EOS and message separators, the verify accept
   length collapses (~2.8) and TPOT sits around 37 ms. A natural-text pool
   keeps the tail readable (accept ~3.6-4.3, TPOT ~28 ms).
2. ``write_data`` pads (or truncates) the rows to the requested count. With
   repeat_rate >= 1 the dataset is just the dp x prefix_num prefix copies, so
   without padding every run is pinned to C=dp no matter what num_prompts
   says.

The cached toolkit on the shared PVC is a different revision from the public
repo, so body text cannot be matched reliably. Each function is located with
``ast`` and replaced by line span, reusing the original signature plus the
original ``AutoTokenizer.from_pretrained(...)`` / ``DataPicker(...)``
expressions. run_aisbench.sh copies the toolkit to /tmp for every attempt and
runs it from there, so only that copy is modified; the cached toolkit on the
shared PVC is never written back.
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
WRITE_FUNC_NAME = "write_data"
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

# repeat_rate >= 1 writes the dp x prefix_num prefix copies as the dataset, so
# the row count equals dp x prefix_num unless the writer pads it to data_num.
# The baked copy does not pad, which pins every run to C=dp no matter what
# num_prompts says. Pad (or truncate) to num here so num_prompts is honoured.
NEW_WRITE_BODY = """\
    if __NUM__ is not None and len(__DATASET__) < __NUM__:
        repeats = __NUM__ // len(__DATASET__)
        remainder = __NUM__ % len(__DATASET__)
        __DATASET__ = __DATASET__ * repeats + __DATASET__[:remainder]
    elif __NUM__ is not None:
        __DATASET__ = __DATASET__[:__NUM__]
    with open(__PATH__, "w", encoding="utf-8") as f:
        for item in __DATASET__:
            f.write(json.dumps({"question": item, "answer": "none"}, ensure_ascii=False))
            f.write("\\n")
"""


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _function_parts(text, path, name):
    tree = ast.parse(text)
    target = _find_function(tree, name)
    if target is None:
        names = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        raise SystemExit(f"{name} not found in {path}; functions={names}")
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: target.lineno - 1])
    end = sum(len(line) for line in lines[: target.end_lineno])
    sig = "".join(lines[target.lineno - 1 : target.body[0].lineno - 1]).rstrip()
    if not sig.endswith(":"):
        raise SystemExit(f"cannot locate the {name} signature: {sig[:120]!r}")
    return {
        "start": start,
        "end": end,
        "orig": text[start:end],
        "sig": sig,
        "params": [a.arg for a in target.args.args],
    }


def _rewrite_toolkit(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    src_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    parts = _function_parts(text, path, FUNC_NAME)
    missing = [p for p in REQUIRED_PARAMS if p not in parts["params"]]
    if missing:
        raise SystemExit(
            f"{FUNC_NAME}{tuple(parts['params'])} lacks {missing} in {path}"
        )
    picker_match = re.search(r"DataPicker\((?:[^()]|\([^()]*\))*\)", parts["orig"])
    picker_expr = (
        picker_match.group(0)
        if picker_match
        else 'DataPicker("./GSM8K.jsonl", "./picked_ids.txt", prefix_flag)'
    )
    tokenizer_match = re.search(
        r"AutoTokenizer\.from_pretrained\([^()]*\)", parts["orig"]
    )
    tokenizer_expr = (
        tokenizer_match.group(0)
        if tokenizer_match
        else "AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)"
    )
    body = NEW_BODY.replace("__TOKENIZER_INIT__", tokenizer_expr).replace(
        "__PICKER_INIT__", picker_expr
    )
    create_lines = (len(parts["orig"].splitlines()), len(body.splitlines()) + 1)
    sig = parts["sig"]
    text = text[: parts["start"]] + sig + "\n" + body + text[parts["end"] :]

    parts = _function_parts(text, path, WRITE_FUNC_NAME)
    if len(parts["params"]) != 3:
        raise SystemExit(
            f"{WRITE_FUNC_NAME}{tuple(parts['params'])} is not the 3-arg row writer"
        )
    write_body = (
        NEW_WRITE_BODY.replace("__PATH__", parts["params"][0])
        .replace("__DATASET__", parts["params"][1])
        .replace("__NUM__", parts["params"][2])
    )
    write_marker = f"{parts['params'][1]} * repeats"
    write_lines = (len(parts["orig"].splitlines()), len(write_body.splitlines()) + 1)
    text = (
        text[: parts["start"]] + parts["sig"] + "\n" + write_body + text[parts["end"] :]
    )

    ast.parse(text)
    if text.count(f"def {FUNC_NAME}") != 1 or "pool_tokens" not in text:
        raise SystemExit(f"rewrite of {path} did not produce {FUNC_NAME} as expected")
    if text.count(f"def {WRITE_FUNC_NAME}") != 1 or write_marker not in text:
        raise SystemExit(
            f"rewrite of {path} did not produce {WRITE_FUNC_NAME} as expected"
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    print(
        f"rewrote {FUNC_NAME} in {path}: {create_lines[0]} -> {create_lines[1]} lines, "
        f"sha256(src)={src_sha}"
    )
    print(f"  signature: {sig.splitlines()[0].strip()}")
    print(f"  tokenizer init: {tokenizer_expr}")
    print(f"  picker init: {picker_expr}")
    print(
        f"rewrote {WRITE_FUNC_NAME} in {path}: {write_lines[0]} -> "
        f"{write_lines[1]} lines (rows now follow data_num)"
    )


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

    _rewrite_toolkit(target)
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
