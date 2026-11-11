"""Rewrite the per-run aisbench prefix toolkit so the shared prefix is built
from a natural-text pool instead of one repeated GSM8K question.

The upstream toolkit (rayn-zzz/aisbench_auto_tools_prefix) pads a prefix by
picking a single GSM8K row and repeating it until input_len. At 128k tokens
that is one question repeated ~1500 times; DSpark then keeps emitting EOS and
message separators, the verify accept length collapses (~2.8) and TPOT sits
around 37 ms. Concatenating distinct rows keeps the prompt tail natural text.

run_aisbench.sh copies the toolkit to /tmp for every attempt and runs it from
there, so only that copy is modified; the cached toolkit on the shared PVC is
never written back.
"""

import argparse
import json
import os
import sys

CORPUS_NAME = "GSM8K.jsonl"
# A 128k prefix needs roughly input_len / 60 = ~2200 distinct GSM8K rows.
MIN_CORPUS_ROWS = 4000

REPEAT_PAD = """        # 根据需求调整长度：重复或截断
        if len(tokens) >= input_len:
            # 截断到 input_len
            adjusted_tokens = tokens[:input_len]
        else:
            # 重复整个序列直到达到 input_len
            repeat_times = (input_len + len(tokens) - 1) // len(tokens)
            repeated_tokens = tokens * repeat_times
            adjusted_tokens = repeated_tokens[:input_len]
"""

POOL_PAD = """        # 根据需求调整长度：重复或截断
        if len(tokens) >= input_len:
            # 截断到 input_len
            adjusted_tokens = tokens[:input_len]
        elif prefix_flag == 1:
            # 前缀改为多题自然文本拼接：连续挑选不同的 GSM8K 样本，避免单题重复
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
            # 重复整个序列直到达到 input_len
            repeat_times = (input_len + len(tokens) - 1) // len(tokens)
            repeated_tokens = tokens * repeat_times
            adjusted_tokens = repeated_tokens[:input_len]
"""

REPEAT_FIXUP = """            if len(corrected_tokens) >= input_len:
                corrected_tokens = corrected_tokens[:input_len]
            else:
                corrected_tokens = (corrected_tokens * ((input_len // len(corrected_tokens)) + 1))[:input_len]
"""

# 解码再编码会掉几个 token，原逻辑是"整篇再复制一遍"，对自然文本池会退回
# 退化重复文本，这里改成只补尾部。
POOL_FIXUP = """            if len(corrected_tokens) >= input_len:
                corrected_tokens = corrected_tokens[:input_len]
            elif prefix_flag == 1:
                # 自然文本池不做整体重复（会退回退化重复文本），只补尾部
                gap = input_len - len(corrected_tokens)
                corrected_tokens = corrected_tokens + corrected_tokens[-gap:]
            else:
                corrected_tokens = (corrected_tokens * ((input_len // len(corrected_tokens)) + 1))[:input_len]
"""


def _replace_once(path, old, new):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"patch anchor not found exactly once in {path} (hits={count}); "
            "the cached toolkit layout changed"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))


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

    _replace_once(target, REPEAT_PAD, POOL_PAD)
    _replace_once(target, REPEAT_FIXUP, POOL_FIXUP)
    rows = _top_up_corpus(corpus, args.train_jsonl, args.min_corpus_rows)
    print(f"natural-text prefix pool enabled: patched {target}, corpus rows={rows}")
    if rows < args.min_corpus_rows:
        print(
            f"WARNING: corpus has {rows} rows (< {args.min_corpus_rows}); a 128k "
            "prefix may run out of distinct rows and fall back to tail padding",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
