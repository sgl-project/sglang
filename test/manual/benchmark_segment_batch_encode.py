#!/usr/bin/env python3
"""Manual benchmark for segment_batch_encode (SGLang PR data)."""

import argparse
import json
import os
import statistics
import time

from transformers import AutoTokenizer

from sglang.srt.utils.hf_transformers.segment_batch_encode import (
    DEFAULT_PASSAGE_DELIMITER,
    make_segment_batch_encode_tokenizer,
    segment_batch_encode_ids,
)


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt-json")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if args.prompt_json:
        msgs = json.load(open(args.prompt_json, encoding="utf-8"))
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = (DEFAULT_PASSAGE_DELIMITER + "demo\n") * 20

    plain_ms = bench(lambda: tok.encode(text, add_special_tokens=False))
    seg_ms = bench(
        lambda: segment_batch_encode_ids(
            tok, text, split_delimiter=DEFAULT_PASSAGE_DELIMITER, add_special_tokens=False,
            interleave_delimiter=False,
        )
    )
    wrapped = make_segment_batch_encode_tokenizer(tok, min_chars=1)
    wrap_ms = bench(lambda: wrapped.encode(text, add_special_tokens=False))

    print(f"tokens={len(tok.encode(text, add_special_tokens=False))} passages={text.count(DEFAULT_PASSAGE_DELIMITER)}")
    print(f"plain_encode_ms={plain_ms:.1f}")
    print(f"segment_batch_ms={seg_ms:.1f} speedup={plain_ms/seg_ms:.2f}x")
    print(f"wrapped_encode_ms={wrap_ms:.1f} speedup={plain_ms/wrap_ms:.2f}x")


if __name__ == "__main__":
    main()
