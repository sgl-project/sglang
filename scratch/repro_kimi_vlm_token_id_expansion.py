#!/usr/bin/env python3
"""Smoke repro for Kimi VLM token-id prompt expansion.

This checks the real failure mode behind routing replay off-by-one:

1. The old SGLang token-id VLM path decoded token IDs to text, expanded
   <|media_pad|> as text, then tokenized again. Kimi's tokenizer is not
   bijective for arbitrary model token sequences, so prompt length can drift.
2. The fixed path expands the media placeholder directly in token-id space.

No model weights, server, Redis, or RL run are required.
"""

from __future__ import annotations

import argparse
from asyncio import run
from types import SimpleNamespace

from transformers import AutoTokenizer


DEFAULT_MODEL = "moonshotai/Kimi-K2.6"

# Real Kimi token sequences found by fuzzing. They contain one media placeholder
# and reproduce both signs seen in routing replay logs when passed through the
# old decode -> text expansion -> tokenize path.
PLUS_ONE_CASE = [
    1699,
    163587,
    87,
    220,
    163605,
    364,
    163588,
]
MINUS_ONE_CASE = [
    55097,
    137198,
    74409,
    30355,
    13315,
    150010,
    106096,
    1737,
    7518,
    113138,
    99491,
    30523,
    106960,
    17521,
    89881,
    163587,
    87,
    220,
    163605,
    364,
    163588,
    20784,
    115475,
    99212,
    106397,
    5464,
    43128,
]


class FakeMediaProcessor:
    def __init__(self, image_token_count: int):
        self.image_token_count = image_token_count

    def media_tokens_calculator(self, media):
        return self.image_token_count

    def preprocess(self, medias, return_tensors="pt"):
        return {}


class FakeProcessor:
    def __init__(self, image_token_count: int):
        self.media_processor = FakeMediaProcessor(image_token_count)


def direct_expand(input_ids: list[int], media_token_id: int, image_token_count: int):
    expanded: list[int] = []
    for token_id in input_ids:
        if token_id == media_token_id:
            expanded.extend([media_token_id] * image_token_count)
        else:
            expanded.append(token_id)
    return expanded


def old_sglang_expand(tokenizer, input_ids, image_token_count: int):
    text = tokenizer.decode(input_ids)
    text = text.replace("<|media_pad|>", "<|media_pad|>" * image_token_count, 1)
    return tokenizer(text, return_tensors="pt")["input_ids"].flatten().tolist()


def patched_sglang_expand(input_ids, media_token_id: int, image_token_count: int):
    from sglang.srt.multimodal.processors.kimi_k25 import KimiK2_5VLImageProcessor
    from sglang.srt.multimodal.processors.kimi_token_ids import (
        process_kimi_token_ids_mm_data,
    )

    processor = KimiK2_5VLImageProcessor.__new__(KimiK2_5VLImageProcessor)
    processor.mm_tokens = SimpleNamespace(
        image_token="<|media_pad|>",
        image_token_id=media_token_id,
    )
    processor._processor = FakeProcessor(image_token_count)
    async def fast_load_mm_data(**_):
        return SimpleNamespace(images=[{"fake": True}])

    processor.fast_load_mm_data = fast_load_mm_data
    processor.process_mm_data = lambda **_: {}
    processor.collect_mm_items_from_processor_output = lambda _: []
    processor.finalize_mm_items = lambda items, _input_ids, _tokens: items

    output = run(process_kimi_token_ids_mm_data(processor, input_ids, ["image"]))
    return output.input_ids


def check_case(name: str, tokenizer, input_ids, media_token_id, image_token_count):
    expected = direct_expand(input_ids, media_token_id, image_token_count)
    old = old_sglang_expand(tokenizer, input_ids, image_token_count)
    patched = patched_sglang_expand(input_ids, media_token_id, image_token_count)

    old_diff = len(old) - len(expected)
    print(
        f"{name}: direct_len={len(expected)} old_len={len(old)} "
        f"old_diff={old_diff} patched_len={len(patched)}"
    )

    if old_diff == 0:
        raise AssertionError(f"{name}: old path did not reproduce a length mismatch")
    if patched != expected:
        raise AssertionError(
            f"{name}: patched path did not match direct token-id expansion"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-token-count", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    media_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
    if media_token_id is None or media_token_id < 0:
        raise RuntimeError("Could not resolve <|media_pad|> token id")

    check_case(
        "old_path_longer_by_one",
        tokenizer,
        PLUS_ONE_CASE,
        media_token_id,
        args.image_token_count,
    )
    check_case(
        "old_path_shorter_by_one",
        tokenizer,
        MINUS_ONE_CASE,
        media_token_id,
        args.image_token_count,
    )
    print(
        "PASS: old path reproduces both off-by-one signs; fixed path matches direct expansion."
    )


if __name__ == "__main__":
    main()
