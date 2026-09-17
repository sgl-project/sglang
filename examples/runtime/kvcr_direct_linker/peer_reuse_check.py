"""Explicit-hint peer reuse check for the KVCR direct linker.

Runs a prompt on a source worker (so its pages are offloaded into the source
KVCR tier), flushes the target worker, then replays the prompt on the target
with a ``kv_hints`` envelope naming the source. Compares the target's greedy
output with the control run and reports the cached-token count the target
served. Requires the ``sglang`` package for the page hashing helpers.

Example:
    python peer_reuse_check.py --source http://host-a:30000 \
        --source-control tcp://host-a:25000 --target http://host-b:30001 \
        --model Qwen/Qwen3-0.6B --page-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid

from sglang.srt.mem_cache.utils import get_storage_hash_str, hash_str_to_int64


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _generate(
    base: str, token_ids: list[int], max_new_tokens: int, kv_hints=None
) -> dict:
    payload = {
        "input_ids": token_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    if kv_hints is not None:
        payload["kv_hints"] = kv_hints
    return _post(f"{base}/generate", payload)


def _tokenize(model: str, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer.encode(text, add_special_tokens=False)


def _hint(source_control: str, token_ids: list[int], page_size: int) -> dict:
    full_pages = len(token_ids) // page_size * page_size
    hashes = get_storage_hash_str(token_ids[:full_pages], None, page_size=page_size)
    return {
        "protocol_version": "0.1",
        "message_id": uuid.uuid4().hex,
        "actions": [
            {
                "action_id": uuid.uuid4().hex,
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": source_control,
                    "block_hashes": [hash_str_to_int64(h) for h in hashes],
                },
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-control", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()

    base_text = " ".join(f"token{i}" for i in range(args.prompt_tokens))
    token_ids = _tokenize(args.model, base_text)[: args.prompt_tokens]
    if len(token_ids) < 2 * args.page_size:
        print("prompt too short for a page-aligned prefix", file=sys.stderr)
        return 2

    control = _generate(args.source, token_ids, args.max_new_tokens)
    # Offload happens asynchronously after the request finishes.
    time.sleep(args.settle_seconds)
    _post(f"{args.target}/flush_cache", {}, timeout=60.0)

    cold = _generate(args.target, token_ids, args.max_new_tokens)
    _post(f"{args.target}/flush_cache", {}, timeout=60.0)
    time.sleep(args.settle_seconds)

    hinted = _generate(
        args.target,
        token_ids,
        args.max_new_tokens,
        kv_hints=_hint(args.source_control, token_ids, args.page_size),
    )

    report = {
        "control_text": control["text"],
        "cold_target_text": cold["text"],
        "hinted_target_text": hinted["text"],
        "cold_cached_tokens": cold["meta_info"].get("cached_tokens"),
        "hinted_cached_tokens": hinted["meta_info"].get("cached_tokens"),
        "outputs_match": control["text"] == hinted["text"] == cold["text"],
    }
    print(json.dumps(report, indent=2))
    ok = report["outputs_match"] and (report["hinted_cached_tokens"] or 0) > (
        report["cold_cached_tokens"] or 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
