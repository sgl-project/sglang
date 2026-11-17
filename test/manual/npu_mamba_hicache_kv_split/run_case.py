#!/usr/bin/env python3
"""构造 mamba HiCache 驱逐-回读场景并判定输出一致性。

流程：
1. 发送 target prompt（greedy），记录首次输出。
2. 发送 NUM_FILLER 个互不共享前缀的 filler 请求制造设备驱逐压力，
   并要求 Mamba D2H 计数增长。
3. 重放同一 target prompt，要求 Mamba H2D 计数增长。
4. 判定：两次 greedy 输出一致，且两个计数器均增长。
"""

import argparse
import json
import re
import sys
import time

import requests


def generate(base_url, text, max_new_tokens, timeout):
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["text"], body.get("meta_info", {})


def make_prompt(tag, chars):
    unit = (
        f"[{tag}] The linear-attention state carries a fixed-size summary "
        f"per request regardless of sequence length. Paragraph {tag} repeats "
        f"this sentence to build a long non-shared prefix. "
    )
    return unit * (chars // len(unit) + 1)


def mamba_metric(base_url, metric, timeout):
    response = requests.get(f"{base_url}/metrics", timeout=timeout)
    response.raise_for_status()
    values = []
    for line in response.text.splitlines():
        if not line.startswith(metric + "{"):
            continue
        labels, value = line.split("}", 1)
        if re.search(r'pool="mamba"', labels):
            values.append(float(value.strip()))
    return sum(values)


def await_metric_increase(base_url, metric, baseline, timeout):
    deadline = time.monotonic() + timeout
    while True:
        value = mamba_metric(base_url, metric, min(timeout, 10))
        if value > baseline:
            return value
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{metric} for pool=mamba did not increase from {baseline}")
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--target-prompt-chars", type=int, default=20000)
    ap.add_argument("--target-new-tokens", type=int, default=64)
    ap.add_argument("--num-filler", type=int, default=32)
    ap.add_argument("--filler-prompt-chars", type=int, default=12000)
    ap.add_argument("--filler-new-tokens", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=600)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result = {
        "target_first": None,
        "target_second": None,
        "match": False,
        "filler_done": 0,
        "errors": [],
        "mamba_backup_delta": 0,
        "mamba_load_delta": 0,
    }

    backup_metric = "sglang:hicache_backup_tokens_total"
    load_metric = "sglang:load_back_tokens_total"
    try:
        backup_before = mamba_metric(args.base_url, backup_metric, 10)
        load_before = mamba_metric(args.base_url, load_metric, 10)
    except Exception as exc:
        result["errors"].append(f"Mamba L2 metrics unavailable: {exc}")
        json.dump(result, open(f"{args.output}/case_result.json", "w"), indent=2)
        return 1

    target_prompt = make_prompt("TARGET", args.target_prompt_chars)

    print("[1/3] target first pass ...", flush=True)
    try:
        out1, meta1 = generate(
            args.base_url, target_prompt, args.target_new_tokens, args.request_timeout
        )
        result["target_first"] = out1
        print(f"      completion_tokens={meta1.get('completion_tokens')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"target first pass failed: {exc}")
        json.dump(result, open(f"{args.output}/case_result.json", "w"), indent=2)
        return 1
    if not out1:
        result["errors"].append("target first pass returned empty output")

    print(f"[2/3] {args.num_filler} filler requests to evict device slots ...", flush=True)
    for i in range(args.num_filler):
        try:
            generate(
                args.base_url,
                make_prompt(f"FILLER-{i}", args.filler_prompt_chars),
                args.filler_new_tokens,
                args.request_timeout,
            )
            result["filler_done"] += 1
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"filler {i} failed: {exc}")
            break
    print(f"      filler completed: {result['filler_done']}/{args.num_filler}", flush=True)

    try:
        backup_after = await_metric_increase(args.base_url, backup_metric, backup_before, 30)
        result["mamba_backup_delta"] = backup_after - backup_before
        print(f"      mamba D2H tokens={result['mamba_backup_delta']}", flush=True)
    except Exception as exc:
        result["errors"].append(f"Mamba D2H not observed: {exc}")

    print("[3/3] target replay (host L2 state H2D via kv_exchange) ...", flush=True)
    try:
        out2, meta2 = generate(
            args.base_url, target_prompt, args.target_new_tokens, args.request_timeout
        )
        result["target_second"] = out2
        print(f"      completion_tokens={meta2.get('completion_tokens')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"target replay failed: {exc}")

    try:
        load_after = await_metric_increase(args.base_url, load_metric, load_before, 30)
        result["mamba_load_delta"] = load_after - load_before
        print(f"      mamba H2D tokens={result['mamba_load_delta']}", flush=True)
    except Exception as exc:
        result["errors"].append(f"Mamba H2D not observed: {exc}")

    result["match"] = (
        bool(result["target_first"])
        and result["target_first"] == result["target_second"]
    )
    json.dump(result, open(f"{args.output}/case_result.json", "w"), indent=2)

    if result["match"] and not result["errors"]:
        print("PASS: Mamba D2H/H2D observed and replay output identical", flush=True)
        return 0
    print("FAIL: replay output diverged (or request failed)", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
