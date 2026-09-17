"""Qwen4-Exp HiCache regression: cold/device/L2 and process-restart File L3.

Run with a local Qwen3.8-Flash-Next checkpoint and the independent PLE slot-state
fix (#39862) applied to the validation environment. That fix is needed on both
the baseline and candidate builds to isolate the QSA compressed-key cache.

Example:
    python test/manual/test_qsa_hicache_e2e.py --model /models/Qwen3.8-Flash-Next-NVFP4 \
        --output /tmp/qsa-hicache-e2e
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.test.test_utils import (
    popen_launch_server,
    terminate_and_kill_process_tree,
)


def make_prompt(tokenizer, seed, tokens=12288):
    rng = random.Random(seed)
    codes = [f"{rng.randrange(10**7, 10**8)}" for _ in range(5)]
    filler = (
        "The archive contains reports about weather, transport, books and buildings. "
        "These background notes do not contain any secret access codes. "
    )
    filler_ids = tokenizer.encode(filler * 300, add_special_tokens=False)
    chunk = tokenizer.decode(filler_ids[: tokens // 5])
    sections = [f"Document identifier: {rng.getrandbits(256):064x}.\n"]
    for i, code in enumerate(codes):
        sections.append(
            f"Checkpoint {i + 1}: the secret access code is {code}.\n{chunk}\n"
        )
    sections.append(
        "Return the five secret access codes for checkpoints 1 through 5, in order. "
        "Copy each code exactly. Output only the five codes, one per line."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "".join(sections)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt, codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--port", type=int, default=30180)
    parser.add_argument("--skip-l3", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    storage = args.output / "storage"
    storage.mkdir(exist_ok=True)
    if any(storage.iterdir()):
        raise ValueError("Use a fresh output directory so the cold leg cannot hit L3")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [make_prompt(tokenizer, 39830 + i) for i in range(args.samples)]
    url = f"http://127.0.0.1:{args.port}"
    results, failures = [], []
    launch_args = [
        "--tp",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.92",
        "--context-length",
        "24576",
        "--max-total-tokens",
        "32768",
        "--page-size",
        "64",
        "--max-running-requests",
        "4",
        "--chunked-prefill-size",
        "4096",
        "--cuda-graph-max-bs-decode",
        "4",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--max-mamba-cache-size",
        "48",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "64",
        "--linear-attn-decode-backend",
        "flashinfer",
        "--linear-attn-prefill-backend",
        "flashinfer",
        "--ple-offload-embedding",
        "--enable-cache-report",
        "--enable-hierarchical-cache",
        "--hicache-size",
        "24",
        "--hicache-write-policy",
        "write_through",
        "--hicache-mem-layout",
        "page_first",
        "--hicache-io-backend",
        "direct",
        "--hicache-storage-backend",
        "file",
        "--hicache-storage-prefetch-policy",
        "wait_complete",
    ]
    if args.mtp:
        launch_args.extend(
            [
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ]
        )
    env = {**os.environ, "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(storage)}

    def launch(name):
        log = open(args.output / f"{name}.log", "w")
        try:
            process = popen_launch_server(
                args.model,
                url,
                timeout=1800,
                other_args=launch_args,
                env=env,
                return_stdout_stderr=(log, log),
            )
        finally:
            log.close()
        return process

    def generate(prompt, codes, sample, leg, source, max_tokens=96):
        start = time.monotonic()
        response = requests.post(
            f"{url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        meta, text = payload["meta_info"], payload["text"]
        hits = meta.get("cached_tokens_details") or {}
        found = [code in text for code in codes]
        result = {
            "sample": sample,
            "leg": leg,
            "seconds": time.monotonic() - start,
            "expected": codes,
            "found": found,
            "text": text,
            "meta_info": meta,
        }
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        if codes and not all(found):
            failures.append(f"sample {sample} {leg}: only {sum(found)}/5 codes")
        if source == "cold" and meta.get("cached_tokens", 0) != 0:
            failures.append(
                f"sample {sample} {leg}: expected a cold request, got {hits}"
            )
        if source in ("device", "host", "storage"):
            if hits.get(source, 0) < meta["prompt_tokens"] - 128:
                failures.append(
                    f"sample {sample} {leg}: expected full {source} hit, got {hits}"
                )
        return meta

    process = launch("l2-server")
    try:
        for i, (prompt, codes) in enumerate(prompts):
            generate(prompt, codes, i, "cold", "cold")
            generate(prompt, codes, i, "device", "device")
            uncached = 0
            for j in range(4):
                churn, _ = make_prompt(tokenizer, 900000 + 4 * i + j)
                meta = generate(churn, [], i, f"churn-{j}", None, max_tokens=1)
                uncached += meta["prompt_tokens"] - meta.get("cached_tokens", 0)
            if uncached <= 32768 * 1.35:
                failures.append(f"sample {i}: insufficient device churn ({uncached})")
            generate(prompt, codes, i, "host", "host")
            generate(prompt, codes, i, "host2", "device")
        # Let background file writes drain before terminating the process.
        time.sleep(10)
    finally:
        terminate_and_kill_process_tree(process)

    if not args.skip_l3:
        process = launch("l3-server")
        try:
            for i, (prompt, codes) in enumerate(prompts):
                generate(prompt, codes, i, "storage", "storage")
                generate(prompt, codes, i, "storage-device", "device")
        finally:
            terminate_and_kill_process_tree(process)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise AssertionError("\n".join(failures))
    print("All cache-tier and content assertions passed", flush=True)


if __name__ == "__main__":
    main()
