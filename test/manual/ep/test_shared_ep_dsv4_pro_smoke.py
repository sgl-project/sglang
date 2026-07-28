"""Manual 8xMI355X DSV4-Pro SharedEP control/TBO/MTP smoke test."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


def main() -> None:
    model = os.environ.get(
        "SGLANG_SHARED_EP_DSV4_MODEL",
        "/models/DeepSeek-V4-Pro",
    )
    base_url = os.environ.get(
        "SGLANG_SHARED_EP_SMOKE_URL",
        "http://127.0.0.1:30000",
    )
    enable_tbo = os.environ.get("SGLANG_SHARED_EP_TBO", "0") == "1"
    enable_graph = os.environ.get("SGLANG_SHARED_EP_GRAPH", "0") == "1"
    mtp_steps = int(os.environ.get("SGLANG_SHARED_EP_MTP_STEPS", "0"))
    rounds = int(os.environ.get("SGLANG_SHARED_EP_SMOKE_ROUNDS", "1"))
    if rounds < 1:
        raise ValueError("SGLANG_SHARED_EP_SMOKE_ROUNDS must be positive")
    mem_fraction_static = "0.87" if mtp_steps else "0.85"
    os.environ.setdefault("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "64")
    os.environ.setdefault("AITER_FLYDSL_FORCE", "1")
    os.environ.setdefault("SGLANG_SHARED_EP_DIRECT_SMALL_BATCH", "1")
    args = [
        "--tp-size",
        "8",
        "--ep-size",
        "8",
        "--dp-size",
        "8",
        "--enable-dp-attention",
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
        "--moe-a2a-backend",
        "shared_ep",
        "--moe-runner-backend",
        "aiter",
        "--ep-dispatch-algorithm",
        "fake",
        "--attention-backend",
        "dsv4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--page-size",
        "256",
        "--swa-full-tokens-ratio",
        "0.1",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "8192",
        "--mem-fraction-static",
        mem_fraction_static,
        "--watchdog-timeout",
        "1800",
        "--random-seed",
        "42",
        "--load-balance-method",
        "round_robin",
        "--skip-server-warmup",
        "--trust-remote-code",
    ]
    if enable_graph:
        args.extend(["--cuda-graph-max-bs-decode", "8"])
    else:
        args.append("--disable-cuda-graph")
    if enable_tbo:
        args.append("--enable-two-batch-overlap")
    if mtp_steps:
        args.extend(
            [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-steps",
                str(mtp_steps),
                "--speculative-num-draft-tokens",
                str(mtp_steps + 1),
                "--speculative-moe-a2a-backend",
                "mori",
                "--speculative-moe-runner-backend",
                "aiter",
            ]
        )

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10,
        other_args=args,
        env=dict(os.environ),
    )
    try:

        def generate(index: int):
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "text": f"Request {index}: The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                    },
                },
                timeout=900,
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("text")
            if not isinstance(text, str) or not text:
                raise RuntimeError(f"SharedEP smoke returned no text: {result}")
            return text

        request_count = 2 if enable_tbo else 1
        started = time.perf_counter()
        outputs = []
        all_outputs = []
        with ThreadPoolExecutor(max_workers=request_count) as executor:
            for round_index in range(rounds):
                offset = round_index * request_count
                outputs = list(
                    executor.map(
                        generate,
                        range(offset, offset + request_count),
                    )
                )
                all_outputs.extend(outputs)
        elapsed = time.perf_counter() - started
        requests_completed = rounds * request_count
        print(
            "SharedEP DSV4-Pro smoke: "
            f"mtp_steps={mtp_steps}, tbo={enable_tbo}, graph={enable_graph}, "
            f"requests={requests_completed}, elapsed={elapsed:.3f}s, "
            f"request_rate={requests_completed / elapsed:.3f}/s, "
            f"last_outputs={outputs!r}"
        )
        output_path = os.environ.get("SGLANG_SHARED_EP_OUTPUT_PATH")
        if output_path:
            with open(output_path, "w", encoding="utf-8") as stream:
                json.dump(all_outputs, stream)
    finally:
        kill_process_tree(process.pid)


if __name__ == "__main__":
    main()
