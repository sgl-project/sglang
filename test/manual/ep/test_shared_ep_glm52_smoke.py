"""Manual 8xMI355X GLM-5.2 block-FP8 pull-cache prefill smoke test."""

from __future__ import annotations

import os
import time

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


def main() -> None:
    model = os.environ.get(
        "SGLANG_SHARED_EP_GLM_MODEL",
        "/models/GLM-5.2-FP8",
    )
    base_url = os.environ.get(
        "SGLANG_SHARED_EP_SMOKE_URL",
        "http://127.0.0.1:30000",
    )
    rounds = int(os.environ.get("SGLANG_SHARED_EP_SMOKE_ROUNDS", "1"))
    if rounds < 1:
        raise ValueError("SGLANG_SHARED_EP_SMOKE_ROUNDS must be positive")
    moe_a2a_backend = os.environ.get(
        "SGLANG_SHARED_EP_MOE_A2A_BACKEND",
        "shared_ep",
    )
    os.environ.setdefault("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")
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
        moe_a2a_backend,
        "--moe-runner-backend",
        "aiter",
        "--ep-dispatch-algorithm",
        "fake",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--disable-radix-cache",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "8192",
        "--mem-fraction-static",
        os.environ.get("SGLANG_SHARED_EP_MEM_FRACTION_STATIC", "0.84"),
        "--watchdog-timeout",
        "1800",
        "--random-seed",
        "42",
        "--skip-server-warmup",
        "--trust-remote-code",
        "--disable-cuda-graph",
    ]

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10,
        other_args=args,
        env=dict(os.environ),
    )
    try:
        started = time.perf_counter()
        outputs = []
        for index in range(rounds):
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
                raise RuntimeError(f"SharedEP GLM smoke returned no text: {result}")
            outputs.append(text)
        elapsed = time.perf_counter() - started
        print(
            "SharedEP GLM-5.2 smoke: "
            f"backend={moe_a2a_backend}, "
            f"requests={rounds}, elapsed={elapsed:.3f}s, "
            f"request_rate={rounds / elapsed:.3f}/s, outputs={outputs!r}"
        )
    finally:
        kill_process_tree(process.pid)


if __name__ == "__main__":
    main()
