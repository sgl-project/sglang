"""Aligned GLM-5.2 batch-64 SharedEP serving benchmark.

The workload mirrors SGLang PR #32482: DP8 + DP-Attention + EP8, 8K input,
1K output, global batch 64, and a DP-adjusted 1K-token prefill chunk.
ROCm uses MoRI+AITER as the materialized baseline and TileLang DSA kernels.
"""

from __future__ import annotations

import os
import subprocess
import sys

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


def main() -> None:
    model = os.environ.get(
        "SGLANG_SHARED_EP_GLM_MODEL",
        "/models/models--zai-org--GLM-5.2-FP8",
    )
    backend = os.environ.get("SGLANG_SHARED_EP_MOE_A2A_BACKEND", "shared_ep")
    if backend not in ("shared_ep", "mori"):
        raise ValueError(f"unsupported ROCm benchmark backend: {backend}")
    base_url = os.environ.get(
        "SGLANG_SHARED_EP_BENCHMARK_URL",
        "http://127.0.0.1:30000",
    )
    output_file = os.environ.get(
        "SGLANG_SHARED_EP_BENCHMARK_OUTPUT",
        f"/tmp/glm52-{backend}-batch64.jsonl",
    )
    num_prompts = os.environ.get("SGLANG_SHARED_EP_NUM_PROMPTS", "64")
    input_len = os.environ.get("SGLANG_SHARED_EP_INPUT_LEN", "8192")
    output_len = os.environ.get("SGLANG_SHARED_EP_OUTPUT_LEN", "1024")
    concurrency = os.environ.get("SGLANG_SHARED_EP_CONCURRENCY", "64")
    warmup_requests = os.environ.get("SGLANG_SHARED_EP_WARMUP_REQUESTS", "1")
    profile_enabled = os.environ.get("SGLANG_SHARED_EP_PROFILE", "0") == "1"
    profile_steps = os.environ.get("SGLANG_SHARED_EP_PROFILE_STEPS", "4")
    os.environ.setdefault("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")
    os.environ.setdefault("SGLANG_USE_AITER", "1")

    server_args = [
        "--tokenizer-path",
        model,
        "--trust-remote-code",
        "--json-model-override-args",
        '{"qk_rope_head_dim":64}',
        "--kv-cache-dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.85",
        "--max-running-requests",
        "256",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--schedule-policy",
        "fcfs",
        "--schedule-conservativeness",
        "0.3",
        "--page-size",
        "64",
        "--swa-full-tokens-ratio",
        "0.8",
        "--tp-size",
        "8",
        "--dp-size",
        "8",
        "--enable-dp-attention",
        "--ep-size",
        "8",
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
        "--moe-a2a-backend",
        backend,
        "--moe-runner-backend",
        "aiter",
        "--ep-dispatch-algorithm",
        "fake",
        "--attention-backend",
        "dsa",
        "--dsa-prefill-backend",
        "tilelang",
        "--dsa-decode-backend",
        "tilelang",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-max-bs-decode",
        "32",
        "--cuda-graph-bs-decode",
        "1",
        "2",
        "4",
        "8",
        "12",
        "16",
        "20",
        "24",
        "28",
        "32",
        "--random-seed",
        "20260730",
        "--host",
        "127.0.0.1",
        "--skip-server-warmup",
    ]
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 12,
        other_args=server_args,
        env=dict(os.environ),
    )
    try:
        host, port = base_url.removeprefix("http://").rsplit(":", 1)
        command = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--host",
            host,
            "--port",
            port,
            "--dataset-name",
            "random-ids",
            "--model",
            model,
            "--tokenizer",
            model,
            "--num-prompts",
            num_prompts,
            "--random-input-len",
            input_len,
            "--random-output-len",
            output_len,
            "--random-range-ratio",
            "1.0",
            "--request-rate",
            "inf",
            "--max-concurrency",
            concurrency,
            "--output-file",
            output_file,
            "--output-details",
            "--disable-tqdm",
            "--temperature",
            "0",
            "--seed",
            "20260730",
            "--flush-cache",
            "--warmup-requests",
            warmup_requests,
            "--tokenize-prompt",
            "--cache-report",
        ]
        if profile_enabled:
            command.extend(
                [
                    "--profile",
                    "--profile-by-stage",
                    "--profile-num-steps",
                    profile_steps,
                    "--profile-stages",
                    "prefill",
                    "decode",
                    "--profile-activities",
                    "CPU",
                    "GPU",
                    "--profile-prefix",
                    backend,
                ]
            )
        print(f"Running aligned GLM-5.2 benchmark: backend={backend}")
        print(" ".join(command))
        subprocess.run(command, check=True, timeout=7200)
        print(f"Benchmark result: {output_file}")
    finally:
        kill_process_tree(process.pid)


if __name__ == "__main__":
    main()
