#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}/sglang/python:${PYTHONPATH:-}"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000}"
LORA_REPO="${LORA_REPO:-opherlie/lora-test-case-Qwen3.5-35B-A3B}"
LORA_NAME="${LORA_NAME:-my_lora}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
HEALTH_RETRIES="${HEALTH_RETRIES:-120}"
USE_TEST_TOKENS="${USE_TEST_TOKENS:-1}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
PROMPT="${PROMPT:-}"
RUN_BASE="${RUN_BASE:-1}"
RUN_LORA="${RUN_LORA:-1}"

SERVER_URL="${SERVER_URL}" \
LORA_REPO="${LORA_REPO}" \
LORA_NAME="${LORA_NAME}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
TOP_K="${TOP_K}" \
NUM_REQUESTS="${NUM_REQUESTS}" \
REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
HEALTH_RETRIES="${HEALTH_RETRIES}" \
USE_TEST_TOKENS="${USE_TEST_TOKENS}" \
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS}" \
PROMPT="${PROMPT}" \
RUN_BASE="${RUN_BASE}" \
RUN_LORA="${RUN_LORA}" \
python3 - <<'PY'
import os
import time

import requests


server_url = os.environ["SERVER_URL"].rstrip("/")
request_timeout = float(os.environ["REQUEST_TIMEOUT"])


def wait_for_server() -> None:
    retries = int(os.environ["HEALTH_RETRIES"])
    last_error = None
    for _ in range(retries):
        try:
            response = requests.get(
                f"{server_url}/health_generate", timeout=min(request_timeout, 5.0)
            )
            if response.status_code == 200:
                return
            last_error = RuntimeError(f"status={response.status_code} body={response.text}")
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"SGLang server did not become healthy: {last_error}")


def load_unit_test_tokens():
    import torch
    from huggingface_hub import snapshot_download

    adapter_path = snapshot_download(os.environ["LORA_REPO"], repo_type="dataset")
    cdata = torch.load(
        os.path.join(adapter_path, "compare_sample_train_data.pt"),
        weights_only=False,
    )
    tokens = cdata["tokens"]
    if torch.is_tensor(tokens):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]

    max_input_tokens = int(os.environ["MAX_INPUT_TOKENS"])
    if max_input_tokens > 0:
        tokens = tokens[:max_input_tokens]
    return tokens


def build_base_payload():
    prompt = os.environ["PROMPT"]

    fallback_prompt = (
        "Solve the problem step by step. What is 17 multiplied by 23?"
    )
    return {"text": fallback_prompt}, f"fallback text prompt ({len(fallback_prompt)} chars)"


def summarize_output(label, response_json):
    outputs = response_json if isinstance(response_json, list) else [response_json]
    output = outputs[0]
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    output_logprobs = meta.get("output_token_logprobs") or []
    token_ids = [item[1] for item in output_logprobs if len(item) >= 2]
    logprobs = [item[0] for item in output_logprobs if len(item) >= 1]
    print(f"\n[{label}]")
    print(f"text_prefix={text[:500]!r}")
    print(f"num_output_logprobs={len(output_logprobs)}")
    if token_ids:
        print(f"first_token_ids={token_ids[:16]}")
    if logprobs:
        finite_logprobs = [x for x in logprobs if x is not None]
        if finite_logprobs:
            print(
                "logprob_range="
                f"({min(finite_logprobs):.6f}, {max(finite_logprobs):.6f})"
            )


def generate(label, base_payload, lora_name=None):
    payload = dict(base_payload)
    payload["sampling_params"] = {
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        "temperature": float(os.environ["TEMPERATURE"]),
        "top_p": float(os.environ["TOP_P"]),
        "top_k": int(os.environ["TOP_K"]),
    }
    payload["return_logprob"] = True
    if lora_name is not None:
        payload["lora_path"] = lora_name

    response = requests.post(
        f"{server_url}/generate",
        json=payload,
        timeout=request_timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"{label} generate failed: status={response.status_code} body={response.text}"
        )
    summarize_output(label, response.json())


wait_for_server()
base_payload, prompt_desc = build_base_payload()
print(f"server_url={server_url}")
print(f"prompt={prompt_desc}")
print(
    "sampling="
    f"max_new_tokens={os.environ['MAX_NEW_TOKENS']} "
    f"temperature={os.environ['TEMPERATURE']} "
    f"top_p={os.environ['TOP_P']} top_k={os.environ['TOP_K']}"
)

for request_idx in range(int(os.environ["NUM_REQUESTS"])):
    print(f"\n=== request {request_idx + 1} ===")
    if os.environ["RUN_BASE"] == "1":
        generate("base", base_payload)
    if os.environ["RUN_LORA"] == "1":
        generate(f"lora:{os.environ['LORA_NAME']}", base_payload, os.environ["LORA_NAME"])
PY
