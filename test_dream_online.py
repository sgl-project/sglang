"""Minimal HTTP smoke test for a running Dream SGLang server."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://127.0.0.1:30000"),
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "16")),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=float(os.getenv("STARTUP_TIMEOUT", "600")),
    )
    return parser.parse_args()


def request_json(url, payload=None, timeout=60):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc

    return json.loads(body) if body else {}


def wait_until_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(f"Dream server was not ready after {timeout:.0f}s: {last_error}")


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"Waiting for Dream server: {base_url}")
    wait_until_ready(base_url, args.startup_timeout)

    models = request_json(f"{base_url}/v1/models")
    assert models.get("data"), models
    model_id = models["data"][0]["id"]
    print(f"Loaded model: {model_id}")

    response = request_json(
        f"{base_url}/v1/completions",
        {
            "model": model_id,
            "prompt": args.prompt,
            "max_tokens": args.max_new_tokens,
            "temperature": 0.0,
        },
        timeout=max(60, args.startup_timeout),
    )
    choices = response.get("choices", [])
    assert choices and isinstance(choices[0].get("text"), str), response

    print("Dream online smoke test passed")
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
