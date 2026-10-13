#!/usr/bin/env python3
"""Collect small SGLang requests to diagnose GPQA ending too quickly.

Run inside the same container as the evaluation, after the server is ready:
  python3 diagnose_gpqa_fast_finish.py --url http://127.0.0.1:30100 --repo /home/l00993641/sglang
Run once per branch, restarting the server after changing code.
Only standard-library dependencies. Does not edit code, stop processes, or alter
server settings. Saves exact request/response evidence in a timestamped JSON.
Forced generation is a diagnostic; it is not an accuracy fix.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def git(repo, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            timeout=20, check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def call(opener, base, path, body=None, timeout=180):
    raw = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base.rstrip("/") + path, data=raw,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            payload = response.read().decode("utf-8", errors="replace")
        data = json.loads(payload) if "event-stream" not in content_type else None
        chunks = []
        if "event-stream" in content_type:
            for line in payload.splitlines():
                if line.startswith("data:"):
                    value = line[5:].strip()
                    if value and value != "[DONE]":
                        try:
                            chunks.append(json.loads(value))
                        except json.JSONDecodeError:
                            chunks.append({"unparsed": value})
        return {"status": status, "elapsed_seconds": round(time.monotonic()-started, 3),
                "content_type": content_type, "json": data,
                "stream_chunks": chunks, "raw": payload}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "elapsed_seconds": round(time.monotonic()-started, 3),
                "error": exc.read().decode("utf-8", errors="replace")}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"elapsed_seconds": round(time.monotonic()-started, 3), "error": str(exc)}


def summary(result):
    response = result.get("json") or {}
    chunks = result.get("stream_chunks", [])
    usage = response.get("usage")
    messages, reasons, metadata = [], [], []
    if chunks:
        for chunk in chunks:
            usage = chunk.get("usage") or usage
            if chunk.get("error"):
                messages.append({"error": chunk["error"]})
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason") is not None:
                    reasons.append(choice["finish_reason"])
                delta = choice.get("delta", {})
                for key in ("content", "reasoning_content"):
                    if delta.get(key):
                        messages.append({key: delta[key]})
                if choice.get("meta_info"):
                    metadata.append(choice["meta_info"])
    else:
        for choice in response.get("choices", []):
            reasons.append(choice.get("finish_reason"))
            messages.append(choice.get("message", {}))
            if choice.get("meta_info"):
                metadata.append(choice["meta_info"])
    if response.get("meta_info"):
        metadata.append(response["meta_info"])
    return {"status": result.get("status"), "seconds": result.get("elapsed_seconds"),
            "usage": usage, "finish_reasons": reasons, "meta_info": metadata,
            "output_preview": json.dumps(messages, ensure_ascii=False)[:1500],
            "error": result.get("error") or response.get("error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30100")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    output = args.output or Path(time.strftime("gpqa_fast_finish_%Y%m%d_%H%M%S.json"))
    record = {"url": args.url, "repo": str(args.repo.resolve()),
              "git_head": git(args.repo, "rev-parse", "HEAD"),
              "git_status": git(args.repo, "status", "--short"),
              "git_diff": git(args.repo, "diff", "HEAD", "--", "python/sglang", "scripts/start_acc_test_gpqa.sh"),
              "tests": []}
    try:
        spec = importlib.util.find_spec("sglang")
        record["imported_sglang_location"] = None if spec is None else spec.origin
    except (ImportError, ValueError) as exc:
        record["imported_sglang_location"] = str(exc)
    record["models"] = call(opener, args.url, "/v1/models")
    record["server_info_before"] = call(opener, args.url, "/server_info")
    models = (record["models"].get("json") or {}).get("data", [])
    model = args.model or (models[0].get("id") if models else None)
    if not model:
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("无法读取模型 ID；请检查服务地址或传入 --model。证据已保存:", output)
        return 1
    prompt = "请逐步计算 17×23，并解释计算过程，最后给出结果。"
    common = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 64, "temperature": 1.0, "top_p": 1.0, "seed": 3407,
              "chat_template_kwargs": {"thinking": True}, "return_meta_info": True}
    cases = [
        ("chat_normal_nonstream", {"stream": False}),
        ("chat_normal_stream", {"stream": True, "stream_options": {"include_usage": True}}),
        ("chat_force_64_diagnostic", {"stream": False, "ignore_eos": True, "min_tokens": 64}),
    ]
    for name, overrides in cases:
        request = {**common, **overrides}
        print("Running:", name, flush=True)
        result = call(opener, args.url, "/v1/chat/completions", request)
        item = {"name": name, "request": request, "response": result, "summary": summary(result)}
        record["tests"].append(item)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item["summary"], ensure_ascii=False, indent=2), flush=True)
    # Native endpoint exposes token IDs and matched stop details more directly.
    # Raw text is intentionally a separate interface test, not a chat-quality comparison.
    native = {"text": prompt, "sampling_params": {"max_new_tokens": 64, "temperature": 0},
              "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 3}
    print("Running: native_generate_token_ids", flush=True)
    native_result = call(opener, args.url, "/generate", native)
    native_json = native_result.get("json") or {}
    if isinstance(native_json, dict):
        print(json.dumps({"status": native_result.get("status"), "seconds": native_result.get("elapsed_seconds"),
                          "text": native_json.get("text"), "meta_info": native_json.get("meta_info"),
                          "error": native_result.get("error")}, ensure_ascii=False, indent=2), flush=True)
    record["tests"].append({"name": "native_generate_token_ids", "request": native, "response": native_result})
    record["server_info_after"] = call(opener, args.url, "/server_info")
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完整诊断结果:", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
