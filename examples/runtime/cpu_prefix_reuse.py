"""Measure complete responses to synthetic maintenance tickets sharing a manual.

Run against a fresh SGLang server launched with --enable-cache-report.
This example uses exact tokenizer-produced IDs and reports errors alongside latency.
See CPU_PREFIX_REUSE.md for CPU launch options and the cache-off comparison.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

MANUAL = """You are a maintenance assistant for a synthetic cooling station.
Classify CURRENT, not HISTORY, using the first matching rule:
1. sensor_valid=false: diagnosis=sensor_fault, action=CHECK_SENSOR.
2. temperature_c>=60 and fan_rpm<500: diagnosis=cooling_fault, action=FAN_ON.
3. load>=90: diagnosis=load_rise, action=KEEP.
4. Otherwise: diagnosis=normal, action=KEEP.
Return a complete JSON object with three string fields: diagnosis, action, reason.
The reason must be a short explanation. No Markdown or extra fields.

Equipment knowledge: the fan cools the motor enclosure. fan_rpm describes the
cooling fan, rather than the motor shaft. A zero fan speed during a cool idle
period does not alone imply a cooling fault. sensor_valid comes from the local
input check; its value is authoritative even if the temperature appears normal.
History is context only: a previous fault does not override a recovered current
sample, and a previous valid probe does not repair a currently invalid one.

Maintenance procedures: for a sensor fault, inspect the temperature-probe
connection and obtain a valid sample before diagnosing the cooling system.
For a cooling fault, keep local protective cooling enabled and inspect fan
obstruction and the cooling path. For a high load, retain local controls and
inspect the process load trend. For normal operation, continue monitoring.

Action boundaries: KEEP does not clear an alarm or force cooling off. FAN_ON
requests cooling only. CHECK_SENSOR requests inspection. Local MCU protection
continues independently; this assistant is not a safety controller. A gateway
must validate any action before transmitting it to a device. The example only
generates text and never controls physical equipment.
"""

CASES = [
    (
        {"sensor_valid": True, "temperature_c": 45, "fan_rpm": 1000, "load": 30},
        "normal",
    ),
    (
        {"sensor_valid": True, "temperature_c": 75, "fan_rpm": 200, "load": 50},
        "cooling_fault",
    ),
    (
        {"sensor_valid": False, "temperature_c": -40, "fan_rpm": 1000, "load": 30},
        "sensor_fault",
    ),
]


def stream_completion(client, url, payload, cache_report_enabled):
    started = time.perf_counter()
    first = None
    texts = []
    usage = None
    done = False
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        lines = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                lines.append(line[5:].lstrip())
                if sum(map(len, lines)) > 65536:
                    raise ValueError("oversized SSE event")
                continue
            if line or not lines:
                continue
            event = "\n".join(lines)
            lines.clear()
            if event == "[DONE]":
                done = True
                break
            chunk = json.loads(event)
            if "error" in chunk:
                raise ValueError(f"backend error: {chunk['error']}")
            for choice in chunk.get("choices", []):
                text = choice.get("text", "")
                if text:
                    if first is None:
                        first = time.perf_counter() - started
                    texts.append(text)
            if chunk.get("usage"):
                usage = chunk["usage"]
    if not done or first is None:
        raise ValueError("incomplete token-bearing response")
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if cache_report_enabled:
        if usage is None:
            raise ValueError("cache reporting was enabled but final usage is missing")
        # The tested SGLang server omits prompt_tokens_details for a zero hit.
        cached = 0 if cached is None else cached
    if cached is not None and (
        type(cached) is not int or not 0 <= cached <= len(payload["prompt"])
    ):
        raise ValueError("invalid cached-token count")
    return {
        "text": "".join(texts),
        "ttft_s": first,
        "full_s": time.perf_counter() - started,
        "cached_tokens": cached,
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def parse_output(text):
    """Normalize an outer Markdown fence, while retaining raw JSON validity."""
    raw_json = True
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raw_json = False
        stripped = text.strip()
        if not (stripped.startswith("```") and stripped.endswith("```")):
            raise
        first_line, separator, body = stripped.partition("\n")
        if not separator or first_line not in {"```", "```json"}:
            raise ValueError("unsupported output wrapper")
        value = json.loads(body[:-3].strip())
    if not (
        isinstance(value, dict)
        and set(value) == {"diagnosis", "action", "reason"}
        and all(isinstance(v, str) for v in value.values())
    ):
        raise ValueError("output does not match the three-string schema")
    return value, raw_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", required=True, help="server's served-model-name")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--cache-report-enabled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failed = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": MANUAL}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("x", encoding="utf-8") as output,
        httpx.Client(timeout=180, trust_env=False) as client,
    ):
        for index, (current, expected) in enumerate(CASES):
            messages = [
                {"role": "system", "content": MANUAL},
                {"role": "user", "content": json.dumps({"CURRENT": current})},
            ]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if ids[: len(prefix)] != prefix:
                raise ValueError(
                    "system prefix is not exact in the final serialization"
                )
            row = {
                "index": index,
                "input_tokens": len(ids),
                "manual_prefix_tokens": len(prefix),
                "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "expected": expected,
                "format_valid": False,
                "raw_json_valid": False,
                "diagnosis_correct": False,
            }
            try:
                result = stream_completion(
                    client,
                    args.base_url.rstrip("/") + "/completions",
                    {
                        "model": args.model,
                        "prompt": ids,
                        "max_tokens": 96,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    args.cache_report_enabled,
                )
                row.update(result, status="completed")
                try:
                    value, raw_json = parse_output(result["text"])
                    row.update(
                        format_valid=True,
                        raw_json_valid=raw_json,
                        diagnosis_correct=value["diagnosis"] == expected,
                    )
                except (ValueError, TypeError):
                    pass
            except (ValueError, httpx.HTTPError) as error:
                failed = True
                row.update(status="error", error=str(error))
            line = json.dumps(row, ensure_ascii=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
