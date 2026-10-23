"""Exercise GPQA request shapes and 32K admission budgets without grading answers."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from sgl_eval.evals._prompts import render_prompt, resolve_prompt
from sgl_eval.evals._registry import _resolve_upstream_metadata
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--stop-tokens", type=int, default=32)
parser.add_argument("--prepare-only", action="store_true")
options = parser.parse_args()
assert options.rounds > 0
assert 0 < options.stop_tokens <= 32768
assert Path(options.tag).name == options.tag
root = Path("/task/results")
output = root / (options.tag + ".requests.json")
assert not output.exists(), "Use a new tag to preserve previous evidence"
tokenizer = AutoTokenizer.from_pretrained(
    "/models/GLM-5.2-w4a8", trust_remote_code=True
)
examples = json.loads((root / "gpqa-diamond-198-seed42.json").read_text())[:8]
_, prompt_name = _resolve_upstream_metadata("gpqa")
prompt_path = resolve_prompt(prompt_name)


def token_ids(content):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


short = [
    token_ids(render_prompt(prompt_path, problem=row["problem"])) for row in examples
]
long = [
    token_ids(
        "A scientific note describes cells, energy, and physical laws. " * 320
        + f"\nWhat is {index + 2} + 2? Explain briefly."
    )
    for index in range(8)
]
record = {
    "scope": "Liveness only; no scoring; max_new_tokens32768 preserved for scheduler admission",
    "tag": options.tag,
    "started_unix": time.time(),
    "requests": [],
    "input_lengths": {"short": list(map(len, short)), "long": list(map(len, long))},
    "input_sha256": {
        "short": [
            hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in short
        ],
        "long": [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in long],
    },
}
lock = threading.Lock()


def save():
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output)


def post(route, body, timeout=180):
    req = urllib.request.Request(
        "http://127.0.0.1:8810" + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def run_one(index, ids, case, round_index):
    rid = options.tag + "-" + uuid.uuid4().hex
    state = {
        "rid": rid,
        "case": case,
        "round": round_index,
        "index": index,
        "prompt_tokens": len(ids),
        "completion_tokens": 0,
        "started_unix": time.time(),
        "state": "pending",
    }
    with lock:
        record["requests"].append(state)
        save()
    try:
        with post(
            "/generate",
            {
                "rid": rid,
                "input_ids": ids,
                "stream": True,
                "sampling_params": {
                    "max_new_tokens": 32768,
                    "temperature": 0,
                    "top_p": 0.95,
                },
            },
        ) as response:
            for raw in response:
                if not raw.startswith(b"data: ") or raw[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(raw[6:])
                assert "error" not in event, event.get("error")
                count = event.get("meta_info", {}).get("completion_tokens", 0)
                with lock:
                    state["completion_tokens"] = count
                    state["updated_unix"] = time.time()
                    state["state"] = "generating"
                    save()
                if count >= options.stop_tokens:
                    with post("/abort_request", {"rid": rid}) as aborted:
                        aborted.read()
                    break
        assert state["completion_tokens"] > 0, "Request produced no token"
        with lock:
            state["state"] = "complete"
            state["finished_unix"] = time.time()
            save()
    except BaseException as error:
        with lock:
            state["state"] = "failed"
            state["error"] = repr(error)
            save()
        raise


save()
if options.prepare_only:
    assert max(map(len, short + long)) + 32768 <= 36864, (
        "Request exceeds the configured context"
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "input_lengths": record["input_lengths"],
                "requests_sent": 0,
            }
        ),
        flush=True,
    )
    raise SystemExit(0)
with urllib.request.urlopen(
    "http://127.0.0.1:8810/server_info", timeout=30
) as response:
    info = json.load(response)
context_length = info.get("server_args", info)["context_length"]
assert max(map(len, short + long)) + 32768 <= context_length, (
    "Request exceeds server context"
)
try:
    for round_index in range(options.rounds):
        for case, rows in [("short", short), ("long", long)]:
            record["active_case"] = case
            record["active_round"] = round_index
            save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(run_one, i, ids, case, round_index)
                    for i, ids in enumerate(rows)
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            time.sleep(0.5)
    record["passed"] = True
finally:
    record["finished_unix"] = time.time()
    save()
print(
    json.dumps(
        {
            "passed": True,
            "requests": len(record["requests"]),
            "short_lengths": record["input_lengths"]["short"],
            "long_lengths": record["input_lengths"]["long"],
        }
    ),
    flush=True,
)
