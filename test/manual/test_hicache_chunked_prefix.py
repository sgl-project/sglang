#!/usr/bin/env python3
"""Exercise chunked-prefix reuse and host restoration through a running server.

Use an isolated GLM-5.3-Flash server with page_size=64, index_kpool=4,
chunked_prefill_size=max_prefill_tokens=320, HiCache write-through, cache
reporting, metrics, and the glm47 tool parser enabled. Set device KV capacity to
twice --pressure-tokens (8192 with the default). Host capacity must be larger.
Run this client with CUDA hidden. It never imports SGLang or executes kernels.

Cold repeats establish the numerical comparison. Different requests then reuse
512- and 768-token prefixes, both on device and after counters prove a host hit.
Only the last 64 prompt tokens are scored, avoiding a full-prompt logits buffer.
Scored requests generate one token; the primer generates 32 to reach a checkpoint.
A separate coding request must complete valid read_file tool calls naturally,
including after device eviction and confirmed host restoration.
The numerical smoke check does not establish exact checkpoint correctness;
test_compressed_dsa_checkpoint_roundtrip.py checks the saved state contents.
"""

import argparse
import json
import math
import os
import re
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

CORPUS = """This is a code review of a small job queue. The queue stores immutable
job descriptions and a separate status record. A worker claims a queued job,
executes it once, and records either success or a retryable failure. Duplicate
notifications must not execute a completed job again. The API must validate
inputs before writing a job. Tests should cover two workers racing to claim
the same job, retries after a timeout, and a process restarting between claim
and completion. A completed job's result must survive a restart. The cache is
only an optimization: evicting it must not change which job the worker sees.
The implementation lives in src/queue.py and tests live in tests/test_queue.py.
"""


def request(base, path, body=None):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return response.read()


def metrics(base):
    result = {}
    for line in request(base, "/metrics").decode().splitlines():
        if not line.startswith("sglang:prefill_effective_tokens_total{"):
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
        if labels.get("tp_rank") not in (None, "0"):
            continue
        mode = labels.get("mode")
        result[mode] = result.get(mode, 0) + float(line.rsplit(" ", 1)[1])
    if not result:
        raise AssertionError("Host-hit metrics are required to prove restoration")
    result.setdefault("host_hit", 0)
    result.setdefault("device_hit", 0)
    return result


def generate(base, ids, salt, *, score=False, max_new_tokens=1):
    body = {
        "input_ids": ids,
        "cache_salt": salt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
    }
    if score:
        body.update(return_logprob=True, logprob_start_len=len(ids) - 64)
    return json.loads(request(base, "/generate", body))


def scores(response):
    values = response["meta_info"]["input_token_logprobs"]
    return [(float(lp), token) for lp, token, *_ in values if lp is not None]


def compare(reference, actual):
    a, b = scores(reference), scores(actual)
    assert len(a) >= 60 and len(a) == len(b), (len(a), len(b))
    assert [x[1] for x in a] == [x[1] for x in b]
    delta = [abs(x[0] - y[0]) for x, y in zip(a, b, strict=True)]
    assert all(math.isfinite(d) for d in delta)
    return {
        "tokens": len(delta),
        "mean_abs_logprob_delta": statistics.mean(delta),
        "max_abs_logprob_delta": max(delta),
    }


def valid_tool_response(response):
    choice = response["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if choice["finish_reason"] == "length" or not calls:
        return False
    paths = []
    for call in calls:
        try:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            if (
                function["name"] != "read_file"
                or not isinstance(arguments, dict)
                or set(arguments) != {"path"}
                or arguments["path"] not in ("src/queue.py", "tests/test_queue.py")
            ):
                return False
            paths.append(arguments["path"])
        except (KeyError, TypeError, ValueError):
            return False
    # Reading the referenced tests is a valid extra action. Require the source
    # read and well-formed calls, without demanding identical generated text.
    return "src/queue.py" in paths and len(paths) == len(set(paths))


def coding_tool_request(base, salt):
    body = {
        "model": "glm-5.3-flash",
        "messages": [
            {"role": "system", "content": "You are a careful programming assistant."},
            {
                "role": "user",
                "content": CORPUS
                * 12
                + "\nReview src/queue.py for concurrency bugs. Begin by reading that file. "
                "Do not infer its implementation from the design notes above.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a source file before reviewing its implementation.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "cache_salt": salt,
        "temperature": 0,
        "max_tokens": 2048,
    }
    response = json.loads(request(base, "/v1/chat/completions", body))
    return {"response": response, "passed": valid_tool_response(response)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pressure-tokens", type=int, default=4096)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a careful programming assistant."},
            {"role": "user", "content": "DOCUMENT_PLACEHOLDER"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    lead, tail = template.split("DOCUMENT_PLACEHOLDER")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    lead_ids, tail_ids = encode(lead), encode(tail)
    corpus_ids = encode(CORPUS * 100)

    def prompt(length, heading="Review this design.\n"):
        front = lead_ids + encode(heading)
        needed = length - len(front) - len(tail_ids)
        assert 0 < needed <= len(corpus_ids)
        return front + corpus_ids[:needed] + tail_ids

    primer = prompt(1000)
    results = {"definition": __doc__, "cases": [], "tool_cases": [], "passed": True}
    nonce = uuid.uuid4().hex
    for shared in (512, 768):
        # The second request has genuinely different content after the shared
        # prefix, so its state must not include the primer's following tokens.
        branch = (
            primer[:shared]
            + encode(
                "\n\nThe excerpt above ends here. A different review now follows. "
                "Please explain why two workers cannot both own one queue job. "
                "Discuss atomic claims, idempotency keys and recovery after a "
                "worker crashes. A status cache may disappear at any time. "
                "The durable job record is authoritative. A database transaction "
                "must make the claim conditional on the queued status. "
                "How should a regression reproduce a crash between recording "
                "a successful result and acknowledging the notification?"
            )
            + tail_ids
        )
        common = next(
            (i for i, (a, b) in enumerate(zip(primer, branch)) if a != b),
            min(len(primer), len(branch)),
        )
        assert common // 256 * 256 == shared, common
        salt = f"{nonce}-{shared}-shared"
        cold = generate(args.base_url, branch, f"{salt}-cold-a", score=True)
        repeat = generate(args.base_url, branch, f"{salt}-cold-b", score=True)
        assert cold["meta_info"]["cached_tokens"] == 0
        assert repeat["meta_info"]["cached_tokens"] == 0
        control = compare(cold, repeat)
        # Chunked insertions alone do not initiate write-through. Let normal
        # decoding cross 1024 so finishing publishes this path to host too.
        primed = generate(args.base_url, primer, salt, max_new_tokens=32)
        assert primed["meta_info"]["completion_tokens"] >= 25
        time.sleep(1)  # allow write-through completion before inducing eviction
        row = {
            "shared_tokens": shared,
            "primer_tokens": len(primer),
            "branch_tokens": len(branch),
            "primer_ids": primer,
            "branch_ids": branch,
            "cold_repeat": control,
            "cold_scores": scores(cold),
            "repeat_scores": scores(repeat),
            "replays": [],
        }
        # BF16 state / FP8 KV can differ slightly with prefill partitioning.
        # Keep the same bounds for the broken and repaired revisions.
        mean_limit = max(0.05, 5 * control["mean_abs_logprob_delta"])
        max_limit = max(0.5, 5 * control["max_abs_logprob_delta"])
        row["limits"] = {"mean": mean_limit, "max": max_limit}
        results["cases"].append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "shared_tokens": shared,
                    "cold_repeat": control,
                    "limits": row["limits"],
                }
            ),
            flush=True,
        )
        for cycle in range(args.cycles + 1):
            if cycle:
                # Do not replay between pressure requests: doing so refreshes
                # this entry's LRU position and can prevent its eviction.
                for pressure_round in range(2):
                    pressure = prompt(
                        args.pressure_tokens,
                        f"Independent queue audit {nonce} {shared} {cycle} {pressure_round}.\n",
                    )
                    generate(
                        args.base_url,
                        pressure,
                        f"{salt}-pressure-{cycle}-{pressure_round}",
                    )
                time.sleep(1)
            before = metrics(args.base_url)
            replay = generate(args.base_url, branch, salt, score=True)
            after = metrics(args.base_url)
            delta = compare(cold, replay)
            cached = replay["meta_info"]["cached_tokens"]
            hit = {k: after.get(k, 0) - before.get(k, 0) for k in after}
            passed = (
                delta["mean_abs_logprob_delta"] <= mean_limit
                and delta["max_abs_logprob_delta"] <= max_limit
                and cached >= shared
                and (
                    hit["host_hit"] >= shared
                    if cycle
                    else hit["device_hit"] >= shared and hit["host_hit"] == 0
                )
            )
            row["replays"].append(
                {
                    "path": "host" if cycle else "device",
                    "cycle": cycle,
                    "cached_tokens": cached,
                    "hit_tokens": hit,
                    "scores": scores(replay),
                    **delta,
                    "passed": passed,
                }
            )
            results["passed"] &= passed
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in row["replays"][-1].items() if k != "scores"}
                ),
                flush=True,
            )
    tool_salt = f"{nonce}-coding-tool"
    for cycle in range(args.cycles + 2):
        if cycle >= 2:
            for pressure_round in range(2):
                generate(
                    args.base_url,
                    prompt(args.pressure_tokens),
                    f"{tool_salt}-pressure-{cycle}-{pressure_round}",
                )
            time.sleep(1)
        before = metrics(args.base_url)
        row = coding_tool_request(args.base_url, tool_salt)
        after = metrics(args.base_url)
        hit = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        row.update(cycle=cycle, hit_tokens=hit)
        if cycle == 0:
            row["passed"] &= hit["device_hit"] == hit["host_hit"] == 0
        elif cycle == 1:
            row["passed"] &= hit["device_hit"] >= 256 and hit["host_hit"] == 0
        else:
            row["passed"] &= hit["host_hit"] >= 256
        results["passed"] &= row["passed"]
        results["tool_cases"].append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "response"}), flush=True)
    raise SystemExit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
