# SPDX-License-Identifier: Apache-2.0
"""Exercise an already running Clover server, without launching other GPU jobs.

Example (NVIDIA CUDA, BF16, TP1; MODEL_PATH is a local Clover checkpoint):
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
        --model-path MODEL_PATH --speculative-algorithm clover \
        --context-length 2048 --max-running-requests 4 \
        --cuda-graph-bs-decode 1 2 4 --disable-prefill-cuda-graph
    python test/manual/test_clover_arverify.py --output /tmp/clover-graph.json

For eager, add --disable-decode-cuda-graph. Save its output, then pass that
file via --reference to compare greedy outputs with a subsequent graph run.
The server is flushed between cases; use a dedicated test server.

"""

import argparse
import json
import uuid
from pathlib import Path

import requests


def check_server(base_url):
    results = {}

    def generate(name, text, sampling_params, *, flush=True):
        if flush:
            requests.post(base_url + "/flush_cache", timeout=30).raise_for_status()
        response = requests.post(
            base_url + "/generate",
            json={"text": text, "sampling_params": sampling_params},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        rows = data if isinstance(data, list) else [data]
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(rows)
        )
        for row, sp in zip(rows, params):
            assert row["meta_info"]["finish_reason"]["type"] != "abort", row
            assert row["meta_info"]["completion_tokens"] == len(row["output_ids"])
            assert len(row["output_ids"]) <= sp["max_new_tokens"]
            if sp.get("ignore_eos"):
                assert len(row["output_ids"]) == sp["max_new_tokens"]
        results[name] = data
        print(name, "passed", flush=True)
        return rows

    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n",
        "请简单介绍一下太阳系。",
    ]
    greedy = {"temperature": 0, "max_new_tokens": 96, "ignore_eos": True}
    generate("single", prompts[0], greedy)
    stop_token = results["single"]["output_ids"][5]
    stopped = generate(
        "stop_token",
        prompts[0],
        dict(greedy, ignore_eos=False, stop_token_ids=[stop_token]),
    )[0]
    first_stop = results["single"]["output_ids"].index(stop_token)
    assert stopped["output_ids"] == results["single"]["output_ids"][: first_stop + 1]
    generate("batch3", prompts, greedy)
    generate(
        "batch4_mixed",
        prompts + ["Write a short story about a fox."],
        [dict(greedy, max_new_tokens=n) for n in (1, 7, 32, 65)],
    )
    generate(
        "sampling",
        prompts[:2],
        {"temperature": 0.8, "top_k": 20, "top_p": 0.9, "max_new_tokens": 50},
    )

    # BF16 prefill chunking can flip near ties, so two cache hits are compared.
    long_prompt = (
        "The following is a test of remembering context. " * 80
        + "\nWhat is 2 + 2? Answer:"
    )
    generate("prefix_cold", long_prompt, dict(greedy, max_new_tokens=48))
    hits = []
    for _ in range(2):
        response = requests.post(
            base_url + "/generate",
            json={
                "text": long_prompt,
                "sampling_params": dict(greedy, max_new_tokens=48),
            },
            timeout=180,
        )
        response.raise_for_status()
        hits.append(response.json())
    assert all(hit["meta_info"]["cached_tokens"] > 0 for hit in hits)
    assert hits[0]["output_ids"] == hits[1]["output_ids"]
    assert hits[0]["meta_info"]["finish_reason"]["type"] == "length"
    results["prefix_hits"] = hits

    generate("presence_penalty", prompts[0], dict(greedy, presence_penalty=0.5))

    response = requests.post(
        base_url + "/generate",
        json={"text": "Hello", "sampling_params": {"min_p": 0.1}},
        timeout=30,
    )
    assert response.status_code == 400 and "min_p" in response.text

    rid = "clover-abort-" + uuid.uuid4().hex
    with requests.post(
        base_url + "/generate",
        json={
            "text": "Count all integers from one:",
            "rid": rid,
            "stream": True,
            "sampling_params": dict(greedy, max_new_tokens=1024),
        },
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        aborted = False
        finished = None
        for line in response.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if not aborted:
                requests.post(
                    base_url + "/abort_request", json={"rid": rid}, timeout=30
                ).raise_for_status()
                aborted = True
            if event.get("meta_info", {}).get("finish_reason") is not None:
                finished = event["meta_info"]["finish_reason"]
        assert aborted and finished and finished["type"] == "abort", finished
    # A slot reused right after an abort must carry no rejected state.
    after = generate("after_abort", prompts[0], greedy, flush=False)[0]
    assert after["output_ids"] == results["single"]["output_ids"]
    requests.get(base_url + "/health", timeout=30).raise_for_status()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    results = check_server(args.base_url.rstrip("/"))
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for name, data in results.items():
            if name == "sampling":
                continue
            rows = data if isinstance(data, list) else [data]
            old = (
                reference[name]
                if isinstance(reference[name], list)
                else [reference[name]]
            )
            assert [r["output_ids"] for r in rows] == [r["output_ids"] for r in old], (
                name
            )
        print("Greedy token comparison passed", flush=True)


if __name__ == "__main__":
    main()
