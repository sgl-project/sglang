"""Exercise real HTTP cache matching/scheduling; no cache/state injection."""

import argparse
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    text = "\n".join(
        f"Record {i}: analyze the cache state, preserve token order, and explain the numerical result."
        for i in range(3000)
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)[:12545]
    other = tokenizer.encode(" Different branch.", add_special_tokens=False)
    assert other[0] != tokens[4096]
    branch = tokens[:4096] + (other * 256)[:256]
    tail = tokenizer.encode(
        "\nSummarize the preceding records briefly.", add_special_tokens=False
    )
    assert tail[0] != tokens[8192]
    target = tokens[:8192] + tail
    (out / "inputs.json").write_text(
        json.dumps(dict(seed=tokens, branch=branch, target=target))
    )

    def generate(ids, tag, n=1):
        body = dict(
            input_ids=ids,
            rid=args.rid_prefix + tag,
            sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
            return_logprob=True,
            top_logprobs_num=5,
            logprob_start_len=len(ids) - 1,
        )
        start = time.perf_counter()
        response = session.post(args.url + "/generate", json=body, timeout=300)
        response.raise_for_status()
        data = response.json()
        data["client_ms"] = (time.perf_counter() - start) * 1000
        return data

    results = []
    for i in range(args.repeats):
        r = session.post(args.url + "/flush_cache", timeout=60)
        r.raise_for_status()
        seed = generate(tokens, f"seed-{i}")
        split = generate(branch, f"branch-{i}")
        result = generate(target, f"target-{i}", args.new_tokens)
        assert result["meta_info"]["cached_tokens"] == 4096, result["meta_info"][
            "cached_tokens"
        ]
        row = dict(iteration=i, seed=seed, branch=split, target=result)
        if args.new_tokens > 1:
            row["warm_control"] = generate(target, f"warm-control-{i}", args.new_tokens)
            assert row["warm_control"]["meta_info"]["cached_tokens"] == 8192
        results.append(row)
        (out / "responses.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                dict(
                    iteration=i,
                    target_meta=result.get("meta_info"),
                    client_ms=result["client_ms"],
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rid-prefix", default="")
    p.add_argument("--model-path", required=True)
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--new-tokens", type=int, default=32)
    run(p.parse_args())
