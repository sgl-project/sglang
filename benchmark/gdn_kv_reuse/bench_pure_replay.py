"""Pure 8192-token input; no query suffix is appended."""

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
    tokens = tokenizer.encode(text, add_special_tokens=False)[:8705]
    encoded = [
        tokenizer.encode(value, add_special_tokens=False)
        for value in "0123456789abcdef"
    ]
    assert all(len(x) == 1 for x in encoded), encoded
    variants = [x[0] for x in encoded][: args.concurrency]
    assert len(set(variants)) == args.concurrency
    seeds = []
    for token in variants:
        seed = tokens.copy()
        seed[4096] = token
        seeds.append(seed)
    other = tokenizer.encode(" Different branch.", add_special_tokens=False)
    assert other[0] not in variants
    branch = tokens[:4096] + (other * 256)[:256]
    targets = [seed[:8192] for seed in seeds]
    (out / "inputs.json").write_text(
        json.dumps(dict(seeds=seeds, branch=branch, targets=targets))
    )

    def generate(ids, rid, n=1):
        batched = isinstance(ids[0], list)
        response = session.post(
            args.url + "/generate",
            json=dict(
                input_ids=ids,
                rid=rid,
                sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
                return_logprob=True,
                top_logprobs_num=5,
                logprob_start_len=[len(x) - 1 for x in ids]
                if batched
                else len(ids) - 1,
            ),
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    rows = []
    for i in range(args.repeats):
        session.post(args.url + "/flush_cache", timeout=60).raise_for_status()
        for j, seed in enumerate(seeds):
            generate(seed[:8449], f"{args.rid_prefix}prepare-{i}-{j}")
            generate(seed, f"{args.rid_prefix}seed-{i}-{j}")
        if not args.zero_prefix:
            generate(branch[: args.branch_length], f"{args.rid_prefix}branch-{i}")
        start = time.perf_counter()
        responses = generate(
            targets,
            [f"{args.rid_prefix}target-{i}-{j}" for j in range(args.concurrency)],
            args.new_tokens,
        )
        wave_ms = 1000 * (time.perf_counter() - start)
        assert len(responses) == args.concurrency
        assert all(
            r["meta_info"]["cached_tokens"] == (0 if args.zero_prefix else 4096)
            for r in responses
        ), [r["meta_info"]["cached_tokens"] for r in responses]
        assert all(len(r["output_ids"]) == args.new_tokens for r in responses)
        row = dict(iteration=i, targets=responses, wave_ms=wave_ms)
        rows.append(row)
        (out / "responses.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(dict(iteration=i, wave_ms=wave_ms)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--url", default="http://127.0.0.1:18090")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rid-prefix", default="")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--new-tokens", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--zero-prefix", action="store_true")
    p.add_argument("--branch-length", type=int, default=4352)
    run(p.parse_args())
