"""Eight concurrent targets with independent replay paths and a shared prefix."""

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
    variants = [
        tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(8)
    ]
    assert len(set(variants)) == 8
    c, h, seed_length = (
        (256, 512, 1025) if args.small_batch_check else (4096, 8192, 12545)
    )
    tokens = tokens[:seed_length]
    seeds = []
    for token in variants:
        seed = tokens.copy()
        seed[c] = token
        seeds.append(seed)
    other = tokenizer.encode(" Different branch.", add_special_tokens=False)
    assert other[0] not in variants
    branch = tokens[:c] + (other * 256)[:256]
    tail = tokenizer.encode(
        "\nSummarize the preceding records briefly.", add_special_tokens=False
    )
    assert tail[0] != tokens[h]
    targets = [seed[:h] + tail for seed in seeds]
    if args.small_batch_check:
        targets = [seeds[0][:h] for _ in range(8)]
        if args.small_batch_check == "mixed":
            targets[4:] = [seed[:h] + tail for seed in seeds[4:]]
    (out / "inputs.json").write_text(
        json.dumps(dict(seeds=seeds, branch=branch, targets=targets))
    )

    def generate(ids, rid, n=1):
        response = session.post(
            args.url + "/generate",
            json=dict(
                input_ids=ids,
                rid=rid,
                sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
                return_logprob=True,
                top_logprobs_num=5,
                logprob_start_len=(
                    [len(x) - 1 for x in ids]
                    if isinstance(ids[0], list)
                    else len(ids) - 1
                ),
            ),
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    results = []
    for i in range(args.repeats):
        session.post(args.url + "/flush_cache", timeout=60).raise_for_status()
        # Distinct continuations prevent an earlier target from warming another
        # target's GDN state. All eight retain the same C=4096 checkpoint.
        for j, seed in enumerate(seeds):
            generate(seed, f"{args.rid_prefix}seed-{i}-{j}")
        generate(branch, f"{args.rid_prefix}branch-{i}")
        start = time.perf_counter()
        responses = generate(
            targets,
            [f"{args.rid_prefix}target-{i}-{j}" for j in range(8)],
            args.new_tokens,
        )
        elapsed = time.perf_counter() - start
        assert len(responses) == 8
        for response in responses:
            assert response["meta_info"]["cached_tokens"] == c, response
        row = dict(iteration=i, targets=responses, wave_ms=elapsed * 1000)
        if args.new_tokens > 1:
            row["warm_controls"] = generate(
                targets,
                [f"{args.rid_prefix}control-{i}-{j}" for j in range(8)],
                args.new_tokens,
            )
            for ids, response in zip(targets, row["warm_controls"]):
                expected_hit = h if len(ids) > h else c
                assert response["meta_info"]["cached_tokens"] == expected_hit, response
        results.append(row)
        (out / "responses.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(dict(iteration=i, wave_ms=row["wave_ms"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--small-batch-check", choices=["all-reuse", "mixed"])
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rid-prefix", default="")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--new-tokens", type=int, default=32)
    run(p.parse_args())
