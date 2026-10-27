"""Check known-answer retrieval after forcing pageable HiCache L2 reloads.

Use an idle HiCache server with cache reports, an 8192-token device KV pool,
and a host/device ratio of 4. The reference endpoint uses the same model and
launch configuration without HiCache. Needle prompts follow the Qwen model
integration tests; response metadata proves that each case exercised L2.
This is a functional test, not a floating-point determinism test.
"""

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def evaluate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    session = requests.Session()
    records = []
    cases = []

    def save(passed=False):
        args.output.write_text(
            json.dumps(
                {"passed": passed, "cases": cases, "responses": records}, indent=2
            )
        )

    def generate(endpoint, text, max_tokens=64):
        response = session.post(
            endpoint + "/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        records.append({"endpoint": endpoint, "response": data})
        save()
        return data

    filler = "Archive note: the weather was mild and no unusual event was reported.\n"
    repeats = args.prompt_tokens // len(tokenizer.encode(filler))
    for case, depth in enumerate((0.1, 0.5, 0.9)):
        for endpoint in (args.url, args.reference_url):
            session.post(endpoint + "/flush_cache", timeout=180).raise_for_status()
        before = int(repeats * depth)
        code = str(739391 + case)
        document = (
            filler * before
            + f"IMPORTANT RECORD: The access code is {code}.\n"
            + filler * (repeats - before)
        )
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "One of these records contains an access code.\n"
                    + document
                    + "\nWhat is the access code? Reply with the digits only.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Seed both caches, then evict only the HiCache device copy.
        generate(args.url, prompt, 1)
        generate(args.reference_url, prompt, 1)
        for i in range(args.churn_requests):
            generate(
                args.url,
                f"Filler {case}-{i}. "
                + (f"Unique filler {case}-{i} about an unrelated subject. " * 300),
                8,
            )
        reloaded = generate(args.url, prompt)
        resident = generate(args.reference_url, prompt)
        host_hits = (reloaded["meta_info"].get("cached_tokens_details") or {}).get(
            "host", 0
        )
        cases.append(
            {
                "depth": depth,
                "expected": code,
                "host_tokens": host_hits,
                "reference": resident["text"],
                "reloaded": reloaded["text"],
            }
        )
        save()
        assert host_hits > 0, f"No L2 reload at depth {depth}"
        for response in (resident, reloaded):
            assert response["text"].strip() == code, (
                f"Needle retrieval failed at depth {depth}: {response}"
            )
    for endpoint in (args.url, args.reference_url):
        session.get(endpoint + "/health", timeout=30).raise_for_status()
    save(passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference-url", required=True)
    parser.add_argument(
        "--churn-requests",
        type=int,
        default=8,
        help="Increase to 16 for a 32768-token device pool",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=2500,
        help="Use 20000 for the upstream Qwen long-context needle length",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    args.reference_url = args.reference_url.rstrip("/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate(args)
