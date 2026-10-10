"""Score SemIf JSONL decisions using its original prompts and an SGLang server."""

import argparse
import json
from pathlib import Path

import requests
from semif_phase1.direct import PROMPT_VERSION, encode_prompt
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    with args.output.open("x") as output, requests.Session() as session:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = [encode_prompt(tokenizer, row, args.max_tokens) for row in batch]
            response = session.post(
                args.base_url.rstrip("/") + "/v1/score",
                json={
                    "model": args.model,
                    "query": [],
                    "items": [ids for ids, _, _ in encoded],
                    "label_token_ids": [slots for _, slots, _ in encoded],
                    "apply_softmax": True,
                    "temperature": args.temperature,
                    "return_token_logprobs": True,
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            for row, (ids, _, prompt_hash), probabilities, logprobs in zip(
                batch, encoded, result["scores"], result["token_logprobs"], strict=True
            ):
                output.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "option_ids": [option["id"] for option in row["options"]],
                            "probabilities": probabilities,
                            "token_logprobs": logprobs,
                            "input_tokens": len(ids),
                            "prompt_sha256": prompt_hash,
                            "prompt_version": PROMPT_VERSION,
                            "temperature": args.temperature,
                            "readout": "sglang candidate token logprobs",
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
