"""Compare warm/cold no_buffer state after speculative finish truncation.

Launch Qwen/Qwen3.5-0.8B with no_buffer, page-size 1, overlap/CUDA graphs
disabled, and Triton attention + linear-attention backends. For --case length,
use NEXTN (3 steps, topk 1, 4 draft tokens). For --case grammar, use NGRAM
(min/max BFS breadth 1, 8 draft tokens, external SAM budget 7).

python test/manual/repro_mamba_spec_checkpoint.py --url http://localhost:30000
python test/manual/repro_mamba_spec_checkpoint.py --case grammar

The script flushes the target server's cache. Run against an isolated server.
"""

import argparse
import json
import re

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--case", choices=("length", "grammar"), default="length")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    newline = tokenizer.encode("\n", add_special_tokens=False)

    def post(path, payload=None):
        response = requests.post(args.url + path, json=payload, timeout=180)
        response.raise_for_status()
        return response

    def generate(ids, **params):
        return post(
            "/generate",
            {
                "input_ids": ids,
                "sampling_params": {"temperature": 0, **params},
                "return_logprob": True,
                "top_logprobs_num": 20,
                "logprob_start_len": -1,
            },
        ).json()

    if args.case == "length":
        cases = []
        for name, prompt, limit in (
            (
                "count-9",
                "Continue the sequence with numbers separated by spaces: 1 2 3 4 5 6 7 8 9 10",
                9,
            ),
            ("story-5", "Write a story about a cat who likes fish.", 5),
            ("story-4", "Write a story about a cat who likes fish.", 4),
            (
                "count-5",
                "Continue the sequence with numbers separated by spaces: 1 2 3 4 5 6 7 8 9 10",
                5,
            ),
            ("story-9", "Write a story about a cat who likes fish.", 9),
        ):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            cases.append((name, ids, {"max_new_tokens": limit}))
    else:
        answer = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty."
        turn = f"<|im_start|>user\nReply exactly {answer}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        post(
            "/add_external_corpus",
            {
                "corpus_id": "mamba-checkpoint-repro",
                "documents": [(turn + answer + "<|endoftext|>\n") * 3],
            },
        )
        ids = tokenizer.encode(turn, add_special_tokens=False)
        cases = [
            (
                "grammar-108",
                newline * (108 - len(ids)) + ids,
                {"max_new_tokens": 64, "regex": re.escape(answer)},
            )
        ]

    for name, ids, params in cases:
        for trial in range(args.repeat):
            post("/flush_cache")
            seed = generate(ids, **params)
            probe = ids + seed["output_ids"] + newline
            warm = generate(probe, max_new_tokens=1)
            post("/flush_cache")
            cold = generate(probe, max_new_tokens=1)
            warm_lp, cold_lp = (
                {
                    entry[1]: entry[0]
                    for entry in response["meta_info"]["output_top_logprobs"][0]
                }
                for response in (warm, cold)
            )
            common = warm_lp.keys() & cold_lp.keys()
            print(
                json.dumps(
                    {
                        "case": name,
                        "trial": trial,
                        "cached_tokens": warm["meta_info"].get("cached_tokens"),
                        "warm": warm["text"],
                        "cold": cold["text"],
                        "same_token": warm["output_ids"] == cold["output_ids"],
                        "common_top20": len(common),
                        "max_common_logprob_delta": max(
                            (abs(warm_lp[t] - cold_lp[t]) for t in common), default=None
                        ),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
