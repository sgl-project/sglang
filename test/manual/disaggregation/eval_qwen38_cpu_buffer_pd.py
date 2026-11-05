"""Evaluate an externally launched Qwen3.8 unified server or CPU-buffer PD router.

Uses the GSM8K recipe and long-context needle prompts from
 test/registered/e2e/disaggregation/test_qwen4_exp_pd_prefill_cp.py.
Launch matching unified and PD endpoints separately, then run this script once
per endpoint with distinct output directories. This does not launch servers.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace


def evaluate_gsm8k(args):
    from sglang.test.sgl_eval_utils import run_sgl_eval

    result = run_sgl_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=args.url,
            model=args.model,
            num_examples=200,
            num_threads=32,
            max_tokens=16384,
            temperature=None,
            top_p=None,
            chat_template_kwargs={"thinking": True},
            sgl_eval_out_dir=str(args.output_dir / "gsm8k-raw"),
        )
    )
    (args.output_dir / "gsm8k.json").write_text(json.dumps(result, indent=2) + "\n")
    # Keep length-limited responses in the evaluator's denominator.
    if result["score"] < 0.94 or result["error_rate"] != 0:
        raise AssertionError(f"GSM8K acceptance failed: {result}")
    return result


def evaluate_needle(args):
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    filler = (
        "Archive note: the weather was mild, the office lights were on, "
        "and no unusual event was reported.\n"
    )
    repeats = 20000 // len(tokenizer.encode(filler))
    results = []
    for depth in (0.1, 0.5, 0.9):
        before = int(repeats * depth)
        document = (
            filler * before
            + "IMPORTANT RECORD: The access code is 739391.\n"
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
        start = time.monotonic()
        response = requests.post(
            args.url.rstrip("/") + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        results.append(
            {"depth": depth, "elapsed": time.monotonic() - start, "response": body}
        )
        (args.output_dir / "needle.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        if "739391" not in body["text"]:
            raise AssertionError(f"Needle retrieval failed at depth {depth}: {body}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="Unified endpoint or PD router URL"
    )
    parser.add_argument(
        "--model", required=True, help="Model path/ID also used by the server"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("gsm8k", "needle", "all"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("gsm8k", "all"):
        evaluate_gsm8k(args)
    if args.mode in ("needle", "all"):
        evaluate_needle(args)


if __name__ == "__main__":
    main()
