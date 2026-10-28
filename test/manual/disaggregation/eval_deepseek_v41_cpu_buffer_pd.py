"""Evaluate an existing DeepSeek-V4.1 unified server or CPU-buffer PD router.

Uses the GSM8K protocol from sglang.test.dsv41_flash_common:
200 zero-shot chat questions, 64 client threads and a 4096-token output limit.
The upstream accuracy floor is 0.95. Run once per endpoint with separate output
paths; this script does not launch servers or enable speculative decoding.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from sglang.test.sgl_eval_utils import run_sgl_eval


def evaluate_boundaries(args):
    import requests
    from transformers import AutoTokenizer

    from sglang.srt.entrypoints.openai.encoding_dsv41 import encode_messages

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt = encode_messages(
        [
            {
                "role": "user",
                "content": "Ignore this filler: INSERT_FILLER_HERE\nWhat is 19 + 23? Reply with the number only.",
            }
        ],
        thinking_mode="chat",
    )
    before, after = prompt.split("INSERT_FILLER_HERE")
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    filler = tokenizer.encode(" x", add_special_tokens=False)[0]
    results = []
    # Odd/even compression, SWA window, KV page and prefill-chunk boundaries.
    for length in (127, 128, 129, 255, 256, 257, 2047, 2048, 2049, 4097):
        ids = prefix + [filler] * (length - len(prefix) - len(suffix)) + suffix
        assert len(ids) == length
        response = requests.post(
            args.url.rstrip("/") + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        results.append({"input_tokens": length, "response": body})
        (args.output_dir / "boundaries.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        assert body["meta_info"]["finish_reason"]["type"] != "abort", body
        assert "42" in body["text"], body
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--model", required=True, help="Model name served by the endpoint"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("gsm8k", "boundaries", "all"), default="gsm8k"
    )
    parser.add_argument(
        "--tokenizer", help="Checkpoint path, required for boundary tests"
    )
    args = parser.parse_args()
    if args.mode != "gsm8k" and not args.tokenizer:
        parser.error("--tokenizer is required for boundary tests")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("boundaries", "all"):
        evaluate_boundaries(args)
    if args.mode == "boundaries":
        return
    result = run_sgl_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=args.url,
            model=args.model,
            num_examples=200,
            num_threads=64,
            max_tokens=4096,
            temperature=None,
            top_p=None,
            chat_template_kwargs=None,
            sgl_eval_out_dir=str(args.output_dir / "gsm8k-raw"),
        )
    )
    (args.output_dir / "gsm8k.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["score"] < 0.95 or result["error_rate"] != 0:
        raise AssertionError(f"GSM8K acceptance failed: {result}")


if __name__ == "__main__":
    main()
