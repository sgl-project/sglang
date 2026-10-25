"""Evaluate a GLM-5.3 unified server or CPU-buffer PD router using sgl-eval.

Matches the full GLM-5.3 GSM8K protocol in test_glm53_eval_mi35x.py: zero-shot
chat with thinking, temperature 1.0, top_p 0.95 and a 32768-token output limit.
Record checkpoint quantization separately: the upstream 0.92 floor was defined
for FP8, while W4AFP8 also needs a unified baseline from the same checkpoint.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from sglang.test.sgl_eval_utils import run_sgl_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_sgl_eval(
        SimpleNamespace(
            base_url=args.url,
            model=args.model,
            eval_name="gsm8k",
            num_examples=None,
            num_threads=64,
            max_tokens=32768,
            temperature=1.0,
            top_p=0.95,
            seed=42,
            sgl_eval_thinking=True,
            sgl_eval_out_dir=str(args.output_dir / "gsm8k-raw"),
        )
    )
    (args.output_dir / "gsm8k.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["score"] < 0.92 or result["error_rate"] != 0:
        raise AssertionError(f"GSM8K acceptance failed: {result}")


if __name__ == "__main__":
    main()
