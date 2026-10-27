"""Evaluate a running Qwen3.8 HiCache or reference endpoint with upstream GSM8K.

Uses test_qwen4_exp_pd_prefill_cp's 200-example thinking recipe and 94% gate.
Run once per otherwise idle endpoint, using the same checkpoint and settings.
The upstream test uses NVFP4; keep the checkpoint precision in reported results.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.sgl_eval_utils import run_sgl_eval


def evaluate(args):
    requests.post(args.url.rstrip("/") + "/flush_cache", timeout=180).raise_for_status()
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
            sgl_eval_out_dir=str(args.output_dir / "raw"),
        )
    )
    (args.output_dir / "gsm8k.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["score"] >= 0.94 and result["error_rate"] == 0, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluate(args)
