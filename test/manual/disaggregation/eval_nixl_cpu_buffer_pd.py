"""Validate externally launched unified/PD endpoints with upstream test helpers.

Run ``parity-reference`` against a fresh unified server, then ``parity`` against
its matching PD router. Both must use identical TP/backend/page/chunk settings.
Run ``gsm8k`` against each endpoint with the same evaluator arguments. Server
lifecycle and cross-node placement are external so this works with SSH hosts.
The parity tolerance is fixed before either run; failures never widen it.
"""

import argparse
import json
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from transformers import AutoTokenizer

from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.run_eval import run_eval_once
from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8KEval


class EndpointParity(PDLogprobParityMixin, unittest.TestCase):
    parity_max_new_tokens = 8
    parity_logprob_delta = 1e-3


def boundary_prompts(model, page_size, chunk_size):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    filler = tokenizer.encode(
        "Archive note: the weather was mild, the office lights were on, "
        "and no unusual event was reported. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode("\nSummarize the archive:", add_special_tokens=False)
    lengths = sorted(
        {n + d for n in (page_size, chunk_size, 2 * chunk_size) for d in (-1, 0, 1)}
        | {4 * chunk_size + 1, 8 * chunk_size + 1}
    )
    if lengths[0] < len(question):
        raise ValueError("Page boundary is too short for the parity question")
    body = filler * (max(lengths) // len(filler) + 1)
    return [body[: n - len(question)] + question for n in lengths]


def health(url):
    response = requests.get(url.rstrip("/") + "/health", timeout=30)
    response.raise_for_status()


def compare(case, reference, actual, label):
    errors = []
    try:
        case._assert_logprob_parity(reference=reference, actual=actual, label=label)
    except AssertionError as exc:
        errors.append(str(exc))
    expected = reference["output_token_logprobs"]
    observed = actual["output_token_logprobs"]
    matching_ids = [x[1] for x in expected] == [x[1] for x in observed]
    # Logprobs after different token histories are not comparable.
    delta = (
        max(abs(a[0] - b[0]) for a, b in zip(expected, observed))
        if matching_ids and expected
        else None
    )
    return {
        "token_ids_equal": matching_ids,
        "max_logprob_delta": delta,
        "errors": errors,
    }


def run_parity(args):
    case = EndpointParity()
    report = {
        "mode": args.mode,
        "model": args.model,
        "tp": args.tp,
        "page_size": args.page_size,
        "chunk_size": args.chunk_size,
        "max_new_tokens": case.parity_max_new_tokens,
        "logprob_delta": case.parity_logprob_delta,
        "cases": [],
    }
    if args.mode == "parity-reference":
        prompts = boundary_prompts(args.model, args.page_size, args.chunk_size)
        references = None
        flush_urls = [args.url]
    else:
        if not args.reference or not args.prefill_url or not args.decode_url:
            raise ValueError("parity requires reference, prefill-url and decode-url")
        ref = json.loads(Path(args.reference).read_text())
        for key in (
            "model",
            "tp",
            "page_size",
            "chunk_size",
            "max_new_tokens",
            "logprob_delta",
        ):
            if ref[key] != report[key]:
                raise ValueError(f"Reference configuration mismatch: {key}")
        if not ref.get("passed"):
            raise ValueError("Unified repeatability check did not pass")
        references = ref["cases"]
        prompts = [row["input_ids"] for row in references]
        flush_urls = [args.prefill_url, args.decode_url]
    health(args.url)
    for i, prompt in enumerate(prompts):
        for url in flush_urls:
            case._flush_cache(url)
        actual = case._generate(base_url=args.url, input_ids=prompt)
        if references is None:
            case._flush_cache(args.url)
            reference = case._generate(base_url=args.url, input_ids=prompt)
        else:
            reference = references[i]["actual"]
        row = {
            "input_ids": prompt,
            "actual": actual,
            "reference": reference,
            **compare(case, reference, actual, f"prompt_tokens={len(prompt)}"),
        }
        report["cases"].append(row)
        print(
            json.dumps(
                {
                    "prompt_tokens": len(prompt),
                    **{
                        k: row[k]
                        for k in ("token_ids_equal", "max_logprob_delta", "errors")
                    },
                }
            ),
            flush=True,
        )
        Path(args.output).write_text(json.dumps(report, indent=2))
    health(args.url)
    for url in flush_urls:
        health(url)
    report["passed"] = all(not row["errors"] for row in report["cases"])
    Path(args.output).write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("Parity failed; see the complete per-prompt report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity-reference", "parity", "gsm8k"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--prefill-url")
    parser.add_argument("--decode-url")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--num-examples", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-data-path")
    parser.add_argument("--accuracy-threshold", type=float, default=0.83)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode != "gsm8k":
        run_parity(args)
        return
    health(args.url)
    # Match the Qwen3.5-35B PP test's five-shot completion protocol. The
    # evaluator excludes the five demonstration questions from the scored set.
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    config = SimpleNamespace(
        model=args.model,
        api="completion",
        max_tokens=args.max_tokens,
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        num_shots=5,
        temperature=0.0,
        top_p=1.0,
    )
    evaluator = GSM8KEval(
        num_examples=config.num_examples,
        num_threads=config.num_threads,
        num_shots=config.num_shots,
        data_path=args.gsm8k_data_path,
    )
    start = time.time()
    result, latency, sampler = run_eval_once(
        config, args.url.rstrip("/") + "/v1", evaluator
    )
    health(args.url)
    errors = []
    if len(result.convos) != args.num_examples:
        errors.append("Evaluator did not score the requested number of examples")
    if len(sampler._completion_tokens) != args.num_examples:
        errors.append("Missing successful completion responses or usage records")
    if any(not convo[-1]["content"].strip() for convo in result.convos):
        errors.append("Empty completion response")
    if result.score < args.accuracy_threshold:
        errors.append(f"Accuracy {result.score} < {args.accuracy_threshold}")
    baseline_score = None
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        if not reference.get("passed"):
            errors.append("Unified reference did not pass")
        if reference["config"] != vars(config) or reference["tp"] != args.tp:
            errors.append("Unified reference configuration mismatch")
        baseline_score = reference["score"]
        if result.score < baseline_score - args.max_accuracy_drop:
            errors.append("PD accuracy drop exceeds the configured tolerance")
        if [c[0] for c in reference["convos"]] != [c[0] for c in result.convos]:
            errors.append("Unified and PD evaluated different prompts")
    report = {
        "mode": args.mode,
        "tp": args.tp,
        "config": vars(config),
        "score": result.score,
        "metrics": result.metrics,
        "accuracy_threshold": args.accuracy_threshold,
        "max_accuracy_drop": args.max_accuracy_drop,
        "baseline_score": baseline_score,
        "convos": result.convos,
        "completion_tokens": sampler._completion_tokens,
        "responses_at_token_limit": sum(
            n == args.max_tokens for n in sampler._completion_tokens
        ),
        "elapsed_s": time.time() - start,
        "evaluator_latency_s": latency,
        "errors": errors,
        "passed": not errors,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("convos", "completion_tokens")
            }
        ),
        flush=True,
    )
    if errors:
        raise AssertionError("GSM8K acceptance failed; see the saved report")
    print("EVALUATION_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
