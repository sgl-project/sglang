"""bench_eval — run lm-evaluation-harness tasks through sglang.bench_serving.

Produces one unified report with accuracy (from lm-eval, including filter-
tagged metrics like exact_match,strict-match) and serving performance
(TTFT, ITL, throughput, output-tokens-per-sec) for the same workload.

Entry points:
    run_bench_eval(...)   — programmatic; returns the merged report dict.
    main()                — CLI (added in Task 5).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional


def run_bench_eval(
    *,
    task: str,
    base_url: str,
    backend: str,
    model: str,
    tokenizer_path: str,
    num_fewshot: int,
    limit: Optional[int],
    max_gen_toks: int,
    request_rate: float,
    max_concurrency: Optional[int],
    apply_chat_template: bool,
    enable_thinking: bool,
    output_file: Optional[str],
    include_per_doc: bool,
    fewshot_as_multiturn: bool = False,
    flush_cache: bool = False,
    extra_request_body: Optional[Dict[str, Any]] = None,
    system_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    import lm_eval
    from transformers import AutoTokenizer

    from sglang.benchmark.eval_harness import (
        BenchServingLM, merge_report, write_report,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    lm = BenchServingLM(
        base_url=base_url,
        backend=backend,
        model_id=model,
        tokenizer=tokenizer,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        enable_thinking=enable_thinking,
        flush_cache=flush_cache,
        extra_request_body=extra_request_body,
    )

    gen_kwargs = f"max_gen_toks={max_gen_toks}"

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        system_instruction=system_instruction,
        gen_kwargs=gen_kwargs,
        batch_size=max_concurrency or "auto",
    )

    if lm.last_perf is None:
        raise RuntimeError(
            f"simple_evaluate({task!r}) returned without invoking generate_until. "
            f"Possible causes: (1) task is not generative — check its output_type "
            f"(expected 'generate_until'); (2) limit={limit} filtered out all "
            f"instances; (3) task produced zero requests."
        )

    report = merge_report(
        task_name=task,
        lm_eval_results=results,
        perf=lm.last_perf,
        run_config={
            "backend": backend,
            "model": model,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "max_gen_toks": max_gen_toks,
            "num_fewshot": num_fewshot,
            "apply_chat_template": apply_chat_template,
            "enable_thinking": enable_thinking,
            "limit": limit,
            "sampling": extra_request_body or {},
            "system_instruction": system_instruction,
        },
    )

    if include_per_doc:
        # Raw lm-eval samples are in results["samples"][task].
        report["per_doc"] = results.get("samples", {}).get(task, [])

    if output_file:
        write_report(output_file, report)

    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sglang.bench_eval",
        description=("Run an lm-eval task through sglang.bench_serving. "
                     "Reports both accuracy (lm-eval) and serving performance "
                     "(TTFT, ITL, throughput) for the same workload."),
    )
    p.add_argument("--task", required=True,
                   help="lm-eval task name (must be generative). Examples: "
                        "gsm8k, mmlu_flan_cot_zeroshot, gpqa_diamond_cot_zeroshot, "
                        "mmlu_pro.")
    p.add_argument("--base-url", required=True,
                   help="SGLang server base URL, e.g. http://127.0.0.1:30000.")
    p.add_argument("--backend", default="sglang-oai",
                   choices=["sglang", "sglang-oai"])
    p.add_argument("--model", required=True, help="Model id (sent as model field).")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer path or repo id.")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of eval docs (None = full task).")
    p.add_argument("--max-gen-toks", type=int, default=32768,
                   help="Max output tokens per request. Default 32768 (Qwen3 recommendation). "
                        "Use 38912 for competition math/code.")
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Requests per second. 'inf' = unlimited (default).")
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--apply-chat-template", action="store_true")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Adds enable_thinking=True to apply_chat_template.")
    p.add_argument("--fewshot-as-multiturn", action="store_true")
    # Sampling parameters (Qwen3 recommended defaults applied automatically).
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature. Default: 0.6 (thinking) or 0.7 (non-thinking).")
    p.add_argument("--top-p", type=float, default=None,
                   help="Top-p sampling. Default: 0.95 (thinking) or 0.8 (non-thinking).")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0,
                   help="Presence penalty. Recommended 1.5 for quantized models.")
    p.add_argument("--output-file", default=None,
                   help="Append-mode JSONL path for the merged report.")
    p.add_argument("--include-per-doc", action="store_true")
    p.add_argument("--flush-cache", action="store_true",
                   help="Flush KV cache before the run (CI parity).")
    p.add_argument("--system-instruction", default=None,
                   help="System message prepended to every prompt. "
                        "Math: 'Please reason step by step, and put your final answer within \\boxed{}.' "
                        "MCQ: 'Please show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".'")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.enable_thinking and not args.apply_chat_template:
        parser.error("--enable-thinking requires --apply-chat-template")

    temperature = args.temperature if args.temperature is not None else (0.6 if args.enable_thinking else 0.7)
    top_p = args.top_p if args.top_p is not None else (0.95 if args.enable_thinking else 0.8)
    sampling: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
    }
    if args.presence_penalty != 0.0:
        sampling["presence_penalty"] = args.presence_penalty

    report = run_bench_eval(
        task=args.task,
        base_url=args.base_url,
        backend=args.backend,
        model=args.model,
        tokenizer_path=args.tokenizer,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        max_gen_toks=args.max_gen_toks,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        apply_chat_template=args.apply_chat_template,
        enable_thinking=args.enable_thinking,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        output_file=args.output_file,
        include_per_doc=args.include_per_doc,
        flush_cache=args.flush_cache,
        extra_request_body=sampling,
        system_instruction=args.system_instruction,
    )

    print("=" * 60)
    print(f"Task: {report['task']}")
    print(f"N samples: {report['n_samples']}")
    print("Accuracy:")
    for k, v in report["accuracy"].items():
        print(f"  {k}: {v}")
    print("Performance:")
    for k, v in report["perf"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
