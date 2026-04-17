"""bench_eval — run lm-evaluation-harness tasks through sglang.bench_serving.

Produces one unified report with accuracy (from lm-eval, including filter-
tagged metrics like exact_match,strict-match) and serving performance
(TTFT, ITL, throughput, output-tokens-per-sec) for the same workload.

Entry points:
    run_bench_eval(...)   — programmatic; returns the merged report dict.
    main()                — CLI (added in Task 5).
"""

from __future__ import annotations

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
        },
    )

    if include_per_doc:
        # Raw lm-eval samples are in results["samples"][task].
        report["per_doc"] = results.get("samples", {}).get(task, [])

    if output_file:
        write_report(output_file, report)

    return report
