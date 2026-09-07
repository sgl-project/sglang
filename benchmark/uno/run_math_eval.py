"""Run full-dataset AR or speculative-decoding math evaluation offline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark.uno.math_data import (
    BENCHMARKS,
    BenchmarkConfig,
    get_benchmark,
    prepare_benchmark_data,
)
from benchmark.uno.math_grader import score_math

CONTEXT_LENGTH = 40960
MAX_TOKENS = 2**15
SPECULATIVE_ALGORITHMS = ("EAGLE", "EAGLE3", "DFLASH", "UNO")
SPECULATIVE_OPTION_NAMES = (
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_draft_model_revision",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_dflash_block_size",
    "speculative_draft_attention_backend",
    "uno_lora_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--speculative-algorithm",
        type=str.upper,
        choices=SPECULATIVE_ALGORITHMS,
    )
    parser.add_argument("--speculative-draft-model-path")
    parser.add_argument("--speculative-draft-model-revision")
    parser.add_argument("--speculative-num-steps", type=int)
    parser.add_argument("--speculative-eagle-topk", type=int)
    parser.add_argument("--speculative-num-draft-tokens", type=int)
    parser.add_argument("--speculative-dflash-block-size", type=int)
    parser.add_argument("--speculative-draft-attention-backend")
    parser.add_argument("--uno-lora-path")
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=tuple(BENCHMARKS),
        help="Benchmark to run; repeat to select multiple (default: all).",
    )
    parser.add_argument("--limit", type=int, help="Maximum problems per benchmark.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    _validate_args(parser=parser, args=args)
    return args


def _validate_args(
    *, parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for name in (
        "num_samples",
        "max_running_requests",
        "context_length",
        "max_tokens",
        "top_k",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")


def _context_reserve(args: argparse.Namespace) -> int:
    width = args.speculative_num_draft_tokens
    if width is None:
        width = args.speculative_dflash_block_size or 1
    if args.speculative_algorithm == "DFLASH":
        return 2 * width
    if args.speculative_algorithm == "UNO" and (args.speculative_eagle_topk or 1) > 1:
        draft_width = (args.speculative_num_steps or 0) + 1
        return max(width, draft_width) + 1
    return width


def _model_ref(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _first_value(row: dict[str, Any], *names: str) -> Any | None:
    return next((row[name] for name in names if row.get(name) is not None), None)


def _format_prompt(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    benchmark: BenchmarkConfig,
) -> tuple[list[int], str]:
    messages = [dict(message) for message in messages]
    system_message = next(
        (message for message in messages if message.get("role") == "system"),
        None,
    )
    if system_message is None:
        messages.insert(0, {"role": "system", "content": benchmark.instruction})
    elif benchmark.instruction not in system_message["content"]:
        existing = system_message["content"].strip()
        system_message["content"] = f"{benchmark.instruction}\n\n{existing}"

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **benchmark.chat_template_kwargs,
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **benchmark.chat_template_kwargs,
    )
    return list(token_ids), str(rendered)


def _prepare_prompts(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    context_reserve: int,
) -> list[dict[str, Any]]:
    prompts = []
    for benchmark_name in args.benchmark or BENCHMARKS:
        benchmark = get_benchmark(benchmark_name)
        data_path = prepare_benchmark_data(
            benchmark.name,
            output_dir=args.data_root,
        )
        rows = _read_jsonl(data_path)
        rows = rows[: args.limit] if args.limit is not None else rows
        for row_index, row in enumerate(rows):
            input_ids, rendered = _format_prompt(
                tokenizer=tokenizer,
                messages=row["chat_input"],
                benchmark=benchmark,
            )
            available = args.context_length - len(input_ids) - context_reserve
            if available < args.max_tokens:
                source = _first_value(row, "id", "problem_id", "index", "row")
                source = row_index if source is None else source
                raise ValueError(
                    f"{benchmark.name}:{source} has only {max(0, available)} "
                    "completion tokens available; increase --context-length"
                )
            source = _first_value(row, "id", "problem_id", "index", "row")
            source = row_index if source is None else source
            for sample_index in range(args.num_samples):
                prompt_id = f"{source}:sample{sample_index}"
                prompts.append(
                    {
                        "id": prompt_id,
                        "benchmark": benchmark.name,
                        "input_ids": input_ids,
                        "row": {
                            "id": prompt_id,
                            "source_row": source,
                            "sample_index": sample_index,
                            "problem": rendered,
                            "ground_truth": row["ground_truth"],
                            "prompt_token_count": len(input_ids),
                            "resolved_max_tokens": args.max_tokens,
                        },
                    }
                )
    if not prompts:
        raise ValueError("No prompts were prepared")
    return prompts


def _engine_options(
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tokenizer_path = _model_ref(args.tokenizer_path or args.model_path)
    options: dict[str, Any] = {
        "model_path": _model_ref(args.model_path),
        "tokenizer_path": tokenizer_path,
        "skip_tokenizer_init": True,
        "context_length": args.context_length,
        "dtype": args.dtype,
        "random_seed": args.random_seed,
        "max_running_requests": args.max_running_requests,
        "attention_backend": args.attention_backend,
        "log_level": "info",
    }
    if args.revision is not None:
        options["revision"] = args.revision
    for name in SPECULATIVE_OPTION_NAMES:
        value = getattr(args, name)
        if value is not None:
            if name in (
                "speculative_draft_model_path",
                "uno_lora_path",
            ):
                value = _model_ref(value)
            options[name] = value
    return options


def _generate(
    *,
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    engine_options: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    import sglang as sgl

    engine = sgl.Engine(**engine_options)
    sampling = [
        {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
        for _ in prompts
    ]
    try:
        start = perf_counter()
        outputs = engine.generate(
            input_ids=[prompt["input_ids"] for prompt in prompts],
            sampling_params=sampling,
            rid=[f"{prompt['benchmark']}:{prompt['id']}" for prompt in prompts],
        )
        elapsed = perf_counter() - start
    finally:
        engine.shutdown()
    return outputs, elapsed


def _build_rows(
    *,
    prompts: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    tokenizer: Any,
    speculative_algorithm: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for prompt, output in zip(prompts, outputs, strict=True):
        token_ids = list(output["output_ids"])
        metadata = output["meta_info"]
        verify_forwards = int(metadata.get("spec_verify_ct", 0))
        if speculative_algorithm == "UNO":
            # Both UNO pathways are full target-model forward passes.
            num_forwards = 2 * verify_forwards
        elif speculative_algorithm is not None:
            # EAGLE and DFLASH TPF conventionally count target verification.
            num_forwards = verify_forwards
        else:
            num_forwards = len(token_ids)
        rows.append(
            {
                "benchmark": prompt["benchmark"],
                **prompt["row"],
                "output_ids": token_ids,
                "generation": tokenizer.decode(token_ids, skip_special_tokens=True),
                "num_tokens": len(token_ids),
                "num_forwards": num_forwards,
                "tokens_per_forward": _divide(len(token_ids), num_forwards),
                "sglang_meta_info": metadata,
            }
        )
    return rows


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(row["num_tokens"] for row in rows)
    forwards = sum(row["num_forwards"] for row in rows)
    tpfs = [
        row["tokens_per_forward"]
        for row in rows
        if row["tokens_per_forward"] is not None
    ]
    return {
        "num_tokens": tokens,
        "num_forwards": forwards,
        "tokens_per_forward": _divide(tokens, forwards),
        "unweighted_mean_tokens_per_forward": (sum(tpfs) / len(tpfs) if tpfs else None),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| Dataset | Accuracy | TPF | tok/s | tok/s/request |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["by_benchmark"].items():
        tokens_per_second = metrics.get("tokens_per_second")
        per_request = metrics.get("tokens_per_second_per_request")
        tokens_per_forward = metrics.get("tokens_per_forward")
        tps = f"{tokens_per_second:.2f}" if tokens_per_second is not None else "—"
        request_tps = f"{per_request:.2f}" if per_request is not None else "—"
        tpf = f"{tokens_per_forward:.3f}" if tokens_per_forward is not None else "—"
        lines.append(
            f"| {name} | {metrics['accuracy']:.2%} | {tpf} | {tps} | {request_tps} |"
        )
    lines.extend(
        [
            "| **Average** | "
            f"**{summary['macro_average_accuracy']:.2%}** | "
            f"**{summary['macro_average_tokens_per_forward']:.3f}** | "
            f"**{summary['tokens_per_second']:.2f}** | "
            f"**{summary['tokens_per_second_per_request']:.2f}** |",
            "",
            "Accuracy and TPF in the Average row are unweighted means across "
            "datasets. Throughput is aggregate output tokens divided by timed "
            "generation seconds; tok/s/request divides it by max running requests.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    generation_seconds: float,
    engine_options: dict[str, Any],
) -> None:
    rows_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output_row = {key: value for key, value in row.items() if key != "benchmark"}
        rows_by_benchmark[row["benchmark"]].append(output_row)

    benchmark_metrics = {}
    totals = {"prompts": 0, "completions": 0, "correct": 0, "tokens": 0, "forwards": 0}
    request_tpfs = []
    for name, benchmark_rows in rows_by_benchmark.items():
        output_dir = args.output_dir / name
        _write_jsonl(output_dir / "generations.jsonl", benchmark_rows)
        graded, scores = score_math(benchmark_rows)
        _write_jsonl(output_dir / "grades.jsonl", graded)
        _write_json(output_dir / "scores.json", scores)

        metrics = {
            "num_prompts": scores["num_problems"],
            "num_completions": scores["num_rows"],
            "num_correct": scores["num_correct"],
            "accuracy": scores["accuracy"],
            **_benchmark_summary(benchmark_rows),
        }
        benchmark_metrics[name] = metrics
        totals["prompts"] += scores["num_problems"]
        totals["completions"] += scores["num_rows"]
        totals["correct"] += scores["num_correct"]
        totals["tokens"] += metrics["num_tokens"]
        totals["forwards"] += metrics["num_forwards"]
        request_tpfs.extend(
            row["tokens_per_forward"]
            for row in benchmark_rows
            if row["tokens_per_forward"] is not None
        )

    tokens_per_second = totals["tokens"] / generation_seconds
    tokens_per_second_per_request = tokens_per_second / args.max_running_requests
    if len(benchmark_metrics) == 1:
        only_metrics = next(iter(benchmark_metrics.values()))
        only_metrics["tokens_per_second"] = tokens_per_second
        only_metrics["tokens_per_second_per_request"] = tokens_per_second_per_request

    dataset_accuracies = [value["accuracy"] for value in benchmark_metrics.values()]
    dataset_tpfs = [
        value["tokens_per_forward"]
        for value in benchmark_metrics.values()
        if value["tokens_per_forward"] is not None
    ]
    summary = {
        "engine": "sglang-offline",
        "mode": _mode_name(args),
        "model_path": engine_options["model_path"],
        "tokenizer_path": engine_options["tokenizer_path"],
        "revision": args.revision,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "speculative_parameters": {
            name: engine_options[name]
            for name in SPECULATIVE_OPTION_NAMES
            if name in engine_options
        },
        "max_running_requests": args.max_running_requests,
        "benchmarks": list(rows_by_benchmark),
        "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "num_samples": args.num_samples,
        "random_seed": args.random_seed,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "num_prompts": totals["prompts"],
        "num_completions": totals["completions"],
        "num_correct": totals["correct"],
        "accuracy": _divide(totals["correct"], totals["completions"]),
        "macro_average_accuracy": sum(dataset_accuracies) / len(dataset_accuracies),
        "generation_seconds": generation_seconds,
        "num_tokens": totals["tokens"],
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_request": tokens_per_second_per_request,
        "num_forwards": totals["forwards"],
        "tokens_per_forward": _divide(totals["tokens"], totals["forwards"]),
        "macro_average_tokens_per_forward": sum(dataset_tpfs) / len(dataset_tpfs),
        "unweighted_mean_tokens_per_forward": (
            sum(request_tpfs) / len(request_tpfs) if request_tpfs else None
        ),
        "by_benchmark": benchmark_metrics,
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_markdown(args.output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2))


def _mode_name(args: argparse.Namespace) -> str:
    algorithm = args.speculative_algorithm
    if algorithm is None:
        return "ar"
    if algorithm == "UNO":
        return "uno-tree" if (args.speculative_eagle_topk or 1) > 1 else "uno-linear"
    return algorithm.lower()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    engine_options = _engine_options(args=args)
    tokenizer = AutoTokenizer.from_pretrained(
        engine_options["tokenizer_path"],
        use_fast=True,
        trust_remote_code=True,
    )
    prompts = _prepare_prompts(
        args=args,
        tokenizer=tokenizer,
        context_reserve=_context_reserve(args),
    )
    outputs, elapsed = _generate(
        args=args,
        prompts=prompts,
        engine_options=engine_options,
    )
    rows = _build_rows(
        prompts=prompts,
        outputs=outputs,
        tokenizer=tokenizer,
        speculative_algorithm=args.speculative_algorithm,
    )
    _write_results(
        args=args,
        rows=rows,
        generation_seconds=elapsed,
        engine_options=engine_options,
    )


if __name__ == "__main__":
    main()
